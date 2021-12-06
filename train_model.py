import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import random
from random import choices
import logging
import tensorflow as tf
import h5py
import copy

import encoder_decoder_attention

import utils

PATH_KMER_F_DICT = "data/ncov_global/kmer_f_word_dictionaries.json"
PRE_TR_GEN_ENC_WEIGHTS = "data/generated_files/pretr_generator_encoder_weights.h5"
PRE_TR_GEN_DEC_WEIGHTS = "data/generated_files/pretr_generator_decoder_weights.h5"
PRETRAIN_GEN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_DEC_MODEL = "data/generated_files/pretrain_gen_decoder"

DISC_WEIGHTS = "data/generated_files/disc_weights.h5"
DISC_PAR_ENC_WEIGHTS = "data/generated_files/disc_par_enc_weights.h5"
DISC_GEN_ENC_WEIGHTS = "data/generated_files/disc_gen_enc_weights.h5"

GEN_ENC_WEIGHTS = "data/generated_files/generator_encoder_weights.h5"
GEN_DEC_WEIGHTS = "data/generated_files/generator_decoder_weights.h5"
TRAIN_GEN_ENC_MODEL = "data/generated_files/gen_enc_model"
TRAIN_GEN_DEC_MODEL = "data/generated_files/gen_dec_model"


pretrain_generator_optimizer = tf.keras.optimizers.Adam() #tf.keras.optimizers.Adam() # learning_rate=1e-3, beta_1=0.5
generator_optimizer = tf.keras.optimizers.Adam() # learning_rate=1e-3, beta_1=0.5
discriminator_optimizer = tf.keras.optimizers.Adam() # learning_rate=3e-5, beta_1=0.5
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
n_disc_step = 2
n_gen_step = 1
unrolled_steps = 1
test_log_step = 10
teacher_forcing_ratio = 0.5

m_loss = encoder_decoder_attention.MaskedLoss()
mae = tf.keras.losses.MeanAbsoluteError()


def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.multiply(y_true, y_pred))


def discriminator_loss(real_output, fake_output):
    #real_loss = -tf.math.reduce_mean(real_output)
    #fake_loss = tf.math.reduce_mean(fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # loss on real parent and generated child sequences
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss, fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    #return -tf.math.reduce_mean(fake_output)


def get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model):
    noise = tf.random.normal((batch_size, 2 * enc_units))
    #transformed_noise = utils.transform_noise(noise)
    _, enc_state = encoder(unrolled_x, training=True)
    # add noise to encoded state to have variations while generating sequences
    transformed_enc_state = tf.math.add(enc_state, noise)
    # generate sequences
    generated_logits, decoder, gen_t_loss = utils.generator_step(seq_len, batch_size, vocab_size, decoder, transformed_enc_state, unrolled_y, True)
    # compute generated sequence variation
    variation_score = utils.get_sequence_variation_percentage(generated_logits)
    print("Generation variation score: {}".format(str(variation_score)))
    # encode parent sequences for discriminator
    real_state_x = disc_par_enc_model(unrolled_x, training=True)
    # unrelated real X
    unrelated_real_state_x = disc_par_enc_model(un_X, training=True)
    # encode true child sequences for discriminator
    # reformat real output to one-hot encoding
    one_hot_real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)
    real_state_y = disc_gen_enc_model(one_hot_real_y, training=True)

    # unrelated real y
    one_hot_unrelated_y = tf.one_hot(un_y, depth=generated_logits.shape[-1], axis=-1)
    unrelated_real_state_y = disc_gen_enc_model(one_hot_unrelated_y, training=True)

    # encode generated child sequences for discriminator
    fake_state_y = disc_gen_enc_model(generated_logits, training=True)
    return real_state_x, real_state_y, fake_state_y, unrelated_real_state_x, unrelated_real_state_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, gen_t_loss


def d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator):
    print("Applying gradient update on discriminator...")
    with tf.GradientTape() as disc_tape:
        real_x, real_y, fake_y, unreal_x, unreal_y, _, _, disc_par_enc, disc_gen_enc, _ = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc)
        # discriminate pairs of true parent and true child sequences
        real_output = discriminator([real_x, real_y], training=True)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        # discriminate pairs of true parent and random sequences
        unreal_output = discriminator([unreal_x, unreal_y], training=True)
        # halve the fake outpus and combine them to keep the final size same as the real output
        t1 = fake_output[:int(batch_size/2.0)]
        t2 = unreal_output[:int(batch_size/2.0)]
        combined_fake_output = tf.concat([t1, t2], 0)
        # compute discriminator loss
        disc_real_loss, disc_fake_loss = discriminator_loss(real_output, combined_fake_output)
        total_disc_loss = disc_real_loss + disc_fake_loss
    # update discriminator's parameters
    disc_trainable_vars = discriminator.trainable_variables + disc_gen_enc.trainable_variables + disc_par_enc.trainable_variables
    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, disc_trainable_vars)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss


def g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator):
    print("Applying gradient update on generator...")
    with tf.GradientTape() as gen_tape:
        real_x, _, fake_y, _, _, encoder, decoder, _, _, gen_true_loss = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        gen_fake_loss = generator_loss(fake_output)
        total_gen_loss = gen_fake_loss + gen_true_loss
    # get all trainable vars for generator
    gen_trainable_vars = decoder.trainable_variables + encoder.trainable_variables
    gradients_of_generator = gen_tape.gradient(total_gen_loss, gen_trainable_vars)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, gen_true_loss, gen_fake_loss, total_gen_loss


def sample_true_x_y(mut_indices, batch_size, X_train, y_train, batch_mut_distribution):
    '''mut_keys = list(mut_indices.keys())
    rand_mut_keys = np.array(choices(mut_keys, k=batch_size))
    x_batch_train = list()
    y_batch_train = list()
    rand_batch_indices = list()
    for key in rand_mut_keys:
        list_mut_rows = mut_indices[key]
        rand_row_index = np.random.randint(0, len(list_mut_rows), 1)[0]
        rand_batch_indices.append(list_mut_rows[rand_row_index])'''

    rand_batch_indices = np.random.randint(0, X_train.shape[0], batch_size)
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]

    batch_mut_distribution = utils.save_batch(x_batch_train, y_batch_train, batch_mut_distribution)

    unrolled_x = utils.convert_to_array(x_batch_train)
    unrolled_y = utils.convert_to_array(y_batch_train)
    return unrolled_x, unrolled_y, batch_mut_distribution


def _loop_step(input_token, target_token, input_mask, enc_output, dec_state, gen_decoder):
  #input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

  # Run the decoder one step.
  decoder_input = encoder_decoder_attention.DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

  dec_result, dec_state = gen_decoder(decoder_input, state=dec_state, training=True)
  #self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
  #self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
  #self.shape_checker(dec_state, ('batch', 'dec_units'))

  # `self.loss` returns the total for non-padded tokens
  y = target_token
  y_pred = dec_result.logits
  
  step_loss = m_loss(y, y_pred)

  return step_loss, dec_state, y_pred

def check_batch_ldist(u_x, u_y, seq_len):
    x_s = ",".join([str(j) for j in u_x[:, 1:]])
    y_s = ",".join([str(k) for k in u_y[:, 1:]])

    l_list = list()
    for i in range(seq_len - 1):
        l_dist = utils.compute_Levenshtein_dist(x_s[i], y_s[i])
        l_list.append(l_dist)
    print(l_list, np.mean(l_dist))

    print("--")

def pretrain_generator(inputs, epo_step, gen_encoder, gen_decoder, enc_units, vocab_size, n_batches, batch_size, pretr_parent_child_mut_indices, epochs, size_stateful):
  X_train, y_train, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches = inputs
  epo_avg_tr_gen_loss = list()
  epo_te_gen_loss = list()
  epo_tr_seq_var = list()
  epo_te_seq_var = list()
  batch_mut_distribution = dict()

  #kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
  #print(kmer_f_dict)

  for step in range(n_batches):
      loss = tf.constant(0.0)
      unrolled_x, unrolled_y, batch_mut_distribution = sample_true_x_y(pretr_parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)
      seq_len = unrolled_x.shape[1]
      '''
      # verify levenshtein distance
      for i in range(len(unrolled_x)):
          re_x = utils.reconstruct_seq([kmer_f_dict[str(pos)] for pos in unrolled_x[i][1:]])
          re_y = utils.reconstruct_seq([kmer_f_dict[str(pos)] for pos in unrolled_y[i][1:]])
          l_dist = utils.compute_Levenshtein_dist(re_x, re_y)
          print(re_x)
          print(re_y)
          print(l_dist)
          print("---")
      import sys
      sys.exit()

      check_batch_ldist(unrolled_x, unrolled_y, seq_len)'''
      
      with tf.GradientTape() as gen_tape:

          enc_output, enc_state = gen_encoder(unrolled_x, training=True)
          enc_state = tf.math.add(enc_state, tf.random.normal((batch_size, enc_units)))
          dec_state = enc_state
          input_mask = unrolled_x != 0
          target_mask = unrolled_y != 0
          gen_logits = list()
          i_tokens = tf.fill([batch_size, 1], 0)
          for t in tf.range(seq_len-1):
              o_tokens = unrolled_y[:, t+1:t+2]
              step_loss, dec_state, dec_logits = _loop_step(i_tokens, o_tokens, input_mask, enc_output, dec_state, gen_decoder)
              if random.random() <= teacher_forcing_ratio:
                  i_tokens = o_tokens
              else:
                  i_tokens = tf.argmax(dec_logits, axis=-1)
              gen_logits.append(dec_logits)
              loss = loss + step_loss

          pred_logits = tf.concat(gen_logits, axis=-2)
          gen_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

          #noise = tf.random.normal((batch_size, enc_units))
 
          '''enc_hidden = gen_encoder.initialize_hidden_state()
          enc_output, enc_h, enc_c = gen_encoder(unrolled_x, enc_hidden)

          enc_h = tf.math.add(enc_h, tf.random.normal((batch_size, enc_units)))
          enc_c = tf.math.add(enc_c, tf.random.normal((batch_size, enc_units)))

          gen_decoder.attention_mechanism.setup_memory(enc_output)
          decoder_initial_state = gen_decoder.build_initial_state(batch_size, [enc_h, enc_c], tf.float32)
          pred = gen_decoder(unrolled_y, decoder_initial_state)
          
          gen_logits = pred.rnn_output
          gen_loss = utils.loss_function(unrolled_y, gen_logits)'''

 

          '''print("Training: true output seq")
          print(unrolled_y)
          print()
          print(tf.argmax(gen_logits, axis=-1))'''

          #enc_out, enc_state_h, enc_state_c = gen_encoder(unrolled_x, training=True)

          # stateful encoding
          '''enc_out, enc_state_h, enc_state_c = utils.stateful_encoding(size_stateful, unrolled_x, gen_encoder, True)
          gen_encoder.reset_states()
          enc_state_h = tf.math.add(enc_state_h, noise)
          enc_state_c = tf.math.add(enc_state_c, noise)
          dec_state_h, dec_state_c = enc_state_h, enc_state_c
          gen_logits, gen_decoder, gen_loss = utils.generator_step(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, unrolled_x, unrolled_y, True)
          gen_decoder.reset_states()'''
          '''print("Training: true output seq")
          #print(unrolled_x)
          #print()
          print(unrolled_y)'''
          # compute generated sequence variation
          variation_score = utils.get_sequence_variation_percentage(pred_logits)
          print("Pretr: generation variation score: {}".format(str(variation_score)))
          gen_loss = gen_loss / variation_score #+ mae([1.0], [variation_score])
          epo_tr_seq_var.append(variation_score)
          print("Pretrain epoch {}/{}, batch {}/{}, gen true loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches), str(gen_loss.numpy())))
          if (step + 1) % test_log_step == 0 and step > 0:
              print("-------")
              print("Pretr: Prediction on test data at epoch {}/{}, batch {}/{}...".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches)))
              print()
              #with tf.device('/device:cpu:0'):
              gen_te_loss, gen_te_seq_var = utils.predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, gen_encoder, gen_decoder, size_stateful)
              #utils.evaluate_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, gen_encoder, gen_decoder)
              #utils.predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, gen_encoder, gen_decoder, size_stateful)
              epo_te_gen_loss.append(gen_te_loss)
              epo_te_seq_var.append(gen_te_seq_var)
              print("-------")
          print()
          epo_avg_tr_gen_loss.append(gen_loss)
      gen_trainable_vars = gen_encoder.trainable_variables + gen_decoder.trainable_variables
      gradients_of_generator = gen_tape.gradient(gen_loss, gen_trainable_vars)
      pretrain_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
  # save model
  #gen_encoder.save_weights(GEN_ENC_WEIGHTS)
  #tf.keras.models.save_model(gen_encoder, PRETRAIN_GEN_ENC_MODEL)
  #tf.keras.models.save_model(gen_decoder, PRETRAIN_GEN_DEC_MODEL)
  #gen_encoder.save_weights(PRE_TR_GEN_ENC_WEIGHTS)
  #gen_decoder.save_weights(PRE_TR_GEN_DEC_WEIGHTS)
  utils.save_as_json("data/generated_files/pretr_ave_batch_x_y_mut_epo_{}.json".format(str(epo_step)), batch_mut_distribution)
  return np.mean(epo_avg_tr_gen_loss), np.mean(epo_te_gen_loss), np.mean(epo_te_seq_var), np.mean(epo_tr_seq_var), gen_encoder, gen_decoder


def start_training_mut_balanced(inputs, epo_step, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, enc_units, vocab_size, n_train_batches, batch_size, parent_child_mut_indices, epochs):
  """
  Training sequences balanced by mutation type
  """
  X_train, y_train, unrelated_X, unrelated_y, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches = inputs

  epo_avg_total_gen_loss = list()
  epo_ave_gen_true_loss = list()
  epo_avg_gen_fake_loss = list()

  epo_avg_total_disc_loss = list()
  epo_avg_disc_fake_loss = list()
  epo_avg_disc_real_loss = list()
  disc_real_loss = tf.constant(0)
  disc_fake_loss = tf.constant(0)
  total_disc_loss = tf.constant(0)
  gen_fake_loss = tf.constant(0)
  gen_true_loss = tf.constant(0)
  total_gen_loss = tf.constant(0)
  batch_mut_distribution = dict()

  epo_te_gen_loss = list()
  epo_te_seq_var = list()

  mut_keys = list(parent_child_mut_indices.keys())
  for step in range(n_train_batches):
      unrolled_x, unrolled_y, batch_mut_distribution = sample_true_x_y(parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)
      un_X, un_y = utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
      seq_len = unrolled_x.shape[1]
      disc_gen = step % n_disc_step
      if disc_gen in list(range(0, n_disc_step - n_gen_step)):
          # train discriminator
          _, _, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss = d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator)
          # share weights with generator's encoder
          disc_par_enc.load_weights(GEN_ENC_WEIGHTS)
          disc_gen_enc.layers[1].set_weights(disc_par_enc.layers[1].get_weights())
          print("Training epoch {}/{}, batch {}/{}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))
      else:
          # train generator with unrolled discriminator
          # save disc weights to reset after unrolling
          discriminator.save_weights(DISC_WEIGHTS)
          disc_par_enc.save_weights(DISC_PAR_ENC_WEIGHTS)
          disc_gen_enc.save_weights(DISC_GEN_ENC_WEIGHTS)
          print("Applying unrolled steps...")
          # unrolling steps
          for i in range(unrolled_steps):
              print("Unrolled step: {}/{}".format(str(i+1), str(unrolled_steps)))
              # sample data for unrolling
              unroll_x, unroll_y, _ = sample_true_x_y(parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)
              un_unroll_X, un_unroll_y = utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
              # train discriminator
              _, _, disc_par_enc, disc_gen_enc, discriminator, d_r_l, d_f_l, d_t_l = d_loop(seq_len, batch_size, vocab_size, enc_units, unroll_x, unroll_y, un_unroll_X, un_unroll_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator)
              print("Unrolled disc losses: real {}, fake {}, total {}".format(str(d_r_l.numpy()), str(d_f_l.numpy()), str(d_t_l.numpy())))
          # finish unrolling
          # train generator with unrolled discriminator
          encoder, decoder, _, _, _, gen_true_loss, gen_fake_loss, total_gen_loss = g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator)
          print("Training epoch {}/{}, batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy())))
          encoder.save_weights(GEN_ENC_WEIGHTS)
          if step % test_log_step == 0:
              print("Training: prediction on test data...")
              with tf.device('/device:cpu:0'):
                  epo_bat_gen_te_loss, gen_bat_te_seq_var = utils.predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, encoder, decoder)
                  epo_te_gen_loss.append(epo_bat_gen_te_loss)
                  epo_te_seq_var.append(gen_bat_te_seq_var)
          # reset weights of discriminator, disc_par_enc and disc_gen_enc after unrolling
          discriminator.load_weights(DISC_WEIGHTS)
          disc_par_enc.load_weights(DISC_PAR_ENC_WEIGHTS)
          disc_gen_enc.load_weights(DISC_GEN_ENC_WEIGHTS)
      print("Training epoch {}/{}, batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy()), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))
      # write off results
      epo_ave_gen_true_loss.append(gen_true_loss.numpy())
      epo_avg_gen_fake_loss.append(gen_fake_loss.numpy())
      epo_avg_total_gen_loss.append(total_gen_loss.numpy())
      epo_avg_disc_fake_loss.append(disc_fake_loss.numpy())
      epo_avg_disc_real_loss.append(disc_real_loss.numpy())
      epo_avg_total_disc_loss.append(total_disc_loss.numpy())
  # save model
  print("Training epoch {} finished, Saving model...".format(str(epo_step+1)))
  print()
  tf.keras.models.save_model(encoder, TRAIN_GEN_ENC_MODEL)
  tf.keras.models.save_model(decoder, TRAIN_GEN_DEC_MODEL)
  encoder.save_weights(GEN_ENC_WEIGHTS)
  decoder.save_weights(GEN_DEC_WEIGHTS)
  utils.save_as_json("data/generated_files/ave_batch_x_y_mut_epo_{}.json".format(str(epo_step)), batch_mut_distribution)
  return np.mean(epo_ave_gen_true_loss), np.mean(epo_avg_gen_fake_loss), np.mean(epo_avg_total_gen_loss), np.mean(epo_avg_disc_real_loss), np.mean(epo_avg_disc_fake_loss), np.mean(epo_avg_total_disc_loss), np.mean(epo_te_gen_loss), np.mean(epo_te_seq_var), encoder, decoder
