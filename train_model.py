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
import glob


import neural_network
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


pretrain_generator_optimizer = tf.keras.optimizers.Adam()
generator_optimizer = tf.keras.optimizers.Adam() # learning_rate=1e-3, beta_1=0.5
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # learning_rate=3e-5, beta_1=0.5
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
n_disc_step = 10
n_gen_step = 5
unrolled_steps = 5
test_log_step = 50
teacher_forcing_ratio = 0.0


m_loss = neural_network.MaskedLoss()
mae = tf.keras.losses.MeanAbsoluteError()


def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.multiply(y_true, y_pred))


def discriminator_loss(real_output, fake_output):
    real_loss = -tf.math.reduce_mean(real_output)
    fake_loss = tf.math.reduce_mean(fake_output)
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # loss on real parent and generated child sequences
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss, fake_loss


def generator_loss(fake_output):
    #return cross_entropy(tf.ones_like(fake_output), fake_output)
    return -tf.math.reduce_mean(fake_output)


def get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, size_stateful):
    #noise = tf.random.normal((batch_size, 2 * enc_units))
    #transformed_noise = utils.transform_noise(noise)
    #_, enc_state = encoder(unrolled_x, training=True)
    # add noise to encoded state to have variations while generating sequences
    #transformed_enc_state = tf.math.add(enc_state, noise)
    # generate sequences
    #generated_logits, decoder, gen_t_loss = utils.generator_step(seq_len, batch_size, vocab_size, decoder, transformed_enc_state, unrolled_y, True)
    # compute generated sequence variation

    generated_logits, encoder, decoder, gen_t_loss = utils.loop_encode_decode(seq_len, batch_size, vocab_size, unrolled_x, unrolled_y, encoder, decoder, enc_units, teacher_forcing_ratio, True, size_stateful)

    '''stateful_batches = list()
    n_stateful_batches = int(unrolled_x.shape[1]/float(size_stateful))
    for i in range(n_stateful_batches):
        s_batch = unrolled_x[:, i*size_stateful: (i+1)*size_stateful]
        enc_output, x_enc_f, x_enc_b = encoder(s_batch, training=True)

    #enc_output, x_enc_f, x_enc_b, encoder = utils.stateful_encoding(size_stateful, unrolled_x, encoder, True)
    #dec_f, dec_b = x_enc_f, x_enc_f

    
    #print()
    #print(enc_output.shape, enc_f.shape, enc_b.shape)
    #if train_test is True:
    #noise_generator = tf.random.Generator.from_non_deterministic_state()
    #dec_f = tf.math.add(dec_f, noise_generator.normal(shape=[batch_size, enc_units]))
    #dec_b = tf.math.add(dec_b, noise_generator.normal(shape=[batch_size, enc_units]))

    target_mask = unrolled_y != 0
    i_tokens = tf.fill([batch_size, seq_len], 0)
    generated_logits, _, _ = decoder([i_tokens, x_enc_f, x_enc_b], training=True)

    loss = m_loss(unrolled_y, generated_logits)
    gen_t_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))'''

    # loop_encode_decode(seq_len, batch_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units, tf_ratio, train_test, s_stateful):
    # return gen_logits, gen_encoder, gen_decoder, loss
    variation_score = utils.get_sequence_variation_percentage(unrolled_x, generated_logits)
    print("Generation variation score: {}".format(str(variation_score)))

    gen_tokens = tf.argmax(generated_logits, axis=-1)
    print("True output")
    print(unrolled_y[:5, 1:])
    print()
    print("Gen output")
    print(gen_tokens[:5, :])
    print()
    #gen_t_loss = gen_t_loss + mae([1.0], [variation_score])
    # encode parent sequences for discriminator
    #enc_output, enc_f, enc_b = utils.stateful_encoding(size_stateful, unrolled_x, encoder, True)
    #enc_output, enc_f, enc_b = stateful_encoding(s_stateful, input_tokens, gen_encoder, train_test)
    #real_state_x = disc_par_enc_model(unrolled_x, training=True)

    _, real_enc_f_x, real_enc_b_x = disc_par_enc_model(unrolled_x, training=True) #utils.stateful_encoding(size_stateful, unrolled_x, disc_par_enc_model, True)
    real_state_x = real_enc_f_x + real_enc_b_x
    #print(real_state_x.shape, real_enc_f_x.shape, real_enc_b_x.shape)
    # unrelated real X
    unrelated_real_state_x = [] #disc_par_enc_model(un_X, training=True)
    # encode true child sequences for discriminator
    # reformat real output to one-hot encoding
    one_hot_real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)
    #real_state_y = disc_gen_enc_model(one_hot_real_y, training=True)

    _, real_enc_f_y, real_enc_b_y = disc_gen_enc_model(one_hot_real_y, training=True) #utils.stateful_encoding(size_stateful, one_hot_real_y, disc_gen_enc_model, True)
    real_state_y = real_enc_f_y + real_enc_b_y
    #print(real_state_y.shape, real_enc_f_y.shape, real_enc_b_y.shape)
    # unrelated real y
    one_hot_unrelated_y = [] #tf.one_hot(un_y, depth=generated_logits.shape[-1], axis=-1)
    unrelated_real_state_y = [] #disc_gen_enc_model(one_hot_unrelated_y, training=True)

    # encode generated child sequences for discriminator
    #fake_state_y = disc_gen_enc_model(generated_logits, training=True)
    _, fake_enc_f_y, fake_enc_b_y = disc_gen_enc_model(generated_logits, training=True) #utils.stateful_encoding(size_stateful, generated_logits, disc_gen_enc_model, True)
    fake_state_y = fake_enc_f_y + fake_enc_b_y
    #print(fake_state_y.shape, fake_enc_f_y.shape, fake_enc_b_y.shape)
    return real_state_x, real_state_y, fake_state_y, unrelated_real_state_x, unrelated_real_state_y, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, gen_t_loss


def d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful):
    print("Applying gradient update on discriminator...")
    with tf.GradientTape() as disc_tape:
        real_x, real_y, fake_y, unreal_x, unreal_y, _, _, disc_par_enc, disc_gen_enc, _ = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, size_stateful)
        # discriminate pairs of true parent and true child sequences
        real_output = discriminator([real_x, real_y], training=True)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        # discriminate pairs of true parent and random sequences
        '''unreal_output = discriminator([unreal_x, unreal_y], training=True)
        # halve the fake outpus and combine them to keep the final size same as the real output
        t1 = fake_output[:int(batch_size/2.0)]
        t2 = unreal_output[:int(batch_size/2.0)]
        combined_fake_output = tf.concat([t1, t2], 0)'''
        # compute discriminator loss
        disc_real_loss, disc_fake_loss = discriminator_loss(real_output, fake_output)
        total_disc_loss = disc_real_loss + disc_fake_loss
    # update discriminator's parameters
    print()
    disc_trainable_vars = discriminator.trainable_variables + disc_gen_enc.trainable_variables + disc_par_enc.trainable_variables
    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, disc_trainable_vars)
    gradients_norm = [tf.norm(gd) for gd in gradients_of_discriminator]
    print("Train Disc gradient norm: {}".format(str(tf.reduce_mean(gradients_norm).numpy())))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss


def g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful):
    print("Applying gradient update on generator...")
    with tf.GradientTape() as gen_tape:
        real_x, _, fake_y, _, _, encoder, decoder, _, _, gen_true_loss = get_par_gen_state(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, size_stateful)
        # discriminate pairs of true parent and generated child sequences
        fake_output = discriminator([real_x, fake_y], training=True)
        gen_fake_loss = generator_loss(fake_output)
        total_gen_loss = gen_true_loss #+ gen_fake_loss
    print()
    # get all trainable vars for generator
    gen_trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients_of_generator = gen_tape.gradient(total_gen_loss, gen_trainable_vars)
    gradients_norm = [tf.norm(gd) for gd in gradients_of_generator]
    print("Train Gen gradient norm: {}".format(str(tf.reduce_mean(gradients_norm).numpy())))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
    return encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, gen_true_loss, gen_fake_loss, total_gen_loss


def sample_true_x_y(mut_indices, batch_size, X_train, y_train, batch_mut_distribution):
    mut_keys = list(mut_indices.keys())
    rand_mut_keys = np.array(choices(mut_keys, k=batch_size))
    x_batch_train = list()
    y_batch_train = list()
    rand_batch_indices = list()
    for key in rand_mut_keys:
        list_mut_rows = mut_indices[key]
        rand_row_index = np.random.randint(0, len(list_mut_rows), 1)[0]
        rand_batch_indices.append(list_mut_rows[rand_row_index])
    #rand_batch_indices = np.random.randint(0, X_train.shape[0], batch_size)
    #print(rand_mut_keys)
    x_batch_train = X_train[rand_batch_indices]
    y_batch_train = y_train[rand_batch_indices]

    batch_mut_distribution = utils.save_batch(x_batch_train, y_batch_train, batch_mut_distribution)

    unrolled_x = utils.convert_to_array(x_batch_train)
    unrolled_y = utils.convert_to_array(y_batch_train)
    return unrolled_x, unrolled_y, batch_mut_distribution


'''def _loop_step(input_token, target_token, input_mask, enc_output, dec_state, gen_decoder):
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

  return step_loss, dec_state, y_pred'''


def redraw_unique(x, y):
    x_str = utils.convert_to_string_list(x)
    y_str = utils.convert_to_string_list(y)
    u_x = list(set(x_str))
    u_y = list(set(y_str))
    print(len(u_x), len(u_y))
    return u_x, u_y


def get_mut_size(mut_dist):
    pos_size = dict()
    for mut in mut_dist:
        size = mut_dist[mut]
        pos = mut.split(">")[1]
        if pos not in pos_size:
            pos_size[pos] = 0
        pos_size[pos] += len(size)
    return pos_size


def get_text_data():
    tr_file = glob.glob("data/ncov_global/data_from_colab/train/*.csv")
    te_file = glob.glob("data/ncov_global/data_from_colab/test/*.csv")

    path_train_x = "data/ncov_global/data_from_colab/train/train_x.csv"
    path_train_y = "data/ncov_global/data_from_colab/train/train_y.csv"
    path_test_x = "data/ncov_global/data_from_colab/test/test_x.csv"
    path_test_y = "data/ncov_global/data_from_colab/test/test_y.csv"

    train_x = pd.read_csv(path_train_x, sep=",", header=None)
    train_y = pd.read_csv(path_train_y, sep=",", header=None)
    test_x = pd.read_csv(path_test_x, sep=",", header=None)
    test_y = pd.read_csv(path_test_y, sep=",", header=None)

    train_x[0] = 0
    train_y[0] = 0
    test_x[0] = 0
    test_y[0] = 0
    print(train_x)
    print(train_y)
    return train_x, train_y, test_x, test_y


def pretrain_generator(inputs, epo_step, gen_encoder, gen_decoder, enc_units, vocab_size, n_batches, batch_size, pretr_parent_child_mut_indices, epochs, size_stateful, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict):
  X_train, y_train, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches = inputs
  epo_avg_tr_gen_loss = list()
  epo_te_gen_loss = list()
  epo_tr_seq_var = list()
  epo_te_seq_var = list()
  batch_mut_distribution = dict()

  X_train, y_train, test_dataset_in, test_dataset_out = get_text_data()

  X_train = np.array(X_train.values.tolist())
  y_train = np.array(y_train.values.tolist())

  test_dataset_in = np.array(test_dataset_in.values.tolist())
  test_dataset_out = np.array(test_dataset_out.values.tolist())

  take_pos = np.where(X_train > vocab_size)
  u_pos = take_pos[0]
  X_train = np.delete(X_train, u_pos, 0)
  y_train = np.delete(y_train, u_pos, 0)

  #print(X_train.shape, y_train.shape)
  
  take_pos_test = np.where(test_dataset_in >= vocab_size)
  u_pos_test = take_pos_test[0]
  #print(u_pos)
  test_dataset_in = np.delete(test_dataset_in, u_pos_test, 0)
  test_dataset_out = np.delete(test_dataset_out, u_pos_test, 0)
  
  #print(test_dataset_in.shape, test_dataset_out.shape)
  #import sys
  #sys.exit()
  
  input_seq_len = X_train.shape[1]
  output_seq_len = y_train.shape[1]

  n_batches = 310 #int(X_train.shape[0] / float(batch_size))

  X_train = X_train[:n_batches * batch_size, :]
  y_train = y_train[:n_batches * batch_size, :]

  
  
  print(X_train.shape, y_train.shape)

  X_train = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
  y_train = tf.data.Dataset.from_tensor_slices((y_train)).batch(batch_size)
  test_dataset_in = tf.data.Dataset.from_tensor_slices((test_dataset_in)).batch(batch_size)
  test_dataset_out = tf.data.Dataset.from_tensor_slices((test_dataset_out)).batch(batch_size)
  #vocab_size = 5000
  #input_seq_len = 12

  pos_size = get_mut_size(pretr_parent_child_mut_indices)
  for step, (unrolled_x, unrolled_y) in enumerate(zip(X_train, y_train)):
  #for step in range(n_batches):
      #unrolled_x, unrolled_y, batch_mut_distribution = sample_true_x_y(pretr_parent_child_mut_indices, batch_size, X_train, y_train, batch_mut_distribution)

      '''str_x = [",".join(str(pos) for pos in item) for item in unrolled_x]
      str_y = [",".join(str(pos) for pos in item) for item in unrolled_y]
      #print(str_x, str_y)
      muts = utils.get_mutation_tr_indices(str_x, str_y, kmer_f_dict, kmer_r_dict, forward_dict, rev_dict)
      print(kmer_f_dict)
      print(muts)
      print(unrolled_x)
      print()
      print(unrolled_y)
      print()'''

      #unrolled_x, unrolled_y = redraw_unique(X_train, y_train)
      seq_len = unrolled_x.shape[1]
      #output_seq_len = unrolled_y.shape[1]
      # verify levenshtein distance
      '''for i in range(len(unrolled_x)):
          re_x = utils.reconstruct_seq([kmer_f_dict[str(pos)] for pos in unrolled_x[i][1:]])
          re_y = utils.reconstruct_seq([kmer_f_dict[str(pos)] for pos in unrolled_y[i][1:]])
          #l_dist = utils.compute_Levenshtein_dist(re_x, re_y)
          print(re_x)
          print(re_y)
          #print(l_dist)
          print("---")'''
      '''import sys
      sys.exit()'''
      #print(pos_size)
      with tf.GradientTape() as gen_tape:
          
          pred_logits, gen_encoder, gen_decoder, gen_loss = utils.loop_encode_decode(seq_len, batch_size, vocab_size, unrolled_x, unrolled_y, gen_encoder, gen_decoder, enc_units, teacher_forcing_ratio, True, size_stateful, pos_size)
          print("Training: true output seq")
          print(unrolled_y[:5, :], unrolled_y.shape)
          print()
          print(tf.argmax(pred_logits, axis=-1)[:5, :], pred_logits.shape)

          # compute generated sequence variation
          variation_score = utils.get_sequence_variation_percentage(unrolled_x, pred_logits)
          print("Pretr: generation variation score: {}".format(str(variation_score)))
          #/ variation_score #+ mae([1.0], [variation_score])
          #var_score = mae([1.0], [variation_score])
          #if variation_score < 1.0:
          #gen_loss = gen_loss + mae([1.0], [variation_score]) #+ mae([1.0], [variation_score])
          epo_tr_seq_var.append(variation_score)
          print("Pretr: teacher forcing ratio: {}".format(str(teacher_forcing_ratio)))
          print("Pretrain epoch {}/{}, batch {}/{}, gen true loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches), str(gen_loss.numpy())))
          '''if (step + 1) % test_log_step == 0 and step > 0:
              print("-------")
              print("Pretr: Prediction on test data at epoch {}/{}, batch {}/{}...".format(str(epo_step+1), str(epochs), str(step+1), str(n_batches)))
              print()
              gen_te_loss, gen_te_seq_var = utils.predict_sequence(epo_step, step, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, gen_encoder, gen_decoder, size_stateful)
              epo_te_gen_loss.append(gen_te_loss)
              epo_te_seq_var.append(gen_te_seq_var)
              print("-------")
          print()'''
          epo_avg_tr_gen_loss.append(gen_loss)
      gen_trainable_vars = gen_encoder.trainable_variables + gen_decoder.trainable_variables
      gradients_of_generator = gen_tape.gradient(gen_loss, gen_trainable_vars)
      #gradients_of_generator = [tf.clip_by_value(grad, clip_value_min=-1e-6, clip_value_max=1e-6) for grad in gradients_of_generator]
      gradients_norm = [tf.norm(gd) for gd in gradients_of_generator]
      print("Pretrain gradient norm: {}".format(str(tf.reduce_mean(gradients_norm).numpy())))
      #gradients_of_generator = [(tf.clip_by_norm(grad, clip_norm=1.0)) for grad in gradients_of_generator]
      pretrain_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_vars))
  # save model

  #gen_encoder.save_weights(GEN_ENC_WEIGHTS)
  #tf.keras.models.save_model(gen_encoder, PRETRAIN_GEN_ENC_MODEL)
  #tf.keras.models.save_model(gen_decoder, PRETRAIN_GEN_DEC_MODEL)
  #gen_encoder.save_weights(PRE_TR_GEN_ENC_WEIGHTS)
  #gen_decoder.save_weights(PRE_TR_GEN_DEC_WEIGHTS)
  #utils.save_as_json("data/generated_files/pretr_ave_batch_x_y_mut_epo_{}.json".format(str(epo_step)), batch_mut_distribution)
  return np.mean(epo_avg_tr_gen_loss), np.mean(epo_te_gen_loss), np.mean(epo_te_seq_var), np.mean(epo_tr_seq_var), gen_encoder, gen_decoder


def start_training_mut_balanced(inputs, epo_step, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, enc_units, vocab_size, n_train_batches, batch_size, parent_child_mut_indices, epochs, size_stateful):
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
      un_X, un_y = [], [] #utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
      seq_len = unrolled_x.shape[1]
      disc_gen = step % n_disc_step
      if disc_gen in list(range(0, n_disc_step - n_gen_step)):
          # train discriminator
          _, _, disc_par_enc, disc_gen_enc, discriminator, disc_real_loss, disc_fake_loss, total_disc_loss = d_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful)
          # share weights with generator's encoder
          disc_par_enc.load_weights(GEN_ENC_WEIGHTS)
          disc_gen_enc.load_weights(GEN_ENC_WEIGHTS)
          #disc_gen_enc.layers[1].set_weights(disc_par_enc.layers[1].get_weights())
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
              un_unroll_X, un_unroll_y = [], [] #utils.sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size)
              # train discriminator
              _, _, disc_par_enc, disc_gen_enc, discriminator, d_r_l, d_f_l, d_t_l = d_loop(seq_len, batch_size, vocab_size, enc_units, unroll_x, unroll_y, un_unroll_X, un_unroll_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful)
              print("Unrolled disc losses: real {}, fake {}, total {}".format(str(d_r_l.numpy()), str(d_f_l.numpy()), str(d_t_l.numpy())))
          # finish unrolling
          # train generator with unrolled discriminator
          encoder, decoder, _, _, _, gen_true_loss, gen_fake_loss, total_gen_loss = g_loop(seq_len, batch_size, vocab_size, enc_units, unrolled_x, unrolled_y, un_X, un_y, encoder, decoder, disc_par_enc, disc_gen_enc, discriminator, size_stateful)
          print("Training epoch {}/{}, batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}".format(str(epo_step+1), str(epochs), str(step+1), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy())))
          encoder.save_weights(GEN_ENC_WEIGHTS)
          # reset weights of discriminator, disc_par_enc and disc_gen_enc after unrolling
          discriminator.load_weights(DISC_WEIGHTS)
          disc_par_enc.load_weights(DISC_PAR_ENC_WEIGHTS)
          disc_gen_enc.load_weights(DISC_GEN_ENC_WEIGHTS)
      # intermediate prediction on test data while training
      if (step + 1) % test_log_step == 0 and step > 0:
          print("Training: prediction on test data...")
          with tf.device('/device:cpu:0'):
              _, _ = utils.predict_sequence(epo_step, step, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, encoder, decoder, size_stateful)
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
