import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py

import utils

ENC_WEIGHTS_SAVE_PATH = "data/generated_files/generator_encoder_weights.h5"
PRETRAIN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_MODEL = "data/generated_files/pretrain_gen_decoder"
TRAIN_ENC_MODEL = "data/generated_files/enc_model"
TRAIN_GEN_MODEL = "data/generated_files/gen_model"

pretrain_generator_optimizer = tf.keras.optimizers.Adam(0.01)
generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(3e-5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
m_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
n_disc_extra_iter = 5
test_perf_iter = 10


'''def gradient_penalty(batch_size, real_seq, fake_seq, discriminator):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_seq - real_seq
    interpolated = real_seq + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp'''


def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.multiply(y_true, y_pred))


def discriminator_loss(real_output, fake_output):
    # loss on real parent-child sequences
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output) #-tf.math.reduce_mean(real_output) #wasserstein_loss(tf.ones_like(real_output), real_output)
    # loss on real parent and generated child sequences
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) #tf.math.reduce_mean(fake_output) #wasserstein_loss(-tf.ones_like(fake_output), fake_output)
    # loss on real sequences that are not parent-child
    real_loss = -tf.math.reduce_mean(real_output)
    fake_loss = tf.math.reduce_mean(fake_output)
    return real_loss, fake_loss


def generator_loss(fake_output):
    return -tf.math.reduce_mean(fake_output) 
    #-tf.math.reduce_mean(fake_output)
    #wasserstein_loss(tf.ones_like(fake_output), fake_output)
    #cross_entropy(tf.ones_like(fake_output), fake_output)


def gen_step_train(seq_len, batch_size, vocab_size, gen_decoder, dec_state, real_o, train_gen):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, seq_len, vocab_size))
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len):
        o_token = real_o[:, t:t+1]
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=train_gen)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        loss = m_loss(o_token, dec_result)
        step_loss += loss
        # teacher forcing, actual output as the next input
        i_token = o_token
    step_loss = step_loss / seq_len
    pred_logits = tf.convert_to_tensor(pred_logits)
    return pred_logits, gen_decoder, step_loss


def start_training(inputs, encoder, decoder, disc_par_enc_model, disc_gen_enc_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, test_data_load):
  X_train, y_train, X_y_l = inputs
  test_dataset_in, test_dataset_out = test_data_load

  # randomize batches
  rand_idx = np.random.randint(0, X_train.shape[0], X_train.shape[0])
  X_train = X_train[rand_idx]
  y_train = y_train[rand_idx]
  X_y_l = X_y_l[rand_idx]

  assert len(rand_idx) == X_train.shape[0]
  
  input_tokens = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
  target_tokens = tf.data.Dataset.from_tensor_slices((y_train)).batch(batch_size)
  input_target_l_dist = tf.data.Dataset.from_tensor_slices((X_y_l)).batch(batch_size)

  epo_avg_total_gen_loss = list()
  epo_ave_gen_true_loss = list()
  epo_avg_gen_fake_loss = list()

  epo_avg_total_disc_loss = list()
  epo_avg_disc_fake_loss = list()
  epo_avg_disc_real_loss = list()
  train_gen = True
  disc_real_loss = tf.constant(0)
  disc_fake_loss = tf.constant(0)
  total_disc_loss = tf.constant(0)
  gen_fake_loss = tf.constant(0)
  gen_true_loss = tf.constant(0)
  total_gen_loss = tf.constant(0)

  for step, (x_batch_train, y_batch_train, l_dist_batch) in enumerate(zip(input_tokens, target_tokens, input_target_l_dist)):
      unrolled_x = utils.convert_to_array(x_batch_train)
      unrolled_y = utils.convert_to_array(y_batch_train)
      # balance x and y in terms of levenshtein distance
      unrolled_x, unrolled_y = utils.balance_train_dataset(unrolled_x, unrolled_y, l_dist_batch)
      seq_len = unrolled_x.shape[1]
      
      # find performance on test data every few batches
      if step % test_perf_iter == 0:
          with tf.device('/device:cpu:0'):
              _ = utils.predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, TRAIN_ENC_MODEL, TRAIN_GEN_MODEL)

      noise = tf.random.normal((batch_size, enc_units))
      # set weights from the discriminator generator's encoder
      disc_par_enc_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
      disc_gen_enc_model.layers[1].set_weights(disc_par_enc_model.layers[1].get_weights())

      if step % n_disc_extra_iter > 0:
          with tf.GradientTape() as disc_tape:
              # encode true parent
              enc_output, enc_state = encoder(unrolled_x, training=True)
              # add noise to encoded state to have variations while generating sequences
              enc_state = tf.math.add(enc_state, noise)
              # generate sequences
              generated_logits, decoder, _ = gen_step_train(seq_len, batch_size, vocab_size, decoder, enc_state, unrolled_y, True)
              # reformat real output to one-hot encoding
              real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)
              # encode parent sequences for discriminator
              par_enc_real_state_x = disc_par_enc_model(unrolled_x, training=True)
              # encode true child sequences for discriminator
              gen_real_enc_state_y = disc_gen_enc_model(real_y, training=True)
              # encode generated child sequences for discriminator
              gen_enc_fake_state_x = disc_gen_enc_model(generated_logits, training=True)
              # discriminate pairs of true parent and true child sequences
              real_output = discriminator([par_enc_real_state_x, gen_real_enc_state_y], training=True)
              # discriminate pairs of true parent and generated child sequences
              fake_output = discriminator([par_enc_real_state_x, gen_enc_fake_state_x], training=True)
              # discriminate pairs of real sequences but not parent-child
              not_par_child_output = discriminator([par_enc_real_state_x, par_enc_real_state_x], training=True)
              # take halves of fake output - real parent and gen child and not parent-child sequences
              h_fake_output = fake_output[:int(batch_size / 2)]
              h_not_par_child_output = not_par_child_output[:int(batch_size / 2)]
              # mix both fake outputs
              merged_fake_output = tf.concat([h_fake_output, h_not_par_child_output], axis=0)
              # compute discriminator loss
              disc_real_loss, disc_fake_loss = discriminator_loss(real_output, merged_fake_output)
              total_disc_loss = disc_real_loss + disc_fake_loss

          gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
          disc_clipped_grad = [tf.clip_by_value(grad, -0.05, 0.05) for grad in gradients_of_discriminator]
          discriminator_optimizer.apply_gradients(zip(disc_clipped_grad, discriminator.trainable_variables))
          print("Applied gradient update on discriminator...")

      if step % n_disc_extra_iter == 0:
          with tf.GradientTape() as gen_tape:
              # encode true parent
              enc_output, enc_state = encoder(unrolled_x, training=True)
              # generate sequences
              generated_logits, decoder, gen_true_loss = gen_step_train(seq_len, batch_size, vocab_size, decoder, enc_state, unrolled_y, True)
              # reformat real output to one-hot encoding
              real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)

              # encode parent sequences for discriminator
              par_enc_real_state_x = disc_par_enc_model(unrolled_x, training=True)
              # encode true child sequences for discriminator
              gen_real_enc_state_y = disc_gen_enc_model(real_y, training=True)
              # encode generated child sequences for discriminator
              gen_enc_fake_state_x = disc_gen_enc_model(generated_logits, training=True)

              # discriminate pairs of true parent and true child sequences
              real_output = discriminator([par_enc_real_state_x, gen_real_enc_state_y], training=True)
              # discriminate pairs of true parent and generated child sequences
              fake_output = discriminator([par_enc_real_state_x, gen_enc_fake_state_x], training=True)
              # discriminate pairs of real sequences but not parent-child
              not_par_child_output = discriminator([par_enc_real_state_x, par_enc_real_state_x], training=True)

              # take halves of fake output - real parent and gen child and not parent-child sequences
              h_fake_output = fake_output[:int(batch_size / 2)]
              h_not_par_child_output = not_par_child_output[:int(batch_size / 2)]
              # mix both fake outputs
              merged_fake_output = tf.concat([h_fake_output, h_not_par_child_output], axis=0)

              gen_fake_loss = generator_loss(merged_fake_output)
              total_gen_loss = gen_fake_loss + gen_true_loss

          gradients_of_decoder = gen_tape.gradient(total_gen_loss, decoder.trainable_variables)
          generator_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))
          encoder.save_weights(ENC_WEIGHTS_SAVE_PATH)
          print("Applied gradient update on generator...")

      print("Batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(step), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy()), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))
      epo_ave_gen_true_loss.append(gen_true_loss.numpy())
      epo_avg_gen_fake_loss.append(gen_fake_loss.numpy())
      epo_avg_total_gen_loss.append(total_gen_loss.numpy())
      epo_avg_disc_fake_loss.append(disc_fake_loss.numpy())
      epo_avg_disc_real_loss.append(disc_real_loss.numpy())
      epo_avg_total_disc_loss.append(total_disc_loss.numpy())
      '''if step % n_disc_iter == 0:
          train_gen = not train_gen

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          noise = tf.random.normal((batch_size, enc_units))
          # encode true parent
          enc_output, enc_state = encoder(unrolled_x, training=train_gen)

          # set weights from the discriminator generator's encoder
          #if not train_gen is True:
          disc_par_enc_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
          disc_gen_enc_model.layers[1].set_weights(disc_par_enc_model.layers[1].get_weights())

          # add noise to encoded state to have variations while generating sequences
          enc_state = tf.math.add(enc_state, noise)

          # generate sequences
          generated_logits, decoder, gen_true_loss = gen_step_train(seq_len, batch_size, vocab_size, decoder, enc_state, unrolled_y, train_gen)
          # reformat real output to one-hot encoding
          real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)

          # encode parent sequences for discriminator
          par_enc_real_state_x = disc_par_enc_model(unrolled_x, training=not train_gen)
          # encode true child sequences for discriminator
          gen_real_enc_state_y = disc_gen_enc_model(real_y, training=not train_gen)
          # encode generated child sequences for discriminator
          gen_enc_fake_state_x = disc_gen_enc_model(generated_logits, training=not train_gen)

          # discriminate pairs of true parent and true child sequences
          real_output = discriminator([par_enc_real_state_x, gen_real_enc_state_y], training=not train_gen)
          # discriminate pairs of true parent and generated child sequences
          fake_output = discriminator([par_enc_real_state_x, gen_enc_fake_state_x], training=not train_gen)
          # discriminate pairs of real sequences but not parent-child
          not_par_child_output = discriminator([par_enc_real_state_x, par_enc_real_state_x], training=not train_gen)

          # take halves of fake output - real parent and gen child and not parent-child sequences
          h_fake_output = fake_output[:int(batch_size / 2)]
          h_not_par_child_output = not_par_child_output[:int(batch_size / 2)]
          # mix both fake outputs
          merged_fake_output = tf.concat([h_fake_output, h_not_par_child_output], axis=0)

          # compute discriminator loss
          disc_real_loss, disc_fake_loss = discriminator_loss(real_output, merged_fake_output)
          total_disc_loss = disc_real_loss + disc_fake_loss

          # compute generator loss
          gen_fake_loss = generator_loss(fake_output)
          total_gen_loss = gen_fake_loss + gen_true_loss
          print("Batch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(step), str(n_train_batches), str(gen_true_loss.numpy()), str(gen_fake_loss.numpy()), str(total_gen_loss.numpy()), str(disc_real_loss.numpy()), str(disc_fake_loss.numpy()), str(total_disc_loss.numpy())))

          epo_ave_gen_true_loss.append(gen_true_loss.numpy())
          epo_avg_gen_fake_loss.append(gen_fake_loss.numpy())
          epo_avg_total_gen_loss.append(total_gen_loss.numpy())

          epo_avg_disc_fake_loss.append(disc_fake_loss.numpy())
          epo_avg_disc_real_loss.append(disc_real_loss.numpy())
          epo_avg_total_disc_loss.append(total_disc_loss.numpy())
      # apply gradients
      # train discriminator more that generator - 5 times discriminator, 1 time generator 
      if train_gen == True:
          encoder.save_weights(ENC_WEIGHTS_SAVE_PATH)
          gradients_of_decoder = gen_tape.gradient(total_gen_loss, decoder.trainable_variables)
          generator_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))
          print("Applied gradient update on generator...")
      elif train_gen == False:
          gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
          disc_clipped_grad = [tf.clip_by_value(grad, -0.05, 0.05) for grad in gradients_of_discriminator]
          discriminator_optimizer.apply_gradients(zip(disc_clipped_grad, discriminator.trainable_variables))
          print("Applied gradient update on discriminator...")'''
      print()
  # save model
  tf.keras.models.save_model(encoder, TRAIN_ENC_MODEL)
  tf.keras.models.save_model(decoder, TRAIN_GEN_MODEL)
  return np.mean(epo_ave_gen_true_loss), np.mean(epo_avg_gen_fake_loss), np.mean(epo_avg_total_gen_loss), np.mean(epo_avg_disc_real_loss), np.mean(epo_avg_disc_fake_loss), np.mean(epo_avg_total_disc_loss), encoder, decoder


def pretrain_generator(inputs, gen_encoder, gen_decoder, enc_units, vocab_size, n_batches):
  input_tokens, target_tokens = inputs  
  epo_avg_gen_loss = list()
  for step, (x_batch_train, y_batch_train) in enumerate(zip(input_tokens, target_tokens)):
      unrolled_x = utils.convert_to_array(x_batch_train)
      unrolled_y = utils.convert_to_array(y_batch_train)
      seq_len = unrolled_x.shape[1]
      batch_size = unrolled_x.shape[0]
      with tf.GradientTape() as gen_tape:
          noise = tf.random.normal((batch_size, enc_units))
          enc_output, enc_state = gen_encoder(unrolled_x, training=True)
          enc_state = tf.math.add(enc_state, noise)
          dec_state = enc_state
          gen_logits, gen_decoder, gen_loss = gen_step_train(seq_len, batch_size, vocab_size, gen_decoder, dec_state, unrolled_y)
          print("Pretrain Generator batch {}/{} step loss: {}".format(str(step), str(n_batches), str(gen_loss)))
          epo_avg_gen_loss.append(gen_loss)
      gradients_of_generator = gen_tape.gradient(gen_loss, gen_decoder.trainable_variables)
      pretrain_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_decoder.trainable_variables))
  # save model
  gen_encoder.save_weights(ENC_WEIGHTS_SAVE_PATH)
  tf.keras.models.save_model(gen_encoder, PRETRAIN_ENC_MODEL)
  tf.keras.models.save_model(gen_decoder, PRETRAIN_GEN_MODEL)
  return np.mean(epo_avg_gen_loss), gen_encoder, gen_decoder
