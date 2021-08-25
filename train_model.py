import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from random import choices
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
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
m_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) #wasserstein_loss(tf.ones_like(real_output), real_output)
    #cross_entropy(tf.ones_like(real_output), real_output) #wasserstein_loss(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) #wasserstein_loss(tf.ones_like(fake_output), fake_output)
    #cross_entropy(tf.zeros_like(fake_output), fake_output) #wasserstein_loss(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def gen_step_train(seq_len, batch_size, vocab_size, gen_decoder, dec_state, real_o):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, seq_len, vocab_size))
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len):
        o_token = real_o[:, t:t+1]
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=True)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        loss = m_loss(o_token, dec_result)
        step_loss += loss
        # teacher forcing, actual output as the next input
        i_token = o_token
    step_loss = step_loss / seq_len
    pred_logits = tf.convert_to_tensor(pred_logits)
    return pred_logits, gen_decoder, step_loss


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



def balance_train_dataset(x, y, x_y_l):

    lst_x = x
    lst_y = y
    #print(x_y_l)
    l_dist = x_y_l.numpy()
    u_l_dist = list(set(l_dist))
    #print(u_l_dist)

    batch_size = x.shape[0]
    
    n_samples = int(batch_size / float(len(u_l_dist)))

    #print(n_samples)

    bal_x = np.zeros((x.shape[0], x.shape[1]))
    bal_y = np.zeros((y.shape[0], y.shape[1]))

    #print(bal_x.shape, bal_y.shape)

    ctr = 0
    for l_val in u_l_dist:
        l_val_indices = np.where(l_dist == int(l_val))
        #print(l_val, l_val_indices)
        len_indices = len(l_val_indices)

        x_rows = np.array(lst_x[l_val_indices])
        y_rows = np.array(lst_y[l_val_indices])

        #print(x_rows.shape, y_rows.shape)

        rand_x_rows = np.array(choices(x_rows, k=n_samples))
        rand_y_rows = np.array(choices(y_rows, k=n_samples))

        #print(rand_x_rows.shape, rand_y_rows.shape)

        #print(ctr, ctr+n_samples)

        bal_x[ctr:ctr+n_samples, :] = rand_x_rows
        bal_y[ctr:ctr+n_samples, :] = rand_y_rows
 
        #print(bal_x) 
        #print()
        #print(bal_y)

        ctr += n_samples
        #print("---")

    bal_x = tf.convert_to_tensor(bal_x, dtype=tf.int32)
    bal_y = tf.convert_to_tensor(bal_y, dtype=tf.int32)
    
    #print(bal_x) 
    #print()
    #print(bal_y)

    return bal_x, bal_y


def start_training(inputs, encoder, decoder, par_enc_model, gen_enc_model, discriminator, enc_units, vocab_size, n_train_batches):
  input_tokens, target_tokens, input_target_l_dist = inputs  
  epo_avg_gen_loss = list()
  epo_ave_gen_true_loss = list()
  epo_avg_disc_loss = list()
  for step, (x_batch_train, y_batch_train, l_dist_batch) in enumerate(zip(input_tokens, target_tokens, input_target_l_dist)):
      unrolled_x = utils.convert_to_array(x_batch_train)
      unrolled_y = utils.convert_to_array(y_batch_train)
      
      unrolled_x, unrolled_y = balance_train_dataset(unrolled_x, unrolled_y, l_dist_batch)

      #sys.exit()
      seq_len = unrolled_x.shape[1]
      batch_size = unrolled_x.shape[0]
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          noise = tf.random.normal((batch_size, enc_units))
          # encode true parent
          enc_output, enc_state = encoder(unrolled_x, training=True)

          # set weights from the generator's encoder
          par_enc_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
          gen_enc_model.layers[1].set_weights(par_enc_model.layers[1].get_weights())

          # add noise to encoded state to have variations while generating sequences
          enc_state = tf.math.add(enc_state, noise)
          gen_loss = tf.constant(0.0)
          dec_state = enc_state

          # generate sequences
          generated_logits, decoder, gen_true_loss = gen_step_train(seq_len, batch_size, vocab_size, decoder, dec_state, unrolled_y)
          # reformat real output to one-hot encoding
          real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)
          # encode parent sequences
          par_enc_real_state_x = par_enc_model(unrolled_x, training=True)
          # encode true child sequences
          gen_real_enc_state_y = gen_enc_model(real_y, training=True)
          # encode generated child sequences
          gen_enc_fake_state_x = gen_enc_model(generated_logits, training=True)
          # discriminate pairs of true parent and generated child sequences
          fake_output = discriminator([par_enc_real_state_x, gen_enc_fake_state_x], training=True)
          # discriminate pairs of true parent and true child sequences
          real_output = discriminator([par_enc_real_state_x, gen_real_enc_state_y], training=True)
          # compute discriminator loss
          total_disc_loss = discriminator_loss(real_output, fake_output)
          # compute generator loss - sum of wasserstein and SCE losses
          gen_fake_loss = generator_loss(fake_output) #wasserstein_loss(tf.ones_like(fake_output), fake_output)
          total_gen_loss = gen_fake_loss + gen_true_loss
          print("Batch {}/{}, Generator fake loss: {}, Generator true loss: {}, Total generator loss: {}, Total discriminator loss: {}".format(str(step), str(n_train_batches), str(gen_fake_loss.numpy()), str(gen_true_loss.numpy()), str(total_gen_loss.numpy()), str(total_disc_loss.numpy())))
          epo_avg_gen_loss.append(total_gen_loss.numpy())
          epo_ave_gen_true_loss.append(gen_true_loss)
          epo_avg_disc_loss.append(total_disc_loss.numpy())
      # apply gradients
      gradients_of_generator = gen_tape.gradient(total_gen_loss, decoder.trainable_variables)
      generator_optimizer.apply_gradients(zip(gradients_of_generator, decoder.trainable_variables))
      gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
      encoder.save_weights(ENC_WEIGHTS_SAVE_PATH)
  # save model
  tf.keras.models.save_model(encoder, TRAIN_ENC_MODEL)
  tf.keras.models.save_model(decoder, TRAIN_GEN_MODEL)
  return np.mean(epo_avg_gen_loss), np.mean(epo_avg_disc_loss), np.mean(epo_ave_gen_true_loss), encoder, decoder
