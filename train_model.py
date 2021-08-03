import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py

import utils

ENC_WEIGHTS_SAVE_PATH = "data/generated_files/generator_encoder_weights.h5"

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    

def start_training(inputs, enc_units, generator, encoder, par_enc_model, gen_enc_model, discriminator):
  input_tokens, target_tokens = inputs  
  epo_avg_gen_loss = list()
  epo_avg_disc_loss = list()
  for step, (x_batch_train, y_batch_train) in enumerate(zip(input_tokens, target_tokens)):
      unrolled_x = utils.convert_to_array(x_batch_train)
      unrolled_y = utils.convert_to_array(y_batch_train)
      (_, input_mask, _, target_mask) = _preprocess(unrolled_x, unrolled_y)
      seq_len = unrolled_x.shape[1]
      batch_size = unrolled_x.shape[0]
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

          new_tokens = tf.fill([batch_size, seq_len], 0)
          noise = tf.random.normal((batch_size, enc_units))

          generated_logits = generator([unrolled_x, new_tokens, noise], training=True)

          generated_tokens = tf.math.argmax(generated_logits, axis=-1)

          encoder.save_weights(ENC_WEIGHTS_SAVE_PATH)

          # update weights of the discriminator's encoder models
          par_enc_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
          gen_enc_model.layers[1].set_weights(par_enc_model.layers[1].get_weights())

          # reformat real output to one-hot encoding
          real_y = tf.one_hot(unrolled_y, depth=generated_logits.shape[-1], axis=-1)

          par_enc_real_state_x = par_enc_model(unrolled_x, training=True)
          gen_real_enc_state_y = gen_enc_model(real_y, training=True)
          gen_enc_fake_state_x = gen_enc_model(generated_logits, training=True)

          fake_output = discriminator([par_enc_real_state_x, gen_enc_fake_state_x], training=True)
          real_output = discriminator([par_enc_real_state_x, gen_real_enc_state_y], training=True)

          disc_loss = discriminator_loss(real_output, fake_output)

          gen_loss = generator_loss(fake_output)

          #print(gen_loss, disc_loss)
          epo_avg_gen_loss.append(gen_loss.numpy())
          epo_avg_disc_loss.append(disc_loss.numpy())

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return np.mean(epo_avg_gen_loss), np.mean(epo_avg_disc_loss), generator

def _preprocess(input_text, target_text):

  # Convert IDs to masks.
  input_mask = input_text != 0
  target_mask = target_text != 0
  return input_text, input_mask, target_text, target_mask


def _loop_step(new_tokens, input_mask, enc_output, dec_state):
  input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

  # Run the decoder one step.
  decoder_input = container_classes.DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

  dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

  # `self.loss` returns the total for non-padded tokens
  y = target_token
  y_pred = dec_result.logits
  step_loss = self.loss(y, y_pred)

  return step_loss, dec_state
