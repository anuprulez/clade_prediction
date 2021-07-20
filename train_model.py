import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf

import preprocess_sequences
import utils
import sequence_to_sequence
import container_classes


class TrainModel(tf.keras.Model):
  def __init__(self, embedding_dim, units,
               vocab_size,
               use_tf_function=True):
    super().__init__()
    # Build the encoder and decoder
    encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, units)
    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, units)

    self.vocab_size = vocab_size
    self.encoder = encoder
    self.decoder = decoder
    self.use_tf_function = use_tf_function

  def train_step(self, inputs):
    return self._train_step(inputs)
      
      
def _preprocess(self, input_text, target_text):

  # Convert IDs to masks.
  input_mask = input_text != 0
  target_mask = target_text != 0
  return input_text, input_mask, target_text, target_mask


def _train_step(self, inputs):
  input_tokens, target_tokens = inputs  

  epo_avg_loss = 0.0
  for step, (x_batch_train, y_batch_train) in enumerate(zip(input_tokens, target_tokens)):
      (_, input_mask, _, target_mask) = self._preprocess(x_batch_train, y_batch_train)
      max_target_length = x_batch_train.shape[1]
      
      with tf.GradientTape() as tape:
          # Encode the input
          enc_output, enc_state = self.encoder(x_batch_train)
          # Initialize the decoder's state to the encoder's final state.
          # This only works if the encoder and decoder have the same number of
          # units.
          dec_state = enc_state
          loss = tf.constant(0.0)
          
          # Run the decoder one step.
          decoder_input = container_classes.DecoderInput(new_tokens=y_batch_train, enc_output=enc_output, mask=input_mask)

          dec_result, dec_state = self.decoder(decoder_input, state=enc_state)
          
          y = y_batch_train
          y_pred = dec_result.logits
          print(y.shape, y_pred.shape)
          loss = self.loss(y, y_pred)

          # Average the loss over all non padding tokens.
          average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
          
          #pred_tokens = tf.argmax(dec_result.logits, axis=-1)
          
          # Apply an optimization step
          variables = self.trainable_variables 
          gradients = tape.gradient(average_loss, variables)
          self.optimizer.apply_gradients(zip(gradients, variables))
          epo_avg_loss += average_loss.numpy()

  # Return a dict mapping metric names to current value
  return {'epo_loss': epo_avg_loss / (step + 1) }
  

'''def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
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

  return step_loss, dec_state'''

TrainModel._preprocess = _preprocess
TrainModel._train_step = _train_step
#TrainModel._loop_step = _loop_step
