import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py
import tensorflow_addons as tfa
import typing
from typing import Any, Tuple

import preprocess_sequences
import bahdanauAttention


ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):
    #shape_checker = ShapeChecker()
    #shape_checker(y_true, ('batch', 't'))
    #shape_checker(y_pred, ('batch', 't', 'logits'))

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)
    #shape_checker(loss, ('batch', 't'))

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    #shape_checker(mask, ('batch', 't'))
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)


class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any

class DecoderOutput(typing.NamedTuple):
  logits: Any
  attention_weights: Any


class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    #shape_checker = ShapeChecker()
    #shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    vectors = tf.keras.layers.Dropout(ENC_DROPOUT)(vectors)
    vectors = tf.keras.layers.BatchNormalization()(vectors)
    #shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The GRU processes the embedding sequence.
    #    output shape: (batch, s, enc_units)
    #    state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    #shape_checker(output, ('batch', 's', 'enc_units'))
    #shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the new sequence and its state.
    return output, state


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super().__init__()
    # For Eqn. (4), the  Bahdanau attention
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(units, use_bias=False)

    self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value, mask):
    #shape_checker = ShapeChecker()
    #shape_checker(query, ('batch', 't', 'query_units'))
    #shape_checker(value, ('batch', 's', 'value_units'))
    #shape_checker(mask, ('batch', 's'))

    # From Eqn. (4), `W1@ht`.
    w1_query = self.W1(query)
    w1_query = tf.keras.layers.Dropout(ENC_DROPOUT)(w1_query)
    w1_query = tf.keras.layers.BatchNormalization()(w1_query)
    #shape_checker(w1_query, ('batch', 't', 'attn_units'))

    # From Eqn. (4), `W2@hs`.
    w2_key = self.W2(value)
    w2_key = tf.keras.layers.Dropout(ENC_DROPOUT)(w2_key)
    w2_key = tf.keras.layers.BatchNormalization()(w2_key)
    #shape_checker(w2_key, ('batch', 's', 'attn_units'))

    query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
    value_mask = mask

    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value, w2_key],
        mask=[query_mask, value_mask],
        return_attention_scores = True,
    )
    #shape_checker(context_vector, ('batch', 't', 'value_units'))
    #shape_checker(attention_weights, ('batch', 't', 's'))

    return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim

    # For Step 1. The embedding layer convets token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)

    # For Step 2. The RNN keeps track of what's been generated so far.
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # For step 3. The RNN output will be the query for the attention layer.
    self.attention = BahdanauAttention(self.dec_units)

    # For step 4. Eqn. (3): converting `ct` to `at`
    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                    use_bias=False)

    # For step 5. This fully connected layer produces the logits for each
    # output token.
    self.fc = tf.keras.layers.Dense(self.output_vocab_size)

def call(self,
         inputs: DecoderInput,
         state=None) -> Tuple[DecoderOutput, tf.Tensor]:
  #shape_checker = ShapeChecker()
  #shape_checker(inputs.new_tokens, ('batch', 't'))
  #shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
  #shape_checker(inputs.mask, ('batch', 's'))

  #if state is not None:
  #  shape_checker(state, ('batch', 'dec_units'))

  # Step 1. Lookup the embeddings
  vectors = self.embedding(inputs.new_tokens)
  vectors = tf.keras.layers.Dropout(ENC_DROPOUT)(vectors)
  vectors = tf.keras.layers.BatchNormalization()(vectors)
  #shape_checker(vectors, ('batch', 't', 'embedding_dim'))

  # Step 2. Process one step with the RNN
  rnn_output, state = self.gru(vectors, initial_state=state)
  rnn_output = tf.keras.layers.Dropout(ENC_DROPOUT)(rnn_output)
  rnn_output = tf.keras.layers.BatchNormalization()(rnn_output)
  #shape_checker(rnn_output, ('batch', 't', 'dec_units'))
  #shape_checker(state, ('batch', 'dec_units'))

  # Step 3. Use the RNN output as the query for the attention over the
  # encoder output.
  context_vector, attention_weights = self.attention(
      query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
  #shape_checker(context_vector, ('batch', 't', 'dec_units'))
  #shape_checker(attention_weights, ('batch', 't', 's'))
  context_vector = tf.keras.layers.Dropout(ENC_DROPOUT)(context_vector)
  # Step 4. Eqn. (3): Join the context_vector and rnn_output
  #     [ct; ht] shape: (batch t, value_units + query_units)
  context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
  context_and_rnn_output = tf.keras.layers.Dropout(ENC_DROPOUT)(context_and_rnn_output)
  context_and_rnn_output = tf.keras.layers.BatchNormalization()(context_and_rnn_output)
  # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
  attention_vector = self.Wc(context_and_rnn_output)
  #shape_checker(attention_vector, ('batch', 't', 'dec_units'))
  attention_vector = tf.keras.layers.Dropout(ENC_DROPOUT)(attention_vector)
  attention_vector = tf.keras.layers.BatchNormalization()(attention_vector)
  # Step 5. Generate logit predictions:
  logits = self.fc(attention_vector)
  #shape_checker(logits, ('batch', 't', 'output_vocab_size'))

  return DecoderOutput(logits, attention_weights), state

Decoder.call = call
