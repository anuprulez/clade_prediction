import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import typing
from typing import Any, Tuple

import preprocess_sequences
import utils
import shape_check
import battention
import container_classes


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out.json"
      

class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   go_backwards=True,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None, training=False):
    vectors = self.embedding(tokens)
    output, state = self.gru(vectors, initial_state=state, training=training)

    return output, state


class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   go_backwards=True,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                    use_bias=False)
    self.fc = tf.keras.layers.Dense(self.output_vocab_size, use_bias=False)


def call(self, inputs, state=None, training=False):
    vectors = self.embedding(inputs)
    rnn_output, state = self.gru(vectors, initial_state=state, training=training)
    attention_vector = self.Wc(rnn_output)
    logits = self.fc(attention_vector)
    return logits, state

Decoder.call = call
