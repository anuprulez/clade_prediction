import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from numpy.random import randn

import preprocess_sequences
import utils
import masked_loss
import sequence_to_sequence


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out.json"

v_size = 17000
embedding_dim = 8
seq_len = 1275




class Generator(tf.keras.Model):

  def __init__(self, vocab_size, embed_dim, enc_units, seq_len):
    super(Generator, self).__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.enc_units = enc_units
    self.seq_len = seq_len
    
    self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
    self.gru1 = tf.keras.layers.GRU(enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    '''self.gru2 = tf.keras.layers.GRU(enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')    '''                          
    self.fc1 = tf.keras.layers.Dense(enc_units, activation=tf.math.tanh,
                                    use_bias=False)
    
    self.fc = tf.keras.layers.Dense(vocab_size)
                    

  def call(self, inputs, training=False):
      #print(real_x.shape)
      vectors = self.embed(inputs)
      #print(vectors.shape)
      enc_output, _ = self.gru1(vectors, initial_state=None)

      # generate random output seq
      #random_vector = self.embed(fake_y)
      #print(random_vector.shape)
      # Step 2. Process one step with the RNN
      # TODO Check decoder RNN
      #rnn_output, dec_state = self.gru2(random_vector, initial_state=enc_state)
      #print(rnn_output.shape)
      fc1_vector = self.fc1(enc_output)
      #print(fc1_vector.shape)
      #logits = self.fc(fc1_vector)
      #print(logits.shape)
      return self.fc(fc1_vector)


def make_generator_model(vocab_size, embed_dim, enc_units, seq_len, real_x, fake_y):
    #model = Generator(vocab_size, embed_dim, enc_units, seq_len)
    inputs = tf.keras.Input(shape=(seq_len, ))
    
    embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
    gru1 = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
                                   
                                   
    #gru2 = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')                         
    fc1 = tf.keras.layers.Dense(enc_units, activation=tf.math.tanh, use_bias=False)
    fc = tf.keras.layers.Dense(vocab_size)

    vectors = embed(inputs)
    enc_output, _ = gru1(vectors, initial_state=None)

    
    #print(rnn_output.shape)
    fc1_vector = fc1(enc_output)
    #print(fc1_vector.shape)
    logits = fc(fc1_vector)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="encoder_decoder")
   
    return model


def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):
    model = tf.keras.Sequential()
    
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    gru_1 = tf.keras.layers.GRU(enc_units, return_sequences=False, return_state=False, recurrent_initializer='glorot_uniform')
    fc_1 = tf.keras.layers.Dense(enc_units, activation=tf.math.tanh, use_bias=False)
    fc = tf.keras.layers.Dense(1)
    
    model.add(embedding)
    model.add(gru_1)
    model.add(fc_1)
    model.add(fc)
    optimizer = tf.optimizers.Adam()
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=optimizer)

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def train_step(batch_real_x, batch_real_y, batch_fake_y, batch_size, seq_len, vocab_size, generator, discriminator):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_seq = generator(batch_real_x, training=True)
        
        fake_tokens = tf.argmax(generated_seq, axis=-1)
        
        real_output = discriminator(batch_real_x, training=True)

        fake_output = discriminator(fake_tokens, training=True)

        gen_loss = generator_loss(fake_output)
        
        gen_loss = tf.reduce_mean(gen_loss)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    print(gradients_of_generator)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
   
    
    
