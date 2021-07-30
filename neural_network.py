import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from numpy.random import randn
from tensorflow.keras import backend as K

import preprocess_sequences
import utils
import masked_loss
import sequence_to_sequence
import ArgMaxLayer


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

seq_len = 50
vocab_size = 20
latent_dim = 100
batch_size = 32
embedding_dim = 16
enc_units = 16
factor = 1
epochs = 2
n_samples = factor * batch_size


def arg_max(logits):
    return tf.math.argmax(logits, axis=-1)

def make_generator_model(vocab_size, embed_dim, enc_units, seq_len, latent_dim=100):
    

    '''model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(seq_len, )))
    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim))
    model.add(tf.keras.layers.Dense(enc_units, use_bias=False))
    model.add(tf.keras.layers.Dense(vocab_size, use_bias=False))'''
    
    '''inputs = tf.keras.Input(shape=(seq_len, ))
    embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
    gru1 = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    fc1 = tf.keras.layers.Dense(enc_units, use_bias=False)
    fc = tf.keras.layers.Dense(vocab_size, use_bias=False)
    
    em_vec = embed(inputs)
    gru_vec, _ = gru1(em_vec)
    fc1_vec = fc1(gru_vec)
    logits = fc(fc1_vec)
    #tokens = tf.math.argmax(logits, axis=-1)
    #tokens = tf.keras.layers.Lambda(arg_max)(logits)
    model = tf.keras.Model(inputs=inputs, outputs=logits)'''
    
    '''model = tf.keras.Sequential()
    #model.add(tf.keras.layers.InputLayer(input_shape=(seq_len, )))
    model.add(tf.keras.layers.Dense(seq_len * vocab_size, use_bias=False, input_shape=(latent_dim,)))
    model.add(tf.keras.layers.Reshape((seq_len, vocab_size)))
    #model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    #model.add(tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=False, recurrent_initializer='glorot_uniform'))
    #model.add(tf.keras.layers.GRU(enc_units, return_sequences=False, return_state=False, recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(enc_units, use_bias=False))
    model.add(tf.keras.layers.Dense(vocab_size, use_bias=False))
    
    return model'''
    
    # Generator Encoder model
    parent_inp = tf.keras.Input(shape=(None, seq_len))
    gen_enc_embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
    gen_enc_BiGRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units, return_state=True))
    
    enc_embed = gen_enc_embedding(parent_inp)
    enc_outputs, state_hPar, state_cPar = gen_enc_BiGRU(enc_embed)
    
    generator_encoder_model = tf.keras.Model([parent_inp], [enc_outputs, state_hPar, state_cPar])
    
    # Generator decoder model
    
    
    

def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):
    '''model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(seq_len, )))
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    model.add(tf.keras.layers.GRU(enc_units, return_sequences=False, return_state=False, recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(enc_units, use_bias=False))
    model.add(tf.keras.layers.Dense(1, use_bias=False))
    #opt = tf.keras.optimizers.Adam(1e-4)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])'''
    
    # parent seq encoder
    parent_inputs = tf.keras.Input(shape=(None,))
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    parent_inputs_embedding = enc_embedding(parent_inputs)
    enc_BiGRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units, return_state=True))
    #enc_outputs, fwd_hPar, fwd_cPar, bwd_hPar, bwd_cPar = enc_BiGRU(parent_inputs_embedding)
    enc_outputs, state_hPar, state_cPar = enc_BiGRU(parent_inputs_embedding)
    print(enc_outputs.shape, state_hPar.shape, state_cPar.shape)
    #state_hPar = #Concatenate()([fwd_hPar, bwd_hPar])
    #state_cPar= #Concatenate()([fwd_cPar, bwd_cPar])
    encoder_statePar = [state_hPar, state_cPar]
    ParentEncoder_model = tf.keras.Model([parent_inputs], encoder_statePar)
    
    # generated seq encoder
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    enc_inputsGen = tf.keras.layers.Dense(embedding_dim, activation = 'linear', use_bias = False)(gen_inputs)
    #enc_outputsGen, fwd_hGen, fwd_cGen, bwd_hGen, bwd_cGen = enc_BiGRU(enc_inputsGen)
    enc_outputsGen, state_hGen, state_cGen = enc_BiGRU(enc_inputsGen)
    #state_hGen = Concatenate()([fwd_hGen, bwd_hGen])
    #state_cGen = Concatenate()([fwd_cGen, bwd_cGen])
    encoder_stateGen = [state_hGen, state_cGen]
    GeneratorEncoder_model = tf.keras.Model([gen_inputs], encoder_stateGen) 
    
    #Load the weights for the pretrained autoencoder
    #ParentEncoder_model.load_weights('Influenza_biLSTM_encoder_model_128_4500_weights.h5')
    GeneratorEncoder_model.layers[1].set_weights(enc_embedding.get_weights())
    
    xPar = ParentEncoder_model([parent_inputs])
    xGen = GeneratorEncoder_model([gen_inputs])
    xConcat = tf.keras.layers.Concatenate()(xPar+xGen)
    x = tf.keras.layers.Dropout(0.2)(xConcat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation = 'linear')(x)
    
    disc_model = tf.keras.Model([parent_inputs, gen_inputs], [output_class])
    
    return disc_model


generator = make_generator_model(vocab_size, embedding_dim, enc_units, seq_len) 
discriminator = make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units)

print(discriminator)
sys.exit()
# do sample prediction and inference
'''input_x = [np.random.randint(vocab_size, size=seq_len) for i in range(1)]
input_x = np.array(input_x)
print(input_x.shape)

gen_seq = generator(input_x, training=False)
#gen_seq = tf.math.argmax(gen_seq, axis=-1)
print(gen_seq)
   


disc_out = discriminator(gen_seq, training=False)
print(disc_out)'''


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    #fake_output = fake_output / tf.math.reduce_max(fake_output)
    #print(fake_output)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


'''def combined_model(g_model, d_model):

    d_model.trainable = False
    model = tf.keras.Sequential()
    model.add(g_model)
    model.add(d_model)
    disc_opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(loss='binary_crossentropy', optimizer=disc_opt)
    
    return model

gan_model = combined_model(generator, discriminator)

def train_step(batch_real_x, batch_real_y, fake_target):

    gen_seq = generator(batch_real_x, training=False)
    shape_var = discriminator(batch_real_y, training=False)
    
    d_y = tf.ones_like(shape_var)
    g_y = tf.zeros_like(shape_var)
    
    X, y = np.vstack((batch_real_y, gen_seq)), np.vstack((d_y, g_y))

    disc_train = discriminator.fit(X, y)
    
    gan_model = combined_model(generator, discriminator)
    
    gan_train = gan_model.fit(batch_real_x, d_y)'''
    
@tf.function
def train_step(batch_real_x, batch_real_y, fake_target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      noise = tf.random.normal([batch_size, 100])
      fake_y = generator(noise, training=True)
      print(fake_y.shape)

      real_output = discriminator(batch_real_y, training=True)
      
      fake_output = discriminator(batch_real_x, training=True) #tf.math.argmax(fake_y, axis=-1)

      print(real_output.shape, fake_output.shape)
      #disc_loss = discriminator_loss(real_output, fake_output)
      gen_loss = generator_loss(fake_output)

      print(gen_loss)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    print(gradients_of_generator)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    #gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
