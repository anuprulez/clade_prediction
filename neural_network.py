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


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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
    self.gru2 = tf.keras.layers.GRU(enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')                              
    self.fc1 = tf.keras.layers.Dense(enc_units, activation=tf.math.tanh,
                                    use_bias=False)
    
    self.fc = tf.keras.layers.Dense(vocab_size)
                    

  def call(self, real_x, fake_y, training=True):
      #print(real_x.shape)
      vectors = self.embed(real_x)
      #print(vectors.shape)
      enc_output, enc_state = self.gru1(vectors, initial_state=None)
      #print(enc_output.shape)
      
      # generate random output seq
      random_vector = self.embed(fake_y)
      #print(random_vector.shape)
      # Step 2. Process one step with the RNN
      # TODO Check decoder RNN
      rnn_output, dec_state = self.gru2(random_vector, initial_state=enc_state)
      #print(rnn_output.shape)
      fc1_vector = self.fc1(rnn_output)
      #print(fc1_vector.shape)
      logits = self.fc(fc1_vector)
      #print(logits.shape)
      return logits, dec_state


def make_generator_model(vocab_size, embed_dim, enc_units, seq_len, real_x, fake_y):
    model = Generator(vocab_size, embed_dim, enc_units, seq_len)
    '''inputs = tf.keras.Input(shape=(None, seq_len))
    
    embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
    gru1 = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
                                   
                                   
    gru2 = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')                         
    fc1 = tf.keras.layers.Dense(enc_units, activation=tf.math.tanh, use_bias=False)
    fc = tf.keras.layers.Dense(vocab_size)

    vectors = embed(real_x)
    enc_output, enc_state = gru1(vectors, initial_state=None)

    # generate random output seq
    random_vector = embed(fake_y)
    # Step 2. Process one step with the RNN
    # TODO Check decoder RNN
    rnn_output, dec_state = gru2(random_vector, initial_state=enc_state)
    #print(rnn_output.shape)
    fc1_vector = fc1(rnn_output)
    #print(fc1_vector.shape)
    logits = fc(fc1_vector)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="encoder_decoder")
    model.summary()'''
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


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)   


def train_step(batch_real_x, batch_real_y, batch_fake_y, batch_size, seq_len, vocab_size, generator, discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_seq, gen_state = generator(batch_real_x, batch_fake_y, training=True)
        print(dir(generator))
        print(generator.trainable)
        gen_tokens = tf.argmax(generated_seq, axis=-1)
        real_output = discriminator(batch_real_x, training=True)

        fake_output = discriminator(gen_tokens, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print(gen_loss, disc_loss)
        #print(generator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    

'''def embedding(vocab_size, out_dim, seq_len):
    embed = tf.keras.layers.Embedding(vocab_size, out_dim, mask_zero=True)
    return embed


def make_generator_model(seq_len, v_size, latent_dim=100):
    model = tf.keras.Sequential()
    #model.add(embedding(v_size, out_size, seq_len))
    #model.add(tf.keras.layers.Embedding(v_size, embedding_dim))
    model.add(tf.keras.layers.InputLayer(input_shape=latent_dim))
    #model.add(tf.keras.layers.GRU(64, return_sequences=False, activation="elu"))
    #model.add(tf.keras.layers.GRU(64, return_sequences=True, activation="elu"))
    model.add(tf.keras.layers.Dense(64))
    #model.add(tf.keras.layers.Dense(64, activation="elu"))
    #model.add(tf.keras.layers.Reshape((seq_len, v_size)))
    
    #assert model.output_shape == (None, seq_len, 1, v_size)
    #model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.GRU(64))
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(v_size, activation='sigmoid'))
    optimizer = tf.optimizers.Adam()
    loss = masked_loss.MaskedLoss(),
    model.compile(loss=loss, optimizer=optimizer)

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
    loss = 'binary_crossentropy' #masked_loss.MaskedLoss(),
    model.compile(loss=loss, optimizer=optimizer)

    return model
    
    
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
    
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))'''
    
    
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

    for image_batch in dataset:
        train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
    #    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
 
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

