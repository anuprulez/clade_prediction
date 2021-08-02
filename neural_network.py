import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py

import preprocess_sequences
import sequence_to_sequence
import utils


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size):

    #encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, enc_units)
    
    gen_inputs = tf.keras.Input(shape=(seq_len,))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    gen_gru = tf.keras.layers.GRU(enc_units, go_backwards=True, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    
    inputs = tf.keras.Input(shape=(seq_len,))
    embed = gen_embedding(gen_inputs)
    gen_output, gen_state = gen_gru(embed)
    
    encoder_model = tf.keras.Model([gen_inputs], [gen_output, gen_state])
    
    enc_output, enc_state = encoder_model(inputs, training=True)
    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, enc_units)


    ## TODO Add noise to enc_state to allow generation of multiple child sequences from a parent sequence
    noise = tf.keras.Input(shape=(enc_units,))
    
    #enc_state = tf.math.add(enc_state, noise)
    new_tokens = tf.keras.Input(shape=(seq_len,))
    
    logits = decoder(new_tokens, state=enc_state, training=True)
    
    gen_model = tf.keras.Model([inputs, new_tokens, noise], [logits])
    
    encoder_model.save_weights('data/generated_files/generator_encoder_weights.h5')
    return gen_model, encoder_model


def make_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units):

    #Load the weights for the pretrained autoencoder
    #ParentEncoder_model.load_weights('Influenza_biLSTM_encoder_model_128_4500_weights.h5')
    #GeneratorEncoder_model.layers[1].set_weights(enc_embedding.get_weights())
    
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(None,))
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    enc_GRU = tf.keras.layers.GRU(enc_units, go_backwards=True,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    parent_inputs_embedding = enc_embedding(parent_inputs)
    enc_outputs, enc_state = enc_GRU(parent_inputs_embedding)
    ParentEncoder_model = tf.keras.Model([parent_inputs], [enc_state])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    enc_inputsGen = tf.keras.layers.Dense(embedding_dim, activation='linear', use_bias=False)(gen_inputs)
    enc_outputsGen, stateGen = enc_GRU(enc_inputsGen)
    encoder_stateGen = [stateGen]
    GeneratorEncoder_model = tf.keras.Model([gen_inputs], encoder_stateGen)

    ParentEncoder_model.load_weights('data/generated_files/generator_encoder_weights.h5')
    GeneratorEncoder_model.layers[1].set_weights(enc_embedding.get_weights())
    
    return ParentEncoder_model, GeneratorEncoder_model


def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):   
    #xPar = ParentEncoder_model([parent_inputs])
    #xGen = GeneratorEncoder_model([gen_inputs])
    xPar = tf.keras.Input(shape=(None, enc_units))
    xGen = tf.keras.Input(shape=(None, enc_units))
    
    xConcat = tf.keras.layers.Concatenate()([xPar+xGen])
    x = tf.keras.layers.Dropout(0.2)(xConcat)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation='linear')(x)
    
    disc_model = tf.keras.Model([xPar, xGen], [output_class]) # parent_inputs, gen_inputs
    
    return disc_model
