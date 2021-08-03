import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import glob
import tensorflow as tf

import matplotlib.pyplot as plt

import preprocess_sequences
import utils
import neural_network
import train_model



PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out_19A_20A.json" #"data/clade_in_clade_out.json"
embedding_dim = 8
batch_size = 16
units = 64
epochs = 10
LEN_AA = 1275

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    '''samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    vocab_size = utils.embedding_info(forward_dict)'''
    
    #print("Transforming generated samples...")
    #train_x, train_y, test_x, test_y = preprocess_sequences.transform_encoded_samples()
    
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    '''train_x = np.array([list(map(int, lst)) for lst in train_x])
    train_y = [list(map(int, lst)) for lst in train_y]

    test_x = [list(map(int, lst)) for lst in test_x]
    test_y = [list(map(int, lst)) for lst in test_y]'''
    
    #print(train_x)
    
    #print("Reading in/out sequences...")
    # = preprocess_sequences.read_in_out_sequences()
    
    '''vocab_size, seq_len = utils.embedding_info(forward_dict, train_samples)
    
    # get train datasets
    train_x = train_samples["Sequence_x"].to_numpy()
    train_y = train_samples["Sequence_y"].to_numpy()

    train_x = [list(map(int, lst)) for lst in train_x]
    train_y = [list(map(int, lst)) for lst in train_y]
    
    print(train_x, train_y)
    
    # get test datasets
    test_x = test_samples["Sequence_x"].to_numpy()
    test_y = test_samples["Sequence_y"].to_numpy()

    test_x = [list(map(int, lst)) for lst in test_x]
    test_y = [list(map(int, lst)) for lst in test_y]
    
    print(test_x, test_y)

    print("Creating neural network...")
    
    factor = 100
    
    train_x = [np.random.randint(vocab_size, size=seq_len) for i in range(factor * batch_size)]
    train_x = np.array(train_x)
    print(train_x.shape)

    train_y = [np.random.randint(vocab_size, size=seq_len) for i in range(factor * batch_size)]
    train_y = np.array(train_y)
    print(train_y.shape)
    
    dataset_in = tf.data.Dataset.from_tensor_slices((train_x)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((train_y)).batch(batch_size)'''
    
    seq_len = 50
    vocab_size = 20
    latent_dim = 100
    batch_size = 32
    embedding_dim = 16
    enc_units = 16
    factor = 1
    epochs = 1
    n_samples = factor * batch_size
    
    train_real_x = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_x = np.array(train_real_x)
    print(train_real_x.shape)
    
    train_real_y = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_y = np.array(train_real_y)
    print(train_real_y.shape)

    start_training(train_real_x, train_real_y, embedding_dim, units, batch_size, vocab_size, seq_len)


def start_training(train_real_x, train_real_y, embedding_dim, units, batch_size, vocab_size, seq_len):
    
    seq_len = 50
    vocab_size = 20
    latent_dim = 100
    batch_size = 32
    embedding_dim = 16
    enc_units = 32
    factor = 5
    epochs = 3

    generator, encoder = neural_network.make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size)
    
    parent_encoder_model, gen_encoder_model = neural_network.make_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units)
    
    discriminator = neural_network.make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units)
    
    print("Start training ...")
    
    dataset_in = tf.data.Dataset.from_tensor_slices((train_real_x)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((train_real_y)).batch(batch_size)

    for n in range(epochs):
        print("Training epoch {}...".format(str(n+1)))
        train_model.start_training([dataset_in, dataset_out], generator, encoder, parent_encoder_model, gen_encoder_model, discriminator)
    
    
    '''tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')
    
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"]
        y = tr_clade_df["Y"]
        print(tr_clade_df.shape)
        dataset_in = tf.data.Dataset.from_tensor_slices((X)).batch(batch_size)
        dataset_out = tf.data.Dataset.from_tensor_slices((y)).batch(batch_size)
        for n in range(epochs):
            print("Training epoch {}...".format(str(n+1)))
            batch_learning = model.train_step([dataset_in, dataset_out])
            encoder = batch_learning['encoder']
            decoder = batch_learning['decoder']
            print("Training loss at step {}: {}".format(str(n+1), str(np.round(batch_learning["epo_loss"], 8))))
            for te_name in te_clade_files:
                te_clade_df = pd.read_csv(te_name, sep="\t")
                te_X = te_clade_df["X"]
                te_y = te_clade_df["Y"]
                #print(te_clade_df.shape)
                #print("Prediction on test data...")
                #predict_sequence(te_X, te_y, encoder, decoder, model, LEN_AA, vocab_size, batch_size)'''

def predict_sequence(test_x, test_y, encoder, decoder, model, seq_len, vocab_size, batch_size):
    avg_test_loss = []
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    i = 0
    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
    
        batch_x_test = utils.convert_to_array(x)
        batch_y_test = utils.convert_to_array(y)
        #print(batch_x_test)
        #print()
        #print(batch_y_test)
        #print("-----")
        #print(batch_x_test.shape, batch_y_test.shape)
        
        if batch_x_test.shape[0] == batch_size:
        
            enc_output, enc_state = encoder(batch_x_test, training=False)
        
            #print(enc_output.shape)
        
            input_mask = batch_x_test != 0
            target_mask = batch_y_test != 0
            new_tokens = tf.fill([batch_size, seq_len], 0)

            #print(new_tokens.shape)

            #decoder_input = container_classes.DecoderInput(new_tokens=new_tokens, enc_output=enc_output, mask=input_mask)
            logits, dec_state = decoder(new_tokens, state=enc_state, training=False)
        
            #print(logits.shape, dec_state.shape)

            if step == 5:
                print("Test: Sample 0, batch {}".format(str(step)))
                print(batch_y_test[0])
                print(tf.argmax(logits, axis=-1)[0])
                print(model.loss(batch_y_test[0], logits[0]))
        
            # compute loss
            y = batch_y_test
            y_pred = logits
            #pred_tokens = tf.argmax(y_pred, axis=-1)
            loss = model.loss(y, y_pred)
        
            #print(target_mask.shape)
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        
            real_loss = average_loss.numpy()
            #print("Batch {} loss: {}".format(str(i), str(real_loss)))
            avg_test_loss.append(real_loss)
            i += 1
    print("Total test loss: {}".format(str(np.mean(avg_test_loss))))
    print()


if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))

