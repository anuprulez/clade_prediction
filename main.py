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
import sequence_to_sequence
import container_classes
import train_model
import masked_loss



PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out_19A_20A.json" #"data/clade_in_clade_out.json"
embedding_dim = 64
batch_size = 64
units = 128
epochs = 20
LEN_AA = 1275

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    vocab_size = utils.embedding_info(forward_dict)
    
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

    start_training(embedding_dim, units, batch_size, vocab_size)

def start_training(embedding_dim, units, batch_size, vocab_size):
    
    model = train_model.TrainModel(
        embedding_dim, units,
        vocab_size,
        use_tf_function=False
    )

    # Configure the loss and optimizer
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=masked_loss.MaskedLoss(),
    )
  
    '''te_factor = 1
    te_batch_size = 1

    print("Generating test data...")
    test_x = [np.random.randint(vocab_size, size=seq_len) for i in range(te_factor * te_batch_size)]
    test_x = np.array(test_x)

    test_y = [np.random.randint(vocab_size, size=seq_len) for i in range(te_factor * te_batch_size)]
    test_y = np.array(test_y)
    print(test_x.shape, test_y.shape)'''
    
    print("Start training ...")  
    
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')
    
    for name in tr_clade_files:
        clade_df = pd.read_csv(name, sep="\t")
        X = clade_df["X"]
        y = clade_df["Y"]
        print(clade_df.shape)
        dataset_in = tf.data.Dataset.from_tensor_slices((X)).batch(batch_size)
        dataset_out = tf.data.Dataset.from_tensor_slices((y)).batch(batch_size)
        print(dataset_in)
        print(dataset_out)

        for n in range(epochs):
            print("Training epoch {}...".format(str(n+1)))
            batch_learning = model.train_step([dataset_in, dataset_out])
            print("Training loss at step {}: {}".format(str(n+1), str(np.round(batch_learning["epo_loss"], 8))))
            for te_name in te_clade_files:
                te_clade_df = pd.read_csv(te_name, sep="\t")
                te_X = clade_df["X"]
                te_y = clade_df["Y"]
                #print(te_clade_df.shape)
                print("Prediction on test data...")
                predict_sequence(te_X, te_y, model, LEN_AA, vocab_size, batch_size)

def predict_sequence(test_x, test_y, model, seq_len, vocab_size, batch_size):
    avg_test_loss = []
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    i = 0
    for x, y in zip(test_dataset_in, test_dataset_out):
    
        batch_x_test = utils.convert_to_array(x)
        batch_y_test = utils.convert_to_array(y)
        print(batch_x_test)
        print()
        print(batch_y_test)
        print("-----")
        #print(batch_x_test.shape, batch_y_test.shape)
        
        if batch_x_test.shape[0] == batch_size:
        
            enc_output, enc_state = model.encoder(batch_x_test)
        
            #print(enc_output.shape)
        
            input_mask = batch_x_test != 0
            target_mask = batch_y_test != 0
            new_tokens = tf.fill([batch_size, seq_len], 0)

            #print(new_tokens.shape)

            #decoder_input = container_classes.DecoderInput(new_tokens=new_tokens, enc_output=enc_output, mask=input_mask)
            logits, dec_state = model.decoder(new_tokens, state=enc_state)
        
            #print(logits.shape, dec_state.shape)
        
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



if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))

