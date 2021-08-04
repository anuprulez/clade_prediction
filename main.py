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
from sklearn.model_selection import train_test_split

import preprocess_sequences
import utils
import neural_network
import train_model
import masked_loss



PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out_19A_20A.json" #"data/clade_in_clade_out.json"
PRETRAIN_GEN_LOSS = "data/generated_files/pretr_gen_loss.txt"
TRAIN_GEN_LOSS = "data/generated_files/tr_gen_loss.txt"
TRAIN_DISC_LOSS = "data/generated_files/tr_disc_loss.txt"
TEST_LOSS = "data/generated_files/te_loss.txt"
embedding_dim = 256
batch_size = 128
enc_units = 128
pretrain_epochs = 10
epochs = 20
LEN_AA = 1273

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    vocab_size = utils.embedding_info(forward_dict)

    start_training(vocab_size)


def start_training(vocab_size):
    
    '''
    # code snippet with random samples
    seq_len = 50
    vocab_size = 20
    batch_size = 32
    embedding_dim = 16
    enc_units = 32
    factor = 5
    epochs = 1
    n_samples = factor * batch_size
    
    train_real_x = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_x = np.array(train_real_x)
    print(train_real_x.shape)
    
    train_real_y = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_y = np.array(train_real_y)
    print(train_real_y.shape)
    dataset_in = tf.data.Dataset.from_tensor_slices((train_real_x)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((train_real_y)).batch(batch_size)
    
    '''

    generator, encoder = neural_network.make_generator_model(LEN_AA, vocab_size, embedding_dim, enc_units, batch_size)

    parent_encoder_model, gen_encoder_model = neural_network.make_disc_par_gen_model(LEN_AA, vocab_size, embedding_dim, enc_units)

    discriminator = neural_network.make_discriminator_model(LEN_AA, vocab_size, embedding_dim, enc_units)

    print("Start training ...")
    
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')

    train_gen_loss = list()
    train_disc_loss = list()
    pretrain_gen_loss = list()
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"]
        y = tr_clade_df["Y"]
        print(tr_clade_df.shape)
        print("Pretraining generator...")
        X_pretrain, X_train, y_pretrain, y_train  = train_test_split(X, y, test_size=0.5)
        pretrain_dataset_in = tf.data.Dataset.from_tensor_slices((X_pretrain)).batch(batch_size)
        pretrain_dataset_out = tf.data.Dataset.from_tensor_slices((y_pretrain)).batch(batch_size)
        
        for i in range(pretrain_epochs):
            print("Pre training epoch {}...".format(str(i+1)))
            epo_pretrain_gen_loss, encoder, generator = train_model.pretrain_generator([pretrain_dataset_in, pretrain_dataset_out], encoder, generator)
            print("Pre training loss at step {}: Generator loss: {}".format(str(n+1), str(epo_gen_loss)))
            pretrain_gen_loss.append(epo_pretrain_gen_loss)
        np.savetxt(PRETRAIN_GEN_LOSS, pretrain_gen_loss) 
            
        sys.exit()
        print("Training Gen and Disc...")
        dataset_in = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
        dataset_out = tf.data.Dataset.from_tensor_slices((y_train)).batch(batch_size)
        total_te_loss = []
        for n in range(epochs):
            print("Training epoch {}...".format(str(n+1)))
            epo_gen_loss, epo_disc_loss, encoder, generator = train_model.start_training([dataset_in, dataset_out], enc_units, generator, encoder, parent_encoder_model, gen_encoder_model, discriminator)
            print("Training loss at step {}: Generator loss: {}, Discriminator loss :{}".format(str(n+1), str(epo_gen_loss), str(epo_disc_loss)))
            train_gen_loss.append(epo_gen_loss)
            train_disc_loss.append(epo_disc_loss)
            
            for te_name in te_clade_files:
                te_clade_df = pd.read_csv(te_name, sep="\t")
                te_X = te_clade_df["X"]
                te_y = te_clade_df["Y"]
                print(te_clade_df.shape)
                print("Prediction on test data...")
                te_loss = predict_sequence(te_X, te_y, LEN_AA, vocab_size, batch_size, encoder, generator)
                total_te_loss.append(te_loss)
        np.savetxt(TRAIN_GEN_LOSS, train_gen_loss)
        np.savetxt(TRAIN_DISC_LOSS, train_disc_loss)
        np.savetxt(TEST_LOSS, total_te_loss)

def predict_sequence(test_x, test_y, seq_len, vocab_size, batch_size, encoder, generator):
    avg_test_loss = []
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    i = 0
    m_loss = masked_loss.MaskedLoss()
    model = tf.keras.models.load_model("data/generated_files/model")
    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
    
        batch_x_test = utils.convert_to_array(x)
        batch_y_test = utils.convert_to_array(y)

        if batch_x_test.shape[0] == batch_size:
            input_mask = batch_x_test != 0
            target_mask = batch_y_test != 0
            new_tokens = tf.fill([batch_size, seq_len], 0)
            noise = tf.random.normal((batch_size, enc_units))
            #noise = tf.random.normal((batch_size, enc_units))
            #generated_logits = generator([unrolled_x, new_tokens, noise], training=True)

            enc_output, enc_state = encoder(batch_x_test)
            enc_state = tf.math.add(enc_state, noise)
            generated_logits, state = generator([new_tokens, enc_state], training=False)
            #enc_state = tf.math.add(enc_state, noise)

            

            #generated_logits = model([batch_x_test, new_tokens, noise], training=False)
            generated_tokens = tf.math.argmax(generated_logits, axis=-1)
            '''if step == 5:
                print("Test: Sample 0, batch {}".format(str(step)))
                print(batch_y_test[0])
                print(tf.argmax(generated_logits, axis=-1)[0])
                print(m_loss(batch_y_test[0], generated_tokens[0]))'''
        
            # compute loss
            y = batch_y_test
            y_pred = generated_logits
            loss = m_loss(y, y_pred)

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
            real_loss = average_loss.numpy()
            print("Test: Batch {} loss: {}".format(str(i), str(real_loss)))
            avg_test_loss.append(real_loss)
            i += 1
    mean_loss = np.mean(avg_test_loss)
    print("Total test loss: {}".format(str(mean_loss)))
    return mean_loss
    

if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
     
