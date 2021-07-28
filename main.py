import time
import sys
import os

import random
import pandas as pd
import numpy as np
from numpy.random import randn
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
embedding_dim = 8
batch_size = 64
units = 128
epochs = 20
LEN_AA = 1275

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    '''samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    vocab_size = utils.embedding_info(forward_dict)
    
    #print("Transforming generated samples...")
    #train_x, train_y, test_x, test_y = preprocess_sequences.transform_encoded_samples()
    
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    train_x = np.array([list(map(int, lst)) for lst in train_x])
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

    #test_x = [list(map(int, lst)) for lst in test_x]
    #test_y = [list(map(int, lst)) for lst in test_y]
    
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
    dataset_out = tf.data.Dataset.from_tensor_slices((train_y)).batch(batch_size)
    #start_training(embedding_dim, units, batch_size, vocab_size)
    '''
    
    '''seq_len = 50
    vocab_size = 20
    latent_dim = 100
    batch_size = 32
    embedding_dim = 16
    enc_units = 16
    factor = 2
    
    n_samples = factor * batch_size
    
    train_real_x = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_x = np.array(train_real_x)
    print(train_real_x.shape)

    train_real_y = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_y = np.array(train_real_y)
    print(train_real_y.shape)
    
    encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, units)
    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, units)
    
    enc_output, enc_state = encoder(train_real_x)
    
    input_mask = train_real_x != 0
    
    x_input = randn(n_samples * seq_len * vocab_size)
    x_input = x_input.reshape(n_samples, seq_len, vocab_size)
    print(x_input.shape)
    
    random_tokens = tf.argmax(x_input, axis=-1)

    decoder_input = container_classes.DecoderInput(new_tokens=random_tokens, enc_output=enc_output, mask=input_mask)

    dec_result, dec_state = decoder(decoder_input, state=enc_state)
    
    print(dec_result.logits.shape, dec_state.shape)
    
    fake_pred_tokens = tf.argmax(dec_result.logits, axis=-1)
    print(fake_pred_tokens)
    
    y_real = np.ones((n_samples, 1))
    y_fake = np.zeros((n_samples, 1))
    
    print(train_real_y.shape, fake_pred_tokens.shape)
    X, y = np.vstack((train_real_y, fake_pred_tokens)), np.vstack((y_real, y_fake))
    print(X.shape, y.shape)
    
    disc_model = neural_network.make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units)
    
    print(disc_model.summary())
    
    print(X)
    print()
    print(y)
    
    disc_model.fit(X, y, epochs=10)'''
    
    '''gan_model = neural_network.make_generator_model(seq_len, vocab_size, (n_samples, seq_len))
    print(gan_model.summary())
    
    gan_X, gan_y = generate_fake_samples(gan_model, latent_dim, batch_size, train_real_x)
    #pred_tokens = tf.argmax(gan_X, axis=-1)
    #print(pred_tokens)
    print(gan_X.shape)
    print()
    print(gan_y.shape)
    
    disc_model = neural_network.make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units)
    
    print(disc_model.summary())
    
    gen_disc_model = define_gan(gan_model, disc_model)'''
    
    
    seq_len = 50
    vocab_size = 20
    latent_dim = 100
    batch_size = 32
    embedding_dim = 16
    enc_units = 16
    factor = 1
    epochs = 2
    n_samples = factor * batch_size
    
    train_real_x = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_x = np.array(train_real_x)
    print(train_real_x.shape)
    
    train_real_y = [np.random.randint(vocab_size, size=seq_len) for i in range(n_samples)]
    train_real_y = np.array(train_real_y)
    print(train_real_y.shape)
    
    fake_output = randn(n_samples * seq_len * vocab_size)
    fake_output = fake_output.reshape(n_samples, seq_len, vocab_size)
    fake_seq = tf.argmax(fake_output, axis=-1)
    print(fake_seq.shape)

    

    input_tokens = tf.data.Dataset.from_tensor_slices((train_real_x)).batch(batch_size)
    target_tokens = tf.data.Dataset.from_tensor_slices((train_real_y)).batch(batch_size)
    fake_target = tf.data.Dataset.from_tensor_slices((fake_seq)).batch(batch_size)

    for n in range(epochs):
        print("Training epoch {}...".format(str(n)))
        for step, (x_batch_train, y_batch_train, y_batch_fake) in enumerate(zip(input_tokens, target_tokens, fake_target)):
            neural_network.train_step(x_batch_train, y_batch_train, y_batch_fake)
    
# generate points in latent space as input for the generator
'''def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input    
    

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples, train_real_x):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    x_output = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = train_real_x #g_model.predict(x_input)
    y = g_model.predict(x_output)
    # create 'fake' class labels (0)
    # y = np.zeros((n_samples, 1))
    return X, y
    
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=masked_loss.MaskedLoss(), optimizer=opt)
    return model
   

def train_gan(gan_model, latent_dim, train_real_y, n_epochs=100, n_batch=256):
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = train_real_y #ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
	    # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
	    # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
	    # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
	    # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
	    # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
	    # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
	    # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))'''




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
            print("Training epoch {}...".format(str(n)))
            batch_learning = model.train_step([dataset_in, dataset_out])
            print("Training loss at step {}: {}".format(str(n+1), str(np.round(batch_learning["epo_loss"], 4))))
            for te_name in te_clade_files:
                te_clade_df = pd.read_csv(te_name, sep="\t")
                te_X = clade_df["X"]
                te_y = clade_df["Y"]
                print(te_clade_df.shape)
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
        
        print(batch_x_test.shape, batch_y_test.shape)
        
        if batch_x_test.shape[0] == batch_size:
        
            enc_output, enc_state = model.encoder(batch_x_test)
        
            print(enc_output.shape)
        
            input_mask = batch_x_test != 0
            target_mask = batch_y_test != 0
            new_tokens = tf.fill([batch_size, seq_len], 0)

            print(new_tokens.shape)

            decoder_input = container_classes.DecoderInput(new_tokens=new_tokens, enc_output=enc_output, mask=input_mask)
            dec_result, dec_state = model.decoder(decoder_input, state=enc_state)
        
            print(dec_result.logits.shape, dec_state.shape)
        
            # compute loss
            y = batch_y_test
            y_pred = dec_result.logits
            pred_tokens = tf.argmax(y_pred, axis=-1)
            loss = model.loss(y, y_pred)
        
        
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        
            real_loss = average_loss.numpy()
            print("Batch {} loss: {}".format(str(i), str(real_loss)))
            avg_test_loss.append(real_loss)
            i += 1
    print("Total test loss: {}".format(str(np.mean(avg_test_loss))))



if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))

