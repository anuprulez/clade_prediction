import itertools
import json
import pandas as pd
import numpy as np
import random
from random import choices

import tensorflow as tf
from Levenshtein import distance as lev_dist

SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def make_kmers(seq, size):
    # remove all letters other than A,C,G and T
    #list(filter(lambda ch: ch in 'ACGT', kmers))
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]


def compute_Levenshtein_dist(seq_in, seq_out):
    return np.random.randint(1, 5)
    #return lev_dist(seq_in, seq_out)

def reconstruct_seq(kmers):
    reconstructed_seq = []
    for i, km in enumerate(kmers):
         if i < len(kmers) - 1:
             reconstructed_seq.append(km[0])
         else:
             reconstructed_seq.append(km)
    return "".join(reconstructed_seq)


def get_all_possible_words(vocab):
    return [char for char in vocab]

def convert_to_array(str_data):
    shp = str_data.shape[0]
    tolst = str_data.numpy()
    f_list = [item.decode("utf-8").split(",") for item in tolst]
    toarray = np.array([list(map(int, lst)) for lst in f_list])
    #tensor = tf.convert_to_tensor(toarray, dtype=tf.int32)
    return toarray

def pred_convert_to_array(str_data):
    shp = str_data.shape[0]
    tolst = str_data.numpy()
    f_list = [item.decode("utf-8").split(",") for item in tolst]
    toarray = np.array([list(map(int, lst)) for lst in f_list])
    return tf.convert_to_tensor(toarray, dtype=tf.int32)
    
    
def one_hot_encoding():
    encoded_seq = list()
    for char in sequence:
        one_hot = np.zeros(len(aa_chars))
        one_hot[int(r_word_dictionaries[char]) - 1] = 1
        encoded_seq.append(one_hot)
        
    print(encoded_seq)
    #print(len(indices_kmers))
    #print(sequence)
    #print(np.zeros(len(aa_chars)))
    #zeros = np.repeat(np.zeros(len(aa_chars)), (LEN_AA - len(sequence) + 1))
    #print(zeros)
    print()
    trailing_zeros = list()
    for i in range(LEN_AA - len(sequence)):
        trailing_zeros.append(np.zeros(len(aa_chars)))
    print(trailing_zeros)
    
    encoded_seq.extend(trailing_zeros)
    print()
    print(encoded_seq)
    encoded_seq = np.array(encoded_seq)
    print(encoded_seq.shape)

def read_in_out(path):
    data_df = pd.read_csv(path, sep="\t")
    samples = data_df[['Sequence_x', 'Sequence_y']]
    samples["Sequence_x"] = samples["Sequence_x"].str.split(",")
    samples["Sequence_y"] = samples["Sequence_y"].str.split(",")
    return samples

def get_words_indices(word_list):
    forward_dictionary = {i + 1: word_list[i] for i in range(0, len(word_list))}
    reverse_dictionary = {word_list[i]: i + 1  for i in range(0, len(word_list))}
    return forward_dictionary, reverse_dictionary


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content
        

def format_clade_name(c_name):
    return c_name.replace("/", "_")
    
    
def embedding_info(dict_json):
    return len(dict_json)
    
def convert_to_string_list(l):
    l = l.numpy()
    l = [",".join([str(i) for i in item]) for item in l]
    return l


def predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, enc_units, enc_path, dec_path):
    avg_test_loss = []
    i = 0
    loaded_encoder = tf.keras.models.load_model(enc_path)
    loaded_generator = tf.keras.models.load_model(dec_path)
    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
        batch_x_test = convert_to_array(x)
        batch_y_test = convert_to_array(y)
        batch_size = batch_x_test.shape[0]
        if batch_x_test.shape[0] == batch_size:
            # generated noise for variation in predicted sequences
            noise = tf.random.normal((batch_size, enc_units))
            enc_output, enc_state = loaded_encoder(batch_x_test, training=False)
            # add noise to the encoder state
            enc_state = tf.math.add(enc_state, noise)
            dec_state = enc_state
            # generate seqs stepwise - teacher forcing
            generated_logits, _, loss = gen_step_predict(seq_len, batch_size, vocab_size, loaded_generator, dec_state, batch_y_test)
            print("Test: Batch {} true loss: {}".format(str(i), str(loss)))
            avg_test_loss.append(loss)
            i += 1
    mean_loss = np.mean(avg_test_loss)
    print("Total test loss: {}".format(str(mean_loss)))
    return mean_loss


def gen_step_predict(seq_len, batch_size, vocab_size, gen_decoder, dec_state, real_o):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, seq_len, vocab_size))
    # set initial token
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len):
        o_token = real_o[:, t:t+1]
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=False)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        loss = SCE(o_token, dec_result)
        step_loss += loss
        dec_tokens = tf.math.argmax(dec_result, axis=-1)
        # teacher forcing, set current output as the next input
        i_token = dec_tokens
    step_loss = step_loss / seq_len
    pred_logits = tf.convert_to_tensor(pred_logits)
    return pred_logits, gen_decoder, step_loss


def balance_train_dataset(x, y, x_y_l):
    lst_x = x
    lst_y = y
    l_dist = x_y_l.numpy()
    u_l_dist = list(set(l_dist))
    batch_size = x.shape[0]
    # create more samples, take out samples equal to the number of batch_size
    n_samples = int(batch_size / float(len(u_l_dist))) + 1
    bal_x = list()
    bal_y = list()

    for l_val in u_l_dist:
        l_val_indices = np.where(l_dist == int(l_val))
        len_indices = len(l_val_indices)
        x_rows = np.array(lst_x[l_val_indices])
        y_rows = np.array(lst_y[l_val_indices])
        rand_x_rows = np.array(choices(x_rows, k=n_samples))
        rand_y_rows = np.array(choices(y_rows, k=n_samples))

        bal_x.extend(rand_x_rows)
        bal_y.extend(rand_y_rows)
        #print(l_val, len(x_rows), len(y_rows), n_samples, len(rand_x_rows), len(rand_y_rows))

    bal_x = np.array(bal_x)
    bal_y = np.array(bal_y)

    rand_idx = np.random.randint(1, bal_x.shape[0], batch_size)
 
    bal_x_bs = bal_x[rand_idx]
    bal_y_bs = bal_y[rand_idx]

    bal_x_bs = tf.convert_to_tensor(bal_x_bs, dtype=tf.int32)
    bal_y_bs = tf.convert_to_tensor(bal_y_bs, dtype=tf.int32)
    #print(bal_x_bs)
    #print()
    #print(bal_y_bs)
    return bal_x_bs, bal_y_bs
