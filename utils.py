import os
import sys
import shutil
import itertools
from itertools import product
import json
import pandas as pd
import numpy as np
import random
from random import choices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Levenshtein import distance as lev_dist


teacher_forcing_ratio = 1.0

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def make_kmers(seq, size):
    return ["".join(seq[x:x+size]) for x in range(len(seq) - size + 1)]


def compute_Levenshtein_dist(seq_in, seq_out):
    #return np.random.randint(1, 5)
    return lev_dist(seq_in, seq_out)


def clean_up(list_folders):
    for folder in list_folders:
        try:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder, e))
            continue


def add_padding_to_seq(seq):
    return "{},{}".format(str(0), seq)


def generate_cross_product(x_seq, y_seq, max_l_dist, len_aa_subseq, cols=["X", "Y"], unrelated=False, unrelated_threshold=15):
    print(len(x_seq), len(y_seq))
    x_y = list(itertools.product(x_seq, y_seq))
    print(len(x_y))
    print("Filtering for range of levenshtein distance...")
    l_distance = list()
    filtered_l_distance = list()
    filtered_x = list()
    filtered_y = list()
    for i, x_i in enumerate(x_seq):
        for j, y_j in enumerate(y_seq):
            # cut sequences of specific length
            sub_x_i = x_i.split(",")[:len_aa_subseq]
            sub_x_i = ",".join(sub_x_i)
            sub_y_j = y_j.split(",")[:len_aa_subseq]
            sub_y_j = ",".join(sub_y_j)
            l_dist = compute_Levenshtein_dist(sub_x_i, sub_y_j)
            l_distance.append(l_dist)
            if unrelated is False:
                if l_dist > 0 and l_dist < max_l_dist:
                    filtered_x.append(add_padding_to_seq(sub_x_i))
                    filtered_y.append(add_padding_to_seq(sub_y_j))
                    filtered_l_distance.append(l_dist)
            else:
                if l_dist > max_l_dist:
                    filtered_x.append(add_padding_to_seq(sub_x_i))
                    filtered_y.append(add_padding_to_seq(sub_y_j))
                    filtered_l_distance.append(l_dist)

    filtered_dataframe = pd.DataFrame(list(zip(filtered_x, filtered_y)), columns=["X", "Y"])
    print("Combined dataframe size: {}".format(str(len(filtered_dataframe.index))))
    np.savetxt("data/generated_files/l_distance.txt", l_distance)
    np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)
    print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
    print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
    print("Filtered dataframe size: {}".format(str(len(filtered_dataframe.index))))
    return filtered_dataframe


def transform_noise(noise):
    shp = noise.shape
    binary = np.random.choice([0, 1], shp[0] * shp[1])
    binary_reshape = np.reshape(binary, (shp[0], shp[1]))
    noise *= binary_reshape
    return tf.convert_to_tensor(noise, dtype=tf.float32)


def get_all_possible_words(vocab, kmer_size=3):
    all_com = [''.join(c) for c in product(vocab, repeat=kmer_size)]
    kmer_f_dict = {i + 1: all_com[i] for i in range(0, len(all_com))}
    kmer_r_dict = {all_com[i]: i + 1  for i in range(0, len(all_com))}
    save_as_json(PATH_KMER_F_DICT, kmer_f_dict)
    save_as_json(PATH_KMER_R_DICT, kmer_r_dict)
    return kmer_f_dict, kmer_r_dict


def convert_to_array(str_data):
    tolst = str_data.tolist()
    f_list = [item.split(",") for item in tolst]
    toarray = np.array([list(map(int, lst)) for lst in f_list])
    tensor = tf.convert_to_tensor(toarray, dtype=tf.int32)
    return toarray


def pred_convert_to_array(str_data):
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


def reconstruct_seq(kmers):
    reconstructed_seq = []
    for i, km in enumerate(kmers):
         if i < len(kmers) - 1:
             reconstructed_seq.append(km[0])
         else:
             reconstructed_seq.append(km)
    return "".join(reconstructed_seq)


def encode_sequences_kmers(f_dict, kmer_r_dict, x_seq, y_seq, s_kmer):
    in_seq = list()
    out_seq = list()
    for index, (x, y) in enumerate(zip(x_seq, y_seq)):
        x = x.split(",")[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)

        y = y.split(",")[1:]
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, s_kmer)
        encoded_x = [str(kmer_r_dict[str(i)]) for i in x_kmers]
        encoded_x = "0," + ",".join(encoded_x)
        in_seq.append(encoded_x)
   
        y_kmers = make_kmers(y_chars, s_kmer)
        encoded_y = [str(kmer_r_dict[str(i)]) for i in y_kmers]
        encoded_y = "0," + ",".join(encoded_y)
        out_seq.append(encoded_y)

    return in_seq, out_seq


def get_all_kmers(x_seq, y_seq, f_dict, s_kmer):
    all_kmers = list()
    for index, (x, y) in enumerate(zip(x_seq, y_seq)):
        x = x.split(",")[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)

        y = y.split(",")[1:]
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, s_kmer)
        y_kmers = make_kmers(y_chars, s_kmer)

        u_x_mers = list(set(x_kmers))
        u_y_mers = list(set(y_kmers))

        all_kmers.extend(u_x_mers)
        all_kmers.extend(u_y_mers)

    return list(set(all_kmers))


'''def ordinal_to_kmer(seq_df, f_dict, r_dict, kmer_f_dict, kmer_r_dict, kmer_s=3):
    in_seq = list()
    out_seq = list()
    for index, (x, y) in seq_df.iterrows():
        x = x.split(",")[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)

        y = y.split(",")[1:]
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, kmer_s)
        y_kmers = make_kmers(y_chars, kmer_s)

        encoded_x = [str(kmer_r_dict[str(i)]) for i in x_kmers]
        encoded_x = "0," + ",".join(encoded_x)
        in_seq.append(encoded_x)
        encoded_y = [str(kmer_r_dict[str(i)]) for i in y_kmers]
        encoded_y = "0," + ",".join(encoded_y)
        out_seq.append(encoded_y)

        # reconstruct seq from predicted kmers
        enc_x = encoded_x.split(",")[1:]
        print(enc_x)
        print()
        enc_x = [kmer_f_dict[int(i)] for i in enc_x]
        print(enc_x)
        orig_x = reconstruct_seq(enc_x)
        print()
        print(orig_x, len(orig_x))
        
    enc_df = pd.DataFrame(list(zip(in_seq, out_seq)), columns=seq_df.columns)
    return enc_df'''
        

def read_wuhan_seq(wu_path, rev_dict):
    with open(wu_path, "r") as wu_file:
        wuhan_seq = wu_file.read()
        wuhan_seq = wuhan_seq.split("\n")
        wuhan_seq = "".join(wuhan_seq)
        enc_wu_seq = [str(rev_dict[item]) for item in wuhan_seq]
        return ",".join(enc_wu_seq)


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


def get_sequence_variation_percentage(logits):
    seq_tokens = tf.math.argmax(logits, axis=-1)
    #print("Gen seqs:")
    #print(seq_tokens)
    l_seq_tokens = convert_to_string_list(seq_tokens)
    df_seqs = pd.DataFrame(l_seq_tokens, columns=["Sequences"])
    u_df_seqs = df_seqs.drop_duplicates()
    percent_variation = len(u_df_seqs.index) / float(len(df_seqs.index))
    return percent_variation


def sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size):
    un_rand_row_index = np.random.randint(0, unrelated_X.shape[0], batch_size)
    un_X = unrelated_X[un_rand_row_index]
    un_y = unrelated_y[un_rand_row_index]
    return convert_to_array(un_X), convert_to_array(un_y)


def predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, loaded_encoder, loaded_generator):
    avg_test_loss = []
    avg_test_seq_var = []
    train_mode = False
    for step in range(n_te_batches):
        batch_x_test, batch_y_test = sample_unrelated_x_y(test_dataset_in, test_dataset_out, te_batch_size)
        # generated noise for variation in predicted sequences
        noise = tf.random.normal((te_batch_size, 2 * enc_units))
        enc_output, enc_state_h, enc_state_c = loaded_encoder(batch_x_test, training=train_mode)
        enc_state_h = tf.math.add(enc_state_h, noise)
        enc_state_c = tf.math.add(enc_state_c, noise)
        dec_state_h, dec_state_c = enc_state_h, enc_state_c
        '''print("Test: true output seq:")
        print(batch_x_test)'''
        #print()
        #print(batch_y_test)
        # generate seqs stepwise - teacher forcing
        generated_logits, _, loss = generated_output_seqs(seq_len, te_batch_size, vocab_size, loaded_generator, dec_state_h, dec_state_c, batch_y_test, train_mode)  
        variation_score = get_sequence_variation_percentage(generated_logits)
        print("Test batch {} variation score: {}".format(str(step+1), str(variation_score)))
        print("Test batch {} true loss: {}".format(str(step+1), str(loss.numpy())))
        avg_test_loss.append(loss)
        avg_test_seq_var.append(variation_score)
    print()
    print("Total test seq variation in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_seq_var))))
    print("Total test loss in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_loss))))
    return np.mean(avg_test_loss), np.mean(avg_test_seq_var)


def generator_step(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = real_o[:, 0:1]
    for t in tf.range(seq_len - 1):
        new_tokens = real_o[:, t:t+2]
        #i_token, o_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        if len(real_o) > 0:
            o_token = new_tokens[:, 1:2]
            loss = cross_entropy_loss(o_token, dec_result)
            step_loss += tf.reduce_mean(loss)
        # randomly select either true output or feed generated output from previous step for forced learning
        if random.random() <= teacher_forcing_ratio:
            #print("True")
            i_token = o_token
        else:
            #print("Generated")
            i_token = tf.argmax(dec_result, axis=-1)
        gen_logits.append(dec_result)
    step_loss = step_loss / float(seq_len)
    pred_logits = tf.concat(gen_logits, axis=-2)
    return pred_logits, gen_decoder, step_loss


def generated_output_seqs(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len - 1):
        #dec_result, dec_state = gen_decoder([i_token, dec_state], training=train_gen)
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        gen_logits.append(dec_result)
        if len(real_o) > 0:
            o_token = real_o[:, t+1:t+2]
            loss = cross_entropy_loss(o_token, dec_result)
            step_loss += tf.reduce_mean(loss)
        #self feeding
        i_token = tf.argmax(dec_result, axis=-1)
    step_loss = step_loss / float(seq_len)
    pred_logits = tf.concat(gen_logits, axis=-2)
    return pred_logits, gen_decoder, step_loss


def balance_train_dataset_by_levenshtein_dist(x, y, x_y_l):
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
        print(l_val, len(x_rows), len(y_rows), n_samples, len(rand_x_rows), len(rand_y_rows))

    bal_x = np.array(bal_x)
    bal_y = np.array(bal_y)

    rand_idx = np.random.randint(1, bal_x.shape[0], batch_size)
 
    bal_x_bs = bal_x[rand_idx]
    bal_y_bs = bal_y[rand_idx]

    bal_x_bs = tf.convert_to_tensor(bal_x_bs, dtype=tf.int32)
    bal_y_bs = tf.convert_to_tensor(bal_y_bs, dtype=tf.int32)
    return bal_x_bs, bal_y_bs


def save_batch(batch_x, batch_y, batch_mut_distribution):
    for index, (x, y) in enumerate(zip(batch_x, batch_y)):
        true_x = x.split(",")
        true_y = y.split(",")
        for i in range(len(true_x)):
            first = true_x[i:i+1]
            sec = true_y[i:i+1]
            first_mut = first[0]
            second_mut = sec[0]
            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in batch_mut_distribution:
                    batch_mut_distribution[key] = 0
                batch_mut_distribution[key] += 1
    return batch_mut_distribution


def get_mutation_tr_indices(train_in, train_out, f_dict, r_dict):
    parent_child_mut_indices = dict()
    for index, (x, y) in enumerate(zip(train_in, train_out)):
        true_x = x.split(",")[1:]
        true_y = y.split(",")[1:]

        for i in range(len(true_x)):
            first = true_x[i:i+1]
            sec = true_y[i:i+1]

            first_aa = [f_dict[int(j)] for j in first]
            sec_aa = [f_dict[int(j)] for j in sec]
        
            first_mut = first_aa[0]
            second_mut = sec_aa[0]

            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in parent_child_mut_indices:
                    parent_child_mut_indices[key] = list()
                parent_child_mut_indices[key].append(index)
    return parent_child_mut_indices
