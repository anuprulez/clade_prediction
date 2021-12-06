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
import tensorflow_addons as tfa
from Levenshtein import distance as lev_dist

import encoder_decoder_attention

PATH_KMER_F_DICT = "data/ncov_global/kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = "data/ncov_global/kmer_r_word_dictionaries.json"

m_loss = encoder_decoder_attention.MaskedLoss()
teacher_forcing_ratio = 0.5

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
mae = tf.keras.losses.MeanAbsoluteError()


def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real, 0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)  
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss 


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


def get_u_kmers(x_seq, y_seq, max_l_dist, len_aa_subseq, forward_dict):
    x_list = list()
    y_list = list()
    s_kmer = 3

    for i, x_i in enumerate(x_seq):
        for j, y_j in enumerate(y_seq):
            # cut sequences of specific length
            sub_x_i = x_i.split(",")[:len_aa_subseq]
            sub_x_i = ",".join(sub_x_i)

            sub_y_j = y_j.split(",")[:len_aa_subseq]
            sub_y_j = ",".join(sub_y_j)

            x_list.append(sub_x_i)

            y_list.append(sub_y_j)
    
    global_kmers = get_all_kmers(x_list, y_list, forward_dict, s_kmer)

    global_kmers = list(set(global_kmers))

    kmer_f_dict = {i + 1: global_kmers[i] for i in range(0, len(global_kmers))}
    kmer_r_dict = {global_kmers[i]: i + 1  for i in range(0, len(global_kmers))}

    save_as_json(PATH_KMER_F_DICT, kmer_f_dict)
    save_as_json(PATH_KMER_R_DICT, kmer_r_dict)

    #print(forward_dict)
    #print(kmer_f_dict)
    #print(kmer_r_dict)

    #print(len(x_list), len(y_list))

    '''ld_f = list()
    for index, (x, y) in enumerate(zip(x_list, y_list)):
        ld = compute_Levenshtein_dist(x, y)
        if ld > 0:
            ld_f.append(index)
    print(ld_f)'''
    #import sys
    #sys.exit()
    #print(x_list[0], len(x_list[0].split(",")))
    #print(y_list[0], len(y_list[0].split(",")))

    enc_x, enc_y = encode_sequences_kmers(forward_dict, kmer_r_dict, x_list, y_list, s_kmer)

    '''print(len(enc_x), len(enc_y))

    print(enc_x[ld_f[0]])
    print(enc_y[ld_f[0]])
    print()
    print(enc_x[ld_f[1]])
    print(enc_y[ld_f[1]])'''
    #print(enc_x[0])
    #print(enc_y[0])
    #mport sys
    #sys.exit()

    fil_x = list()
    fil_y = list()

    l_distance = list()
    filtered_l_distance = list()
    for i, (enc_i, enc_j) in enumerate(zip(enc_x, enc_y)):
        l_dist = compute_Levenshtein_dist(enc_i, enc_j)
        l_distance.append(l_dist)
        #print(i, l_dist)
        if l_dist > 0 and l_dist < max_l_dist:
            filtered_l_distance.append(l_dist)
            fil_x.append(add_padding_to_seq(enc_i))
            fil_y.append(add_padding_to_seq(enc_j))

    #print(fil_x[0])
    #print(fil_y[0])
    #print(np.mean(filtered_l_distance))
    return fil_x, fil_y, kmer_f_dict, kmer_r_dict

    '''train_kmers = get_all_kmers(combined_X, combined_y, forward_dict, s_kmer)
    kmers_global.extend(train_kmers)

    kmers_global = list(set(kmers_global))

    kmer_f_dict = {i + 1: kmers_global[i] for i in range(0, len(kmers_global))}
    kmer_r_dict = {kmers_global[i]: i + 1  for i in range(0, len(kmers_global))}
    utils.save_as_json(PATH_KMER_F_DICT, kmer_f_dict)
    utils.save_as_json(PATH_KMER_R_DICT, kmer_r_dict)

    kmer_f_dict[0] = "<start>"
    #kmer_f_dict[len(kmers_global)+1] = "<end>"
    kmer_r_dict["<start>"] = 0'''


def generate_cross_product(x_seq, y_seq, max_l_dist, len_aa_subseq, forward_dict, cols=["X", "Y"], unrelated=False, unrelated_threshold=15):
    print(len(x_seq), len(y_seq))
    x_y = list(itertools.product(x_seq, y_seq))
    print(len(x_y))
    print("Filtering for range of levenshtein distance...")
    l_distance = list()
    filtered_l_distance = list()
    filtered_x = list()
    filtered_y = list()

    filtered_x, filtered_y, kmer_f_dict, kmer_r_dict = get_u_kmers(x_seq, y_seq, max_l_dist, len_aa_subseq, forward_dict)
    
    #import sys
    #sys.exit()
    '''for i, x_i in enumerate(x_seq):
        for j, y_j in enumerate(y_seq):
            # cut sequences of specific length
            
            sub_x_i = x_i.split(",")[:len_aa_subseq]
            sub_x_i = ",".join(sub_x_i)
            #print(len(x_i.split(",")), len(sub_x_i.split(",")))

            sub_y_j = y_j.split(",")[:len_aa_subseq]
            sub_y_j = ",".join(sub_y_j)
            #print(len(y_j.split(",")), len(sub_y_j.split(",")))

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
                    filtered_l_distance.append(l_dist)'''

    print(len(filtered_l_distance), np.mean(filtered_l_distance))
    filtered_dataframe = pd.DataFrame(list(zip(filtered_x, filtered_y)), columns=["X", "Y"])
    print("Combined dataframe size: {}".format(str(len(filtered_dataframe.index))))
    np.savetxt("data/generated_files/l_distance.txt", l_distance)
    np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)
    print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
    print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
    print("Filtered dataframe size: {}".format(str(len(filtered_dataframe.index))))
    return filtered_dataframe, kmer_f_dict, kmer_r_dict


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
        x = x.split(",") #[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)
        #print(x_seq)
        y = y.split(",") #[1:]
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, s_kmer)

        #print(x_kmers)
        encoded_x = [str(kmer_r_dict[str(i)]) for i in x_kmers]
        encoded_x = ",".join(encoded_x) #+ "," + str(len(kmer_r_dict) - 1)
        in_seq.append(encoded_x)

        y_kmers = make_kmers(y_chars, s_kmer)
        encoded_y = [str(kmer_r_dict[str(i)]) for i in y_kmers]
        encoded_y = ",".join(encoded_y) #+ "," + str(len(kmer_r_dict) - 1)
        out_seq.append(encoded_y)

        #print(encoded_x)
        #print(encoded_y)
        #	import sys
        #sys.exit()
    return in_seq, out_seq


def get_all_kmers(x_seq, y_seq, f_dict, s_kmer):
    all_kmers = list()
    for index, (x, y) in enumerate(zip(x_seq, y_seq)):
        x = x.split(",") #[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)

        y = y.split(",") #[1:]
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


def stateful_encoding(size_stateful, inputs, enc, training=False):
    stateful_batches = list()
    n_stateful_batches = int(inputs.shape[1]/float(size_stateful))
    for i in range(n_stateful_batches):
        s_batch = inputs[:, i*size_stateful: (i+1)*size_stateful]
        enc_out, enc_state_h, enc_state_c = enc(s_batch, training=training)
    return enc_out, enc_state_h, enc_state_c


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


def evaluate_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, loaded_encoder, loaded_decoder):
  
  for step in range(n_te_batches):
        batch_x_test, batch_y_test = sample_unrelated_x_y(test_dataset_in, test_dataset_out, te_batch_size)
        enc_start_state = [tf.zeros((te_batch_size, enc_units)), tf.zeros((te_batch_size, enc_units))]

        print(batch_x_test.shape)

        enc_out, enc_h, enc_c = loaded_encoder(batch_x_test, enc_start_state)

        print(enc_out.shape)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([te_batch_size], 0)

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        print(greedy_sampler)

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=loaded_decoder.rnn_cell, sampler=greedy_sampler, output_layer=loaded_decoder.fc)

        print(decoder_instance)
        # Setup Memory in decoder stack
        loaded_decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = loaded_decoder.build_initial_state(te_batch_size, [enc_h, enc_c], tf.float32)

        print(decoder_initial_state)
        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
        ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        decoder_embedding_matrix = loaded_decoder.embedding.variables[0]
        print(decoder_embedding_matrix.shape)

        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token=vocab_size-2, initial_state=decoder_initial_state)
        print(outputs, dir(outputs))

        '''loaded_decoder.attention_mechanism.setup_memory(enc_out)
        decoder_initial_state = gen_decoder.build_initial_state(batch_size, [enc_h, enc_c], tf.float32)
        
        pred = gen_decoder(, decoder_initial_state)
        gen_logits = pred.rnn_output
        gen_loss = loss_function(unrolled_y, gen_logits)'''

        #return outputs.sample_id.numpy()

'''def translate(sentence):
  result = evaluate_sentence(sentence)
  print(result)
  result = targ_lang.sequences_to_texts(result)
  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))'''


def _loop_pred_step(seq_len, batch_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units):

  #input_tokens = self.input_text_processor(input_text)
  enc_output, enc_state = gen_encoder(input_tokens, training=False)

  dec_state = enc_state
  dec_state = tf.math.add(dec_state, tf.random.normal((batch_size, enc_units)))
  new_tokens = tf.fill([batch_size, 1], 0)
  gen_logits = list()
  #result_tokens = []
  attention = []
  #done = tf.zeros([batch_size, 1], dtype=tf.bool)
  loss = tf.constant(0.0)
  target_mask = output_tokens != 0

  for t in range(seq_len - 1):
   
    o_tokens = output_tokens[:, t+1:t+2]
    dec_input = encoder_decoder_attention.DecoderInput(new_tokens=new_tokens,
                             enc_output=enc_output,
                             mask=(input_tokens!=0))

    dec_result, dec_state = gen_decoder(dec_input, state=dec_state, training=False)

    gen_logits.append(dec_result.logits)

    #print(output_tokens.shape, o_tokens.shape, dec_result.logits.shape, dec_state.shape)

    new_tokens = tf.argmax(dec_result.logits, axis=-1)

    step_loss = m_loss(o_tokens, dec_result.logits)
    loss += step_loss

    #result_tokens.append(dec_result)

  t_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
  #result_tokens = tf.concat(result_tokens, axis=-1)
  gen_logits = tf.concat(gen_logits, axis=-2)
  return gen_logits, t_loss


def predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, loaded_encoder, loaded_generator, s_stateful):
    avg_test_loss = []
    avg_test_seq_var = []
    train_mode = False
    for step in range(n_te_batches):
        batch_x_test, batch_y_test = sample_unrelated_x_y(test_dataset_in, test_dataset_out, te_batch_size)
        # generated noise for variation in predicted sequences
        #noise = tf.random.normal((te_batch_size, enc_units))
        #enc_output, enc_state_h, enc_state_c = loaded_encoder(batch_x_test, training=train_mode)
        #enc_out, enc_state_h, enc_state_c = stateful_encoding(s_stateful, batch_x_test, loaded_encoder, False)
        #loaded_encoder.reset_states()
        #enc_state_h = tf.math.add(enc_state_h, noise)
        #enc_state_c = tf.math.add(enc_state_c, noise)
        #dec_state_h, dec_state_c = enc_state_h, enc_state_c
        '''print("Test: true output seq:")
        print(batch_x_test)'''
        #print()
        #print(batch_y_test)
        # generate seqs stepwise - teacher forcing
        generated_logits, loss = _loop_pred_step(seq_len, te_batch_size, batch_x_test, batch_y_test, loaded_encoder, loaded_generator, enc_units)
         #generated_output_seqs(seq_len, te_batch_size, vocab_size, loaded_generator, dec_state_h, dec_state_c, batch_x_test, batch_y_test, False)  
        variation_score = get_sequence_variation_percentage(generated_logits)
        loss = loss / variation_score #+ mae([1.0], [variation_score]) #variation_score
        print("Test batch {} variation score: {}".format(str(step+1), str(variation_score)))
        print("Test batch {} true loss: {}".format(str(step+1), str(loss.numpy())))
        avg_test_loss.append(loss)
        avg_test_seq_var.append(variation_score)
    print()
    print("Total test seq variation in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_seq_var))))
    print("Total test loss in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_loss))))
    return np.mean(avg_test_loss), np.mean(avg_test_seq_var)


def generator_step(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_i, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = real_o[:, 0:1]
    for t in tf.range(seq_len - 1):
        new_tokens = real_o[:, t:t+2]
        #i_token, o_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        if len(real_o) > 0:
            o_token = new_tokens[:, 1:2]
            #loss = loss_function(o_token, dec_result)
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


def generated_output_seqs(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_i, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len - 1):
        #dec_result, dec_state = gen_decoder([i_token, dec_state], training=train_gen)
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        gen_logits.append(dec_result)
        if len(real_o) > 0:
            o_token = real_o[:, t+1:t+2]
            #loss = loss_function(o_token, dec_result)
            loss = cross_entropy_loss(o_token, dec_result)
            step_loss += tf.reduce_mean(loss)
        #self feeding
        i_token = tf.argmax(dec_result, axis=-1)
    step_loss = step_loss / float(seq_len)
    pred_logits = tf.concat(gen_logits, axis=-2)
    gen_decoder.reset_states()
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
