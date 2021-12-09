import time
import sys
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

import preprocess_sequences
import utils
import neural_network
import train_model


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
GALAXY_CLADE_ASSIGNMENT = PATH_PRE + "clade_assignment_2.9_Mil_samples.tabular"
PATH_SAMPLES_CLADES = PATH_PRE + "sample_clade_sequence_df.csv"
PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_KMER_F_DICT = "data/ncov_global/kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = "data/ncov_global/kmer_r_word_dictionaries.json"
PATH_TRAINING_CLADES = "data/train_clade_in_out.json"
PATH_UNRELATED_CLADES = "data/unrelated_clades.json"

PRETRAIN_DATA = "data/pretrain/pretrain.csv"
PRETRAIN_GEN_LOSS = "data/generated_files/pretrain_gen_train_loss.txt"
PRETRAIN_GEN_TEST_LOSS = "data/generated_files/pretrain_gen_test_loss.txt"

TRAIN_GEN_TOTAL_LOSS = "data/generated_files/train_gen_total_loss.txt"
TRAIN_GEN_FAKE_LOSS = "data/generated_files/train_gen_fake_loss.txt"
TRAIN_GEN_TRUE_LOSS = "data/generated_files/train_gen_true_loss.txt"

TRAIN_DISC_TOTAL_LOSS = "data/generated_files/train_disc_total_loss.txt"
TRAIN_DISC_FAKE_LOSS = "data/generated_files/train_disc_fake_loss.txt"
TRAIN_DISC_TRUE_LOSS = "data/generated_files/train_disc_true_loss.txt"

TEST_LOSS = "data/generated_files/train_te_loss.txt"

PRETRAIN_GEN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_DEC_MODEL = "data/generated_files/pretrain_gen_decoder"
TRAIN_GEN_ENC_MODEL = "data/generated_files/gen_enc_model"
TRAIN_GEN_DEC_MODEL = "data/generated_files/gen_dec_model"

SAVE_TRUE_PRED_SEQ = "data/generated_files/true_predicted_df.csv"
TR_MUT_INDICES = "data/generated_files/tr_mut_indices.json"
PRETR_MUT_INDICES = "data/generated_files/pretr_mut_indices.json"


'''
Best run

s_kmer = 3
LEN_AA = 1274
len_aa_subseq = 50
#len_final_aa_padding = len_aa_subseq + 1
len_final_aa_padding = len_aa_subseq - s_kmer + 2
# Neural network parameters
embedding_dim = 32
batch_size = 4
te_batch_size = batch_size
n_te_batches = 2
enc_units = 128

'''

s_kmer = 3
LEN_AA = 1274
len_aa_subseq = 31
#len_final_aa_padding = len_aa_subseq + 1
len_final_aa_padding = len_aa_subseq - s_kmer + 2
size_stateful = 10
# Neural network parameters
embedding_dim = 32
batch_size = 32
te_batch_size = batch_size
n_te_batches = 10
enc_units = 64
pretrain_epochs = 5
epochs = 2
max_l_dist = 11
test_train_size = 0.85
pretrain_train_size = 0.01
random_clade_size = 1000
to_pretrain = True
pretrained_model = False
gan_train = False
stale_folders = ["data/generated_files/", "data/train/", "data/test/", "data/tr_unrelated/", "data/te_unrelated/", "data/pretrain/"]
amino_acid_codes = "QNKWFPYLMTEIARGHSDVC"


def get_samples_clades():
    print("Reading clade assignments...")
    #samples_clades = preprocess_sequences.get_samples_clades(GALAXY_CLADE_ASSIGNMENT)
    samples_clades = preprocess_sequences.get_galaxy_samples_clades(GALAXY_CLADE_ASSIGNMENT)
    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq_galaxy_clades(PATH_SEQ, samples_clades, LEN_AA)
    print(encoded_sequence_df)

def read_files():
    #to preprocess once, uncomment get_samples_clades
    #get_samples_clades()
    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    encoder = None
    decoder = None
    #kmer_f_dict, kmer_r_dict = utils.get_all_possible_words(amino_acid_codes, s_kmer)

    if pretrained_model is False:
        print("Cleaning up stale folders...")
        utils.clean_up(stale_folders)
        print("Preprocessing sample-clade assignment file...")
        dataf = pd.read_csv(PATH_SAMPLES_CLADES, sep=",")
        filtered_dataf = preprocess_sequences.filter_samples_clades(dataf)

        clades_in_clades_out = utils.read_json(PATH_TRAINING_CLADES)
        print(clades_in_clades_out)
        unrelated_clades = utils.read_json(PATH_UNRELATED_CLADES)
        print("Generating cross product of real parent child...")
        preprocess_sequences.make_cross_product(clades_in_clades_out, filtered_dataf, len_aa_subseq, train_size=test_train_size, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=False)
        #print("Generating cross product of real sequences but not parent-child...")
        #preprocess_sequences.make_cross_product(unrelated_clades, filtered_dataf, len_aa_subseq, train_size=1.0, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=True)
    else:
        encoder = tf.keras.models.load_model(PRETRAIN_GEN_ENC_MODEL)
        decoder = tf.keras.models.load_model(PRETRAIN_GEN_DEC_MODEL)
    start_training(forward_dict, rev_dict, encoder, decoder)


def verify_ldist(X, Y):
    lev_list = list()
    for index, (x, y) in enumerate(zip(X, Y)):
        seq_x = x
        seq_y = y
        #print(seq_x, seq_y)
        lev = utils.compute_Levenshtein_dist(x, y)
        lev_list.append(lev)
    print(lev_list)
    print(np.mean(lev_list))

def start_training(forward_dict, rev_dict, gen_encoder=None, gen_decoder=None):
    start_time = time.time()
    print("Loading datasets...")
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')
    

    combined_X = list()
    combined_y = list()
    # load train data
    print("Loading training datasets...")
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"].tolist()
        y = tr_clade_df["Y"].tolist()
        combined_X.extend(X)
        combined_y.extend(y)
        

    #verify_ldist(combined_X, combined_y)

    #print(combined_X[0])

    #sys.exit()
    combined_te_X = list()
    combined_te_y = list()
    # load test data
    print("Loading test datasets...")
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"].tolist()
        te_y = te_clade_df["Y"].tolist()
        combined_te_X.extend(te_X)
        combined_te_y.extend(te_y)
        print(len(te_X), len(te_y))
    print()

    #verify_ldist(combined_te_X, combined_te_y)

    '''
    tr_unrelated_files = glob.glob("data/tr_unrelated/*.csv")
    print("Loading unrelated datasets...")
    unrelated_X = list()
    unrelated_y = list()
    for tr_unrelated in tr_unrelated_files:
        unrelated_clade_df = pd.read_csv(tr_unrelated, sep="\t")
        un_X = unrelated_clade_df["X"].tolist()
        un_y = unrelated_clade_df["Y"].tolist()
        unrelated_X.extend(un_X)
        unrelated_y.extend(un_y)
        print(len(un_X), len(un_y))

    unrelated_X = np.array(unrelated_X)
    unrelated_y = np.array(unrelated_y)
    print("Unrelated data sizes")
    print(len(unrelated_X), len(unrelated_y))'''

    print("train and test data sizes")
    print(len(combined_X), len(combined_y), len(combined_te_X), len(combined_te_y))

    # convert test and train datasets to kmers
    '''kmers_global = list()
    #print(forward_dict)
    train_kmers = utils.get_all_kmers(combined_X, combined_y, forward_dict, s_kmer)
    kmers_global.extend(train_kmers)

    test_kmers = utils.get_all_kmers(combined_te_X, combined_te_y, forward_dict, s_kmer)
    kmers_global.extend(test_kmers)

    kmers_global = list(set(kmers_global))

    kmer_f_dict = {i + 1: kmers_global[i] for i in range(0, len(kmers_global))}
    kmer_r_dict = {kmers_global[i]: i + 1  for i in range(0, len(kmers_global))}
    utils.save_as_json(PATH_KMER_F_DICT, kmer_f_dict)
    utils.save_as_json(PATH_KMER_R_DICT, kmer_r_dict)

    kmer_f_dict[0] = "<start>"
    #kmer_f_dict[len(kmers_global)+1] = "<end>"
    kmer_r_dict["<start>"] = 0
    #kmer_r_dict["<end>"] = len(kmers_global)+1

    print(kmer_f_dict, len(kmer_f_dict))
    print()
    print(print(kmer_r_dict, len(kmer_r_dict)))

    #sys.exit()
    vocab_size = len(kmer_f_dict) + 1

    print("Number of kmers: {}".format(str(len(kmer_f_dict) - 1)))
    print("Vocab size: {}".format(str(len(kmer_f_dict) + 1)))

    combined_X, combined_y = utils.encode_sequences_kmers(forward_dict, kmer_r_dict, combined_X, combined_y, s_kmer)
    combined_te_X, combined_te_y = utils.encode_sequences_kmers(forward_dict, kmer_r_dict, combined_te_X, combined_te_y, s_kmer)

    print(combined_X[0])
    print(combined_y[0])'''
 
    kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
    kmer_r_dict = utils.read_json(PATH_KMER_R_DICT)

    vocab_size = len(kmer_f_dict) + 1

    print("Number of kmers: {}".format(str(len(kmer_f_dict) - 1)))
    print("Vocab size: {}".format(str(len(kmer_f_dict) + 1)))

    combined_X = np.array(combined_X)
    combined_y = np.array(combined_y)

    test_dataset_in = np.array(combined_te_X)
    test_dataset_out = np.array(combined_te_y)

    #sys.exit()

    if gen_encoder is None or gen_decoder is None:
        encoder, decoder = neural_network.make_generator_model(len_final_aa_padding, vocab_size, embedding_dim, enc_units, batch_size, size_stateful)

        #encoder = encoder_decoder.Encoder(vocab_size, embedding_dim, enc_units, batch_size)
        #decoder = encoder_decoder.Decoder(vocab_size, embedding_dim, enc_units, batch_size, 'luong')
        out_vocab_size = 1
        #encoder = encoder_decoder_attention.Encoder(vocab_size, embedding_dim, enc_units)
        #decoder = encoder_decoder_attention.Decoder(vocab_size, embedding_dim, enc_units)
        #print(encoder, decoder)

    else:
        encoder = gen_encoder
        decoder = gen_decoder

    # divide into pretrain and train
    if to_pretrain is False:
        X_train = combined_X
        y_train = combined_y
    else:
        X_pretrain, X_train, y_pretrain, y_train  = train_test_split(combined_X, combined_y, test_size=pretrain_train_size)
        #utils.split_test_train(combined_X, combined_y, pretrain_train_size) 
        #train_test_split(combined_X, combined_y, test_size=pretrain_train_size)
        X_pretrain = np.array(X_pretrain)
        y_pretrain = np.array(y_pretrain)
        df_pretrain = pd.DataFrame(list(zip(X_pretrain, y_pretrain)), columns=["X", "Y"])
        df_pretrain.to_csv(PRETRAIN_DATA, sep="\t", index=None)
        print("Pretrain data sizes")
        print(X_pretrain.shape, y_pretrain.shape)
        # save update train dataset
        df_train = pd.DataFrame(list(zip(X_train, y_train)), columns=["X", "Y"])
        df_train.to_csv(tr_clade_files[0], sep="\t", index=None)

    print("Train data sizes")
    print(X_train.shape, y_train.shape)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #sys.exit()

    # pretrain generator
    if to_pretrain is True:
        pretrain_gen_train_loss = list()
        pretrain_gen_test_loss = list()

        pretrain_gen_test_seq_var = list()
        pretrain_gen_train_seq_var = list()
        pretrain_gen_batch_test_loss = list()
        pretrain_gen_batch_test_seq_var = list()

        print("Pretraining generator...")
        # balance tr data by mutations
        pretr_parent_child_mut_indices = utils.get_mutation_tr_indices(X_pretrain, y_pretrain, kmer_f_dict, kmer_r_dict, forward_dict, rev_dict)
        utils.save_as_json(PRETR_MUT_INDICES, pretr_parent_child_mut_indices)
        # get pretraining dataset as sliced tensors
        n_pretrain_batches = int(X_pretrain.shape[0]/float(batch_size))
        print("Num of pretrain batches: {}".format(str(n_pretrain_batches)))
        for i in range(pretrain_epochs):
            print("Pre training epoch {}/{}...".format(str(i+1), str(pretrain_epochs)))
            pretrain_gen_tr_loss, bat_te_gen_loss, bat_te_seq_var, bat_tr_seq_var, encoder, decoder = train_model.pretrain_generator([X_pretrain, y_pretrain, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches], i, encoder, decoder, enc_units, vocab_size, n_pretrain_batches, batch_size, pretr_parent_child_mut_indices, pretrain_epochs, size_stateful)
            print("Pre training loss at epoch {}/{}: Generator loss: {}, variation score: {}".format(str(i+1), str(pretrain_epochs), str(pretrain_gen_tr_loss), str(np.mean(bat_tr_seq_var))))
            pretrain_gen_train_loss.append(pretrain_gen_tr_loss)
            pretrain_gen_batch_test_loss.append(bat_te_gen_loss)
            pretrain_gen_batch_test_seq_var.append(bat_te_seq_var)
            pretrain_gen_train_seq_var.append(bat_tr_seq_var)
            print()
            print("Pretrain: predicting on test datasets...")
            #with tf.device('/device:cpu:0'):
            pretrain_gen_te_loss, pretrain_gen_te_seq_var = utils.predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, len_final_aa_padding, vocab_size, enc_units, encoder, decoder, size_stateful)
            pretrain_gen_test_loss.append(pretrain_gen_te_loss)
            pretrain_gen_test_seq_var.append(pretrain_gen_te_seq_var)
            print("Pre-training epoch {} finished".format(str(i+1)))
            print()
        np.savetxt(PRETRAIN_GEN_LOSS, pretrain_gen_train_loss)
        np.savetxt(PRETRAIN_GEN_TEST_LOSS, pretrain_gen_test_loss)
        np.savetxt("data/generated_files/pretrain_gen_test_seq_var.txt", pretrain_gen_test_seq_var)
        np.savetxt("data/generated_files/pretrain_gen_batch_test_loss.txt", pretrain_gen_batch_test_loss)
        np.savetxt("data/generated_files/pretrain_gen_batch_test_seq_var.txt", pretrain_gen_batch_test_seq_var)
        np.savetxt("data/generated_files/pretrain_gen_batch_train_seq_var.txt", pretrain_gen_train_seq_var)
        print("Pre-training finished")
        print()

        end_time = time.time()
        print("Pretraining finished in {} seconds".format(str(np.round(end_time - start_time, 2))))

    if gan_train is False:
        sys.exit()
    # GAN training
    # create discriminator model
    disc_parent_encoder_model, disc_gen_encoder_model = neural_network.make_disc_par_gen_model(len_final_aa_padding, vocab_size, embedding_dim, enc_units)
    discriminator = neural_network.make_discriminator_model(enc_units)

    # use the pretrained generator and train it along with discriminator
    print("Training Generator and Discriminator...")

    train_gen_total_loss = list()
    train_gen_true_loss = list()
    train_gen_fake_loss = list()
    train_disc_total_loss = list()
    train_disc_true_loss = list()
    train_disc_fake_loss = list()
    train_te_loss = list()
    train_gen_test_seq_var = list()
    train_gen_batch_test_loss = list()
    train_gen_batch_test_seq_var = list()

    n_train_batches = int(X_train.shape[0]/float(batch_size))
    print("Num of train batches: {}".format(str(n_train_batches)))

    # balance tr data by mutations
    tr_parent_child_mut_indices = dict() #utils.get_mutation_tr_indices(X_train, y_train, forward_dict, rev_dict)
    utils.save_as_json(TR_MUT_INDICES, tr_parent_child_mut_indices)

    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_true_loss, epo_gen_fake_loss, epo_total_gen_loss, epo_disc_true_loss, epo_disc_fake_loss, epo_total_disc_loss, epo_bat_te_loss, epo_bat_gen_seq_var, encoder, decoder = train_model.start_training_mut_balanced([X_train, y_train, unrelated_X, unrelated_y, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches], n, encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, tr_parent_child_mut_indices, epochs)

        print("Training loss at epoch {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(n+1), str(epochs), str(epo_gen_true_loss), str(epo_gen_fake_loss), str(epo_total_gen_loss), str(epo_disc_true_loss), str(epo_disc_fake_loss), str(epo_total_disc_loss)))

        train_gen_total_loss.append(epo_total_gen_loss)
        train_gen_true_loss.append(epo_gen_true_loss)
        train_gen_fake_loss.append(epo_gen_fake_loss)

        train_disc_total_loss.append(epo_total_disc_loss)
        train_disc_true_loss.append(epo_disc_true_loss)
        train_disc_fake_loss.append(epo_disc_fake_loss)

        train_gen_batch_test_loss.append(epo_bat_te_loss)
        train_gen_batch_test_seq_var.append(epo_bat_gen_seq_var)

        # predict seq on test data
        print("Prediction on test data...")
        with tf.device('/device:cpu:0'):
            epo_tr_gen_te_loss, epo_tr_gen_seq_var = utils.predict_sequence(test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, len_final_aa_padding, vocab_size, enc_units, encoder, decoder)
            train_te_loss.append(epo_tr_gen_te_loss)
            train_gen_test_seq_var.append(epo_tr_gen_seq_var)
        print()
    print("Training finished")
    # save loss files
    np.savetxt(TRAIN_GEN_TOTAL_LOSS, train_gen_total_loss)
    np.savetxt(TRAIN_GEN_FAKE_LOSS, train_gen_fake_loss)
    np.savetxt(TRAIN_GEN_TRUE_LOSS, train_gen_true_loss)
    np.savetxt(TRAIN_DISC_FAKE_LOSS, train_disc_fake_loss)
    np.savetxt(TRAIN_DISC_TRUE_LOSS, train_disc_true_loss)
    np.savetxt(TRAIN_DISC_TOTAL_LOSS, train_disc_total_loss)
    np.savetxt(TEST_LOSS, train_te_loss)
    np.savetxt("data/generated_files/train_gen_batch_test_loss.txt", train_gen_batch_test_loss)
    np.savetxt("data/generated_files/train_gen_batch_test_seq_var.txt", train_gen_batch_test_seq_var)
    np.savetxt("data/generated_files/train_gen_test_seq_var.txt", train_gen_test_seq_var)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))


if __name__ == "__main__":
    read_files()
