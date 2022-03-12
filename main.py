import time
import sys
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
# comment this if running on GPU
tf.config.set_visible_devices([], 'GPU')

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
LEN_AA = 302 # 1273 for considering entire seq length
len_aa_subseq = LEN_AA
#len_final_aa_padding = len_aa_subseq + 1
len_final_aa_padding = len_aa_subseq - s_kmer + 1 # write 2 here when there is padding of zero in in and out sequences
size_stateful = 300 # 50 for 302
# Neural network parameters
embedding_dim = 128
batch_size = 8
te_batch_size = batch_size
n_te_batches = 20
enc_units = 64 # 128 for 302
pretrain_epochs = 20
epochs = 1
max_l_dist = 32
test_train_size = 0.8
#pretrain_train_size = 0.01 # all dataset as pretrain and not as test
random_clade_size = 500
to_pretrain = True
pretrained_model = False
retrain_pretrain_start_index = 0
gan_train = False
start_token = 0

pretr_lr = 0.01 #1e-2
parent_collection_start_month = "2020-01-20"
stale_folders = ["data/generated_files/", "data/train/", "data/test/", "data/tr_unrelated/", "data/te_unrelated/", "data/pretrain/", "data/validation/"]
amino_acid_codes = "QNKWFPYLMTEIARGHSDVC"


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
    if pretrained_model is False:
        print("Cleaning up stale folders...")
        utils.clean_up(stale_folders)
        clades_in_clades_out = utils.read_json(PATH_TRAINING_CLADES)
        print(clades_in_clades_out)
        print("Preprocessing sample-clade assignment file...")
        dataf = pd.read_csv(PATH_SAMPLES_CLADES, sep=",")
        filtered_dataf = preprocess_sequences.filter_samples_clades(dataf)
        
        
        unrelated_clades = utils.read_json(PATH_UNRELATED_CLADES)
        print("Generating cross product of real parent child...")
        preprocess_sequences.make_cross_product(clades_in_clades_out, filtered_dataf, len_aa_subseq, start_token, parent_collection_start_month, train_size=test_train_size, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=False)
        #print("Generating cross product of real sequences but not parent-child...")
        #preprocess_sequences.make_cross_product(unrelated_clades, filtered_dataf, len_aa_subseq, start_token, train_size=1.0, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=True)
        #sys.exit()
    else:
        if retrain_pretrain_start_index == 0:
            encoder = tf.keras.models.load_model(PRETRAIN_GEN_ENC_MODEL)
            decoder = tf.keras.models.load_model(PRETRAIN_GEN_DEC_MODEL)
        else:
            print("retrain_pretrain_start_index", retrain_pretrain_start_index)
            enc_path = "data/generated_files/pre_train/" + str(retrain_pretrain_start_index) + "/enc"
            dec_path = "data/generated_files/pre_train/" + str(retrain_pretrain_start_index) + "/dec"
            encoder = tf.keras.models.load_model(enc_path)
            decoder = tf.keras.models.load_model(dec_path)

    start_training(forward_dict, rev_dict, encoder, decoder)


def start_training(forward_dict, rev_dict, gen_encoder=None, gen_decoder=None):
    pos_variations = dict()
    pos_variations_count = dict()
    start_time = time.time()
    print("Loading datasets...")
    #pretr_clade_files = glob.glob('data/pretrain/*.csv')
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')
    
    pretr_combined_X = list()
    pretr_combined_y = list()

    '''print("Loading pre-training datasets...")
    for name in pretr_clade_files:
        pretr_clade_df = pd.read_csv(name, sep="\t")
        pretr_X = pretr_clade_df["X"].tolist()
        pretr_y = pretr_clade_df["Y"].tolist()
        pretr_combined_X.extend(pretr_X)
        pretr_combined_y.extend(pretr_y)'''

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
    print(len(unrelated_X), len(unrelated_y))

    print("train and test data sizes")
    print(len(combined_X), len(combined_y), len(combined_te_X), len(combined_te_y))

    kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
    kmer_r_dict = utils.read_json(PATH_KMER_R_DICT)

    vocab_size = len(kmer_f_dict) + 1

    print("Number of kmers: {}".format(str(len(kmer_f_dict))))
    print("Vocab size: {}".format(str(len(kmer_f_dict) + 1)))

    combined_X = np.array(combined_X)
    combined_y = np.array(combined_y)

    X_train = combined_X
    y_train = combined_y

    test_dataset_in = np.array(combined_te_X)
    test_dataset_out = np.array(combined_te_y)

    if gen_encoder is None or gen_decoder is None:
        encoder, decoder = neural_network.make_generator_model(len_final_aa_padding, vocab_size, embedding_dim, enc_units, batch_size, size_stateful)
    else:
        encoder = gen_encoder
        decoder = gen_decoder

    #print(len(pretr_combined_X))

    '''if len(pretr_combined_X) == 0:
        X_pretrain, X_train, y_pretrain, y_train  = train_test_split(combined_X, combined_y, test_size=pretrain_train_size)
        X_pretrain = np.array(X_pretrain)
        y_pretrain = np.array(y_pretrain)
        pre_train_cluster_indices, pre_train_cluster_indices_dict = utils.find_cluster_indices(y_pretrain, batch_size)
        df_pretrain = pd.DataFrame(list(zip(X_pretrain, y_pretrain)), columns=["X", "Y"])
        df_pretrain.to_csv(PRETRAIN_DATA, sep="\t", index=None)
        # save update train dataset
        df_train = pd.DataFrame(list(zip(X_train, y_train)), columns=["X", "Y"])
        df_train.to_csv(tr_clade_files[0], sep="\t", index=None)
    else: '''
    #X_pretrain = np.array(pretr_combined_X)
    #y_pretrain = np.array(pretr_combined_y)

    #print("Pretrain data sizes")
    #print(X_pretrain.shape, y_pretrain.shape)

    # divide into pretrain and train
    print("Train data sizes")
    print(X_train.shape, y_train.shape)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # pretrain generator
    if to_pretrain is True:
        
        utils.create_dirs("data/generated_files/pre_train")
        pretrain_gen_train_loss = list()
        pretrain_gen_test_loss = list()

        pretrain_gen_test_seq_var = list()
        pretrain_gen_train_seq_var = list()
        pretrain_gen_batch_test_loss = list()
        pretrain_gen_batch_test_seq_var = list()

        print("Pretraining generator...")

        # balance tr data by mutations
        pretr_parent_child_mut_indices, pos_variations, pos_variations_count = utils.get_mutation_tr_indices(X_train, y_train, kmer_f_dict, kmer_r_dict, forward_dict, rev_dict, pos_variations, pos_variations_count)
        print(pos_variations)
        print()
        print(pos_variations_count)

        pre_train_cluster_indices, pre_train_cluster_indices_dict = utils.find_cluster_indices(y_train, batch_size)

        utils.calculate_sample_weights(X_train, y_train, batch_size, pos_variations_count, pre_train_cluster_indices)

        #pre_train_cluster_indices, pre_train_cluster_indices_dict = utils.find_cluster_indices(y_train, batch_size)
        #pre_train_cluster_indices_dict = dict()
        
        mut_pattern, mut_pattern_dist, mut_pattern_dist_freq, mut_buckets = utils.create_mut_balanced_dataset(X_train, y_train, kmer_f_dict, len_final_aa_padding, batch_size)

        sys.exit()
        
        utils.save_as_json(PRETR_MUT_INDICES, pretr_parent_child_mut_indices)
        # get pretraining dataset as sliced tensors
        n_pretrain_batches = int(X_train.shape[0]/float(batch_size))
        print("Num of pretrain batches: {}".format(str(n_pretrain_batches)))
        #updated_lr = pretr_lr
        for i in range(retrain_pretrain_start_index, pretrain_epochs):
            #pretrain_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=pretr_lr)
            print("Pre training epoch {}/{}...".format(str(i+1), str(pretrain_epochs)))
            pretrain_gen_tr_loss, bat_te_gen_loss, bat_te_seq_var, bat_tr_seq_var, encoder, decoder, _ = train_model.pretrain_generator([X_train, y_train, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches], i, encoder, decoder, pretr_lr, enc_units, vocab_size, n_pretrain_batches, batch_size, pretr_parent_child_mut_indices, pretrain_epochs, size_stateful, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, pos_variations, pos_variations_count, pre_train_cluster_indices_dict, mut_pattern, mut_pattern_dist, mut_pattern_dist_freq, mut_buckets)
            print("Pre training loss at epoch {}/{}: Generator loss: {}, variation score: {}".format(str(i+1), str(pretrain_epochs), str(pretrain_gen_tr_loss), str(np.mean(bat_tr_seq_var))))
            pretrain_gen_train_loss.append(pretrain_gen_tr_loss)
            pretrain_gen_batch_test_loss.append(bat_te_gen_loss)
            pretrain_gen_batch_test_seq_var.append(bat_te_seq_var)
            pretrain_gen_train_seq_var.append(bat_tr_seq_var)
            print()
            print("Pretrain: predicting on test datasets...")
            with tf.device('/device:cpu:0'):
                pretrain_gen_te_loss, pretrain_gen_te_seq_var = utils.predict_sequence(i, 0, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, len_final_aa_padding, vocab_size, enc_units, encoder, decoder, size_stateful, "pretrain")
                pretrain_gen_test_loss.append(pretrain_gen_te_loss)
                pretrain_gen_test_seq_var.append(pretrain_gen_te_seq_var)
            print("Pre-training epoch {} finished".format(str(i+1)))
            print()
            epoch_type_name = "pretrain_epoch_{}".format(str(i+1))
            utils.save_predicted_test_data(test_dataset_in, test_dataset_out, te_batch_size, enc_units, vocab_size, len_final_aa_padding, size_stateful, epoch_type_name, PRETRAIN_GEN_ENC_MODEL, PRETRAIN_GEN_DEC_MODEL) #
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

    utils.create_dirs("data/generated_files/gan_train")
    train_cluster_indices, train_cluster_indices_dict = utils.find_cluster_indices(y_train, batch_size)
    disc_parent_encoder_model, disc_gen_encoder_model = neural_network.make_disc_par_gen_model(len_final_aa_padding, vocab_size, embedding_dim, enc_units, batch_size, size_stateful)

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
    tr_parent_child_mut_indices, pos_variations, pos_variations_count = utils.get_mutation_tr_indices(X_train, y_train, kmer_f_dict, kmer_r_dict, forward_dict, rev_dict, pos_variations, pos_variations_count)
    print(pos_variations)
    print()
    print(pos_variations_count)
    utils.save_as_json(TR_MUT_INDICES, tr_parent_child_mut_indices)
    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_true_loss, epo_gen_fake_loss, epo_total_gen_loss, epo_disc_true_loss, epo_disc_fake_loss, epo_total_disc_loss, epo_bat_te_loss, epo_bat_gen_seq_var, encoder, decoder = train_model.start_training_mut_balanced([X_train, y_train, unrelated_X, unrelated_y, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches], n, encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, tr_parent_child_mut_indices, epochs, size_stateful, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, pos_variations, pos_variations_count, train_cluster_indices_dict)

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
            epo_tr_gen_te_loss, epo_tr_gen_seq_var = utils.predict_sequence(n, 0, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, len_final_aa_padding, vocab_size, enc_units, encoder, decoder, size_stateful, "gan_train")
            train_te_loss.append(epo_tr_gen_te_loss)
            train_gen_test_seq_var.append(epo_tr_gen_seq_var)
        print()
        epoch_type_name = "gan_train_epoch_{}".format(str(n+1))
        utils.save_predicted_test_data(test_dataset_in, test_dataset_out, te_batch_size, enc_units, vocab_size, len_final_aa_padding, size_stateful, epoch_type_name, TRAIN_GEN_ENC_MODEL, TRAIN_GEN_DEC_MODEL)
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
