import sys
import os
import pandas as pd
import numpy as np
import json
import glob
from Bio import SeqIO
import h5py

import utils

LEN_AA = 1275


def get_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    samples_clades = dict()
    for idx in range(len(ncov_global_df)):
        sample_row = ncov_global_df.take([idx])
        s_name = sample_row["strain"].values[0]
        clade_name = sample_row["Nextstrain_clade"].values[0]
        clade_name = utils.format_clade_name(clade_name)
        samples_clades[s_name] = clade_name
    utils.save_as_json("data/generated_files/samples_clades.json", samples_clades)
    return samples_clades


def preprocess_seq(fasta_file, samples_clades):
    encoded_samples = list()
    amino_acid_codes = "ARNDCQEGHILKMFPOSUTWYVBZXJ"
    max_seq_size = LEN_AA
    aa_chars = utils.get_all_possible_words(amino_acid_codes)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(aa_chars)
    for sequence in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence.id.split("|")[1]
        sequence = str(sequence.seq)
        sequence = sequence.replace("*", '')
        if seq_id in samples_clades:
            row.append(seq_id)
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            seq_chars = [char for char in sequence]
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            if len(sequence) <= LEN_AA:
                zeros = np.repeat('0', (LEN_AA - len(sequence)))
                indices_kmers = np.hstack([indices_chars, zeros])
                joined_indices_kmers = ','.join(indices_kmers)
                row.append(joined_indices_kmers)
                encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv("data/generated_files/sample_clade_sequence_df.csv", index=None)
    utils.save_as_json("data/generated_files/f_word_dictionaries.json", f_word_dictionaries)
    utils.save_as_json("data/generated_files/r_word_dictionaries.json", r_word_dictionaries)
    return sample_clade_sequence_df, f_word_dictionaries, r_word_dictionaries


def make_cross_product(clade_in_clade_out, dataframe):
    total_samples = 0
    merged_train_df = None
    merged_test_df = None
    
    for in_clade in clade_in_clade_out:
        # get df for parent clade
        in_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == in_clade]
        in_len = len(in_clade_df.index)
        print("Size of clade {}: {}".format(in_clade, str(in_len)))
        # get df for child clades
        for out_clade in clade_in_clade_out[in_clade]:
            out_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == out_clade]
            out_len = len(out_clade_df.index)
            # add tmp key to obtain cross join and then drop it
            in_clade_df["_tmpkey"] = np.ones(in_len)
            out_clade_df["_tmpkey"] = np.ones(out_len)
            cross_joined_df = pd.merge(in_clade_df, out_clade_df, on="_tmpkey").drop("_tmpkey", 1)
            print("Size of clade {}: {}".format(out_clade, str(out_len)))
            merged_size = in_len * out_len
            print("Merged size ({} * {}) : {}".format(str(in_len), str(out_len), merged_size))
            print()
            total_samples += merged_size
            file_name = "data/merged_clades/{}_{}.csv".format(in_clade, out_clade)
            train_size = 0.8
            cross_joined_df = cross_joined_df.sample(frac=1)

            train_df = cross_joined_df.sample(frac=train_size, random_state=200)
            
            print("Converting to array...")
            
            train_x = train_df["Sequence_x"].tolist()
            train_y = train_df["Sequence_y"].tolist()
            print(train_df.shape)
            test_df = cross_joined_df.drop(train_df.index)
            print(test_df.shape)
            test_x = test_df["Sequence_x"].tolist()
            test_y = test_df["Sequence_y"].tolist()
            merged_train_df = pd.DataFrame(list(zip(train_x, train_y)), columns=["X", "Y"])          
            print(merged_train_df)
            tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            merged_train_df.to_csv(tr_filename, sep="\t", index=None)
            
            te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
            merged_test_df = pd.DataFrame(list(zip(test_x, test_y)), columns=["X", "Y"])
            merged_test_df.to_csv(te_filename, sep="\t", index=None)

    print()
    print("Total number of samples: {}".format(str(total_samples)))
    
    
def transform_encoded_samples(train_size=0.8):
    clade_files = glob.glob('data/merged_clades/*.csv')
    train_df = None
    test_df = None
    for name in clade_files:
        file_path_w_ext = os.path.splitext(name)[0]
        file_name_w_ext = os.path.basename(file_path_w_ext)
        clade_df = pd.read_csv(name, sep="\t")
        # randomize rows
        clade_df = clade_df.sample(frac=1)
        train_df = clade_df.sample(frac=train_size, random_state=200)
        test_df = clade_df.drop(train_df.index)
        train_file_path = "data/train/{}.csv".format(file_name_w_ext)
        test_file_path = "data/test/{}.csv".format(file_name_w_ext)
        #train_df = train_df.drop(["SampleName_x", "Clade_x", "SampleName_y", "Clade_y"], axis=1)
        #test_df = test_df.drop(["SampleName_x", "Clade_x", "SampleName_y", "Clade_y"], axis=1)
        train_df.to_csv(train_file_path, sep="\t", index=None)
        test_df.to_csv(test_file_path, sep="\t", index=None)
    return read_in_out(train_df, test_df)
        

def read_in_out(train, test):
    train_x = utils.convert_to_array(train["Sequence_x"])
    train_y = utils.convert_to_array(train["Sequence_y"])
    test_x = utils.convert_to_array(test["Sequence_x"])
    test_y = utils.convert_to_array(test["Sequence_y"])
    return train_x, train_y, test_x, test_y
