import sys
import os
import pandas as pd
import numpy as np
import json
import glob
from Bio import SeqIO
import h5py

import utils

LEN_AA = 1273
edit_threshold = 6
train_size = 0.8
l_dist_name = "levenshtein_distance"


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
    amino_acid_codes = "QNKWFPYLMTEIARGHSDVC" #"ARNDCQEGHILKMFPSTWYV"
    max_seq_size = LEN_AA
    aa_chars = utils.get_all_possible_words(amino_acid_codes)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(aa_chars)
    u_list = list()
    for sequence in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence.id.split("|")[1]
        sequence = str(sequence.seq)
        sequence = sequence.replace("*", '')
        if "X" not in sequence and seq_id in samples_clades and len(sequence) == LEN_AA:
            row.append(seq_id)
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            seq_chars = list(sequence) #[char for char in sequence]
            #u_list.extend(list(set(seq_chars)))
            #print("".join(list(set(u_list))))
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            joined_indices_kmers = ','.join(indices_chars)
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
            #total_samples += merged_size
            file_name = "data/merged_clades/{}_{}.csv".format(in_clade, out_clade)
            cross_joined_df = cross_joined_df.sample(frac=1)
            cross_columns = list(cross_joined_df.columns)
            cross_columns.append(l_dist_name)

            filtered_rows = list()
            l_distance = list()
            filtered_l_distance = list()
            parent = list()
            child = list()
            print("Filtering sequences...")
            for index, item in cross_joined_df.iterrows():
                x = item["Sequence_x"]
                y = item["Sequence_y"]
                parent.append(x)
                child.append(y)
                l_dist = utils.compute_Levenshtein_dist(x, y)
                l_distance.append(l_dist)
                if l_dist > 0 and l_dist < edit_threshold:
                    n_item = item.tolist()
                    n_item.append(l_dist)
                    filtered_rows.append(n_item)
                    filtered_l_distance.append(l_dist)
            filtered_dataframe = pd.DataFrame(filtered_rows, columns=cross_columns)
            filtered_dataframe.to_csv(file_name, index=None)

            print("Unique parents: {}".format(str(len(list(set(parent))))))
            print("Unique children: {}".format(str(len(list(set(child))))))

            np.savetxt("data/generated_files/l_distance.txt", l_distance)
            np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)

            print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
            print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))

            total_samples += len(filtered_dataframe.index)
            print("Filtered dataframe size: {}".format(str(len(filtered_dataframe.index))))

            train_df = filtered_dataframe.sample(frac=train_size, random_state=200)
            print("Converting to array...")
            train_x = train_df["Sequence_x"].tolist()
            train_y = train_df["Sequence_y"].tolist()
            train_l_dist = train_df[l_dist_name].tolist()

            merged_train_df = pd.DataFrame(list(zip(train_x, train_y, train_l_dist)), columns=["X", "Y", l_dist_name])
            tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            merged_train_df.to_csv(tr_filename, sep="\t", index=None)

            test_df = filtered_dataframe.drop(train_df.index)
            test_x = test_df["Sequence_x"].tolist()
            test_y = test_df["Sequence_y"].tolist()
            test_l_dist = test_df[l_dist_name].tolist()

            te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
            merged_test_df = pd.DataFrame(list(zip(test_x, test_y, test_l_dist)), columns=["X", "Y", l_dist_name])
            merged_test_df = merged_test_df.drop_duplicates()
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
