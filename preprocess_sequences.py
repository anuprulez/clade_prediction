import sys
import os
import pandas as pd
import numpy as np
import json
import itertools
import glob
import random
from Bio import SeqIO
import h5py

import utils

LEN_AA = 1273
l_dist_name = "levenshtein_distance"
PATH_SAMPLES_CLADES = "data/ncov_global/sample_clade_sequence_df.csv"
PATH_F_DICT = "data/ncov_global/f_word_dictionaries.json"
PATH_R_DICT = "data/ncov_global/r_word_dictionaries.json"
PATH_ALL_SAMPLES_CLADES = "data/ncov_global/samples_clades.json"


def get_galaxy_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    samples_clades = dict()
    for idx in range(len(ncov_global_df)):
        sample_row = ncov_global_df.take([idx])
        s_name = sample_row["seqName"].values[0]
        clade_name = sample_row["clade"].values[0]
        if sample_row["qc.overallStatus"].values[0] and sample_row["qc.overallStatus"].values[0] == "good":
            clade_name = utils.format_clade_name(clade_name)
            samples_clades[s_name] = clade_name
    utils.save_as_json(PATH_ALL_SAMPLES_CLADES, samples_clades)
    return samples_clades


def preprocess_seq_galaxy_clades(fasta_file, samples_clades):
    encoded_samples = list()
    amino_acid_codes = "QNKWFPYLMTEIARGHSDVC"
    max_seq_size = LEN_AA
    aa_chars = utils.get_all_possible_words(amino_acid_codes)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(aa_chars)
    all_sample_names = list(samples_clades.keys()) 
    u_list = list()
    for sequence_obj in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence_obj.id
        sequence = str(sequence_obj.seq)
        sequence = sequence.replace("*", '')
        if "X" not in sequence and all_sample_names.count(seq_id) > 0 and len(sequence) == LEN_AA:
            row.append(seq_id)
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            seq_chars = list(sequence)
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            joined_indices_kmers = ','.join(indices_chars)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv(PATH_SAMPLES_CLADES, index=None)
    utils.save_as_json(PATH_F_DICT, f_word_dictionaries)
    utils.save_as_json(PATH_R_DICT, r_word_dictionaries)
    return sample_clade_sequence_df, f_word_dictionaries, r_word_dictionaries


def filter_samples_clades(dataframe):
    f_df = list()
    for index, item in dataframe.iterrows():
        sample_name = item["SampleName"]
        s_name = sample_name.split("|")[1]
        split_s_name = s_name.split("/")
        # filter out incomplete/duplicate samples
        if len(split_s_name) > 2:
            f_df.append(item.tolist())
    new_df = pd.DataFrame(f_df, columns=list(dataframe.columns))
    return new_df

####################### OLD preprocessing #######################


'''def get_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    print(ncov_global_df)
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
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            joined_indices_kmers = ','.join(indices_chars)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv("data/generated_files/sample_clade_sequence_df.csv", index=None)
    utils.save_as_json("data/generated_files/f_word_dictionaries.json", f_word_dictionaries)
    utils.save_as_json("data/generated_files/r_word_dictionaries.json", r_word_dictionaries)
    return sample_clade_sequence_df, f_word_dictionaries, r_word_dictionaries'''

##############################################

def divide_list_tr_te(seq_list, size):
    random.shuffle(seq_list)
    tr_num = int(size * len(seq_list))
    tr_list = seq_list[:tr_num]
    te_list = seq_list[tr_num:]
    print(len(tr_list), len(te_list))
    return tr_list, te_list


def make_dataframes(l_tuples):
    filtered_test_x = list()
    filtered_true_y = list()
    for item in l_tuples:
        filtered_test_x.append(item[0])
        filtered_true_y.append(item[1])
    dataframe = pd.DataFrame(list(zip(filtered_test_x, filtered_true_y)), columns=["Sequence_x", "Sequence_y"])
    return dataframe


def make_u_combinations(u_p_list, u_c_list, size):
    p_tr, p_te = divide_list_tr_te(u_p_list, size)
    c_tr, c_te = divide_list_tr_te(u_c_list, size)
    tr_data = list(itertools.product(p_tr, c_tr))
    te_data = list(itertools.product(p_te, c_te))
    print(len(tr_data), len(te_data))
    return tr_data, te_data


def make_cross_product(clade_in_clade_out, dataframe, train_size=0.8, edit_threshold=3, random_size=200):
    total_samples = 0
    merged_train_df = None
    merged_test_df = None
    
    for in_clade in clade_in_clade_out:
        # get df for parent clade
        in_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == in_clade]
        in_clade_df = in_clade_df.sample(n=random_size, replace=True)
        in_len = len(in_clade_df.index)
        print("Size of clade {}: {}".format(in_clade, str(in_len)))
        # get df for child clades
        for out_clade in clade_in_clade_out[in_clade]:
            out_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == out_clade]
            print(out_clade_df)
            out_clade_df = out_clade_df.sample(n=random_size, replace=True)
            out_len = len(out_clade_df.index)
            # add tmp key to obtain cross join and then drop it
            in_clade_df["_tmpkey"] = np.ones(in_len)
            out_clade_df["_tmpkey"] = np.ones(out_len)
            cross_joined_df = pd.merge(in_clade_df, out_clade_df, on="_tmpkey").drop("_tmpkey", 1)
            print("Size of clade {}: {}".format(out_clade, str(out_len)))
            merged_size = in_len * out_len
            print("Merged size ({} * {}) : {}".format(str(in_len), str(out_len), merged_size))
            print()
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

            np.savetxt("data/generated_files/l_distance.txt", l_distance)
            np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)

            print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
            print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
            print("Filtered dataframe size: {}".format(str(len(filtered_dataframe.index))))

            filtered_x_y = filtered_dataframe[["Sequence_x", "Sequence_y"]]
            filtered_x_y = filtered_x_y.rename(columns={"Sequence_x": "X", "Sequence_y": "Y"})
            u_filtered_x_y = filtered_x_y.drop_duplicates()
            print("Unique size of clade combination {}_{}: {}".format(in_clade, out_clade, str(len(u_filtered_x_y.index))))
            total_samples += len(u_filtered_x_y.index)

            train_df = u_filtered_x_y.sample(frac=train_size, random_state=200)
            tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            train_df.to_csv(tr_filename, sep="\t", index=None)
            print("train size: {}".format(len(train_df.index)))

            test_df = u_filtered_x_y.drop(train_df.index)
            te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
            print("test size: {}".format(len(test_df.index)))
            test_df.to_csv(te_filename, sep="\t", index=None)
            print()
    print()
    print("Total number of samples: {}".format(str(total_samples)))


'''def make_cross_product(clade_in_clade_out, dataframe, train_size=0.8, edit_threshold=4):
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
            print("Merged raw size ({} * {}) : {}".format(str(in_len), str(out_len), merged_size))

            cross_joined_df = cross_joined_df.sample(frac=1)
            cross_columns = list(cross_joined_df.columns)

            filtered_rows = list()
            l_distance = list()
            filtered_l_distance = list()
            parent = list()
            child = list()
            print("Filtering sequences...")

            for index, item in cross_joined_df.iterrows():
                x = item["Sequence_x"]
                y = item["Sequence_y"]
                l_dist = utils.compute_Levenshtein_dist(x, y)
                
                l_distance.append(l_dist)
                if l_dist > 0 and l_dist < edit_threshold:
                    parent.append(x)
                    child.append(y)
                    filtered_l_distance.append(l_dist)
            print(filtered_l_distance)
            u_p = list(set(parent))
            u_c = list(set(child))

            print("Unique parents: {}".format(str(len(u_p))))
            print("Unique children: {}".format(str(len(u_c))))

            tr_data, te_data = make_u_combinations(u_p, u_c, train_size)

            # make cross product of unique parent and children
            test_x_true_y = list(itertools.product(u_p, u_c))
            print()

            train_df = make_dataframes(tr_data)
            test_df = make_dataframes(te_data)

            np.savetxt("data/generated_files/l_distance.txt", l_distance)
            np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)

            print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
            print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))

            train_x = train_df["Sequence_x"].tolist()
            train_y = train_df["Sequence_y"].tolist()

            merged_train_df = pd.DataFrame(list(zip(train_x, train_y)), columns=["X", "Y"])
            tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            merged_train_df.to_csv(tr_filename, sep="\t", index=None)

            test_x = test_df["Sequence_x"].tolist()
            test_y = test_df["Sequence_y"].tolist()

            te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
            merged_test_df = pd.DataFrame(list(zip(test_x, test_y)), columns=["X", "Y"])
            merged_test_df = merged_test_df.drop_duplicates()
            merged_test_df.to_csv(te_filename, sep="\t", index=None)
            print()
            break
    print()'''
        

def read_in_out(train, test):
    train_x = utils.convert_to_array(train["Sequence_x"])
    train_y = utils.convert_to_array(train["Sequence_y"])
    test_x = utils.convert_to_array(test["Sequence_x"])
    test_y = utils.convert_to_array(test["Sequence_y"])
    return train_x, train_y, test_x, test_y
