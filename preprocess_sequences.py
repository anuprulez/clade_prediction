import sys

import pandas as pd
import numpy as np
import json
from Bio import SeqIO

import utils


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


def preprocess_seq(fasta_file, samples_clades, kmer_size):
    encoded_samples = list()
    possible_words = utils.get_all_possible_words(kmer_size)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(possible_words)
    for sequence in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence.id
        sequence = sequence.seq
        row.append(seq_id)
        if seq_id in samples_clades:
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            kmers = utils.make_kmers(str(sequence), size=kmer_size)
            indices_kmers = [str(r_word_dictionaries[i]) for i in kmers]
            joined_indices_kmers = ' '.join(indices_kmers)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv("data/generated_files/sample_clade_sequence_df.csv", index=None)
    utils.save_as_json("data/generated_files/f_word_dictionaries.json", f_word_dictionaries)
    utils.save_as_json("data/generated_files/r_word_dictionaries.json", r_word_dictionaries)
    return sample_clade_sequence_df


def make_cross_product(clade_in_clade_out, dataframe):
    total_samples = 0
    for in_clade in clade_in_clade_out:
        # get df for parent clade
        in_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == in_clade]
        print("Size of clade {}: {}".format(in_clade, str(len(in_clade_df.index))))
        # get df for child clades
        for out_clade in clade_in_clade_out[in_clade]:
            out_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == out_clade]
            # add tmp key to obtain cross join and then drop it
            in_clade_df["_tmpkey"] = np.ones(len(in_clade_df.index))
            out_clade_df["_tmpkey"] = np.ones(len(out_clade_df.index))
            cross_joined_df = pd.merge(in_clade_df, out_clade_df, on="_tmpkey").drop("_tmpkey", 1)
            print("Size of clade {}: {}".format(out_clade, str(len(out_clade_df.index))))
            product = len(in_clade_df.index) * len(out_clade_df.index)
            print("Merged size ({} * {}) : {}".format(str(len(in_clade_df.index)), str(len(out_clade_df.index), product)))
            print()
            total_samples += product
            file_name = "data/merged_clades/{}_{}.csv".format(in_clade, out_clade)
            cross_joined_df.to_csv(file_name, index=None)
    print()
    print("Total number of samples: {}".format(str(total_samples)))
