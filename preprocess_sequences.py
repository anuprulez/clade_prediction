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
        samples_clades[s_name] = clade_name
    utils.save_as_json("data/generated_files/samples_clades.json", samples_clades)
    return samples_clades


def preprocess_seq(fasta_file, samples_clades, kmer_size):
    encoded_samples = list()
    possible_words = utils.get_all_possible_words(kmer_size)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(possible_words)
    utils.save_as_json("data/generated_files/f_word_dictionaries.json", f_word_dictionaries)
    utils.save_as_json("data/generated_files/r_word_dictionaries.json", r_word_dictionaries)
    print("Making Kmers...")
    for sequence in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence.id
        sequence = sequence.seq
        row.append(seq_id)
        if seq_id in samples_clades:
            clade_name = samples_clades[seq_id]
            row.append(clade_name)
            kmers = utils.make_kmers(str(sequence), size=kmer_size)
            indices_kmers = [str(r_word_dictionaries[i]) for i in kmers]
            joined_indices_kmers = ' '.join(indices_kmers)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv("data/generated_files/sample_clade_sequence_df.csv", index=None)
