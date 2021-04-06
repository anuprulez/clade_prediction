import sys

import pandas as pd
import numpy as np
import json
from Bio import SeqIO


def get_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    print(ncov_global_df)
    #sample_df = ncov_global_df[ncov_global_df["strain"] == s_line.strip()]

def preprocess_seq(fasta_file, kmer_size):
    
    kmer_size = 3

    for sequence in SeqIO.parse(fasta_file, "fasta"):
        print(sequence.id)
        print(sequence.seq)
        print(len(sequence))
        kmers = make_kmers(str(sequence.seq), size=kmer_size)
        #kmers = ' '.join(kmers)
        print(kmers)
        print(len(kmers))
        break
