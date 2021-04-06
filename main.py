import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging

import preprocess_sequences


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out.json"
KMER_SIZE = 3


def read_files():
    preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    #preprocess_sequences.preprocess_seq(PATH_SEQ, KMER_SIZE)
  
if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
