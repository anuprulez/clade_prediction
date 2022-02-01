import time
import sys
import os

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr

import utils

data_path = "test_results/01_02_22_CPU_1/" #"test_results/19_10_20A_20B_unrolled_GPU/" # 08_10_one_hot_3_CPU_20A_20B
test_file = "test/20A_20B.csv"
gen_file = "model_generated_sequences/generated_seqs_20A_20B_1096660.csv" # generated_seqs_20A_20B_1127915 # generated_seqs_20A_20B_302510.csv

kmer_f_dict = utils.read_json(data_path + "kmer_f_word_dictionaries.json")
#parent_clade = "20B"
#child_clade = "20I_Alpha_20F_20D_21G_Lambda_21H" #"20B" #"20I_Alpha_20F_20D_21G_Lambda_21H"


def read_wuhan_hu_1_spike(r_dict):
    wuhan_seq = ""
    path = data_path + "wuhan-hu-1-spike-prot.txt"
    with open(path, "r") as f_seq:
        original_seq = f_seq.read()
    wuhan_seq = original_seq.split("\n")
    wuhan_seq = wuhan_seq[:len(wuhan_seq) - 1]
    wuhan_seq = "".join(wuhan_seq)
    enc_wuhan_seq = [str(r_dict[aa]) for aa in wuhan_seq]
    return ",".join(enc_wuhan_seq)


def read_dataframe(file_path, sep, cols, gen=False):

    #f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    #r_dict = utils.read_json(data_path + "r_word_dictionaries.json")
    dataframe = pd.read_csv(data_path + file_path, sep=sep)
    #print(dataframe)
    
    
    '''u_gen_seqs = u_gen_seqs.tolist()
    u_gen_seqs = u_gen_seqs[0].split(",")
    #print(u_gen_seqs)

    enc_original_wuhan_seq = read_wuhan_hu_1_spike(r_dict)

    original_wuhan = enc_original_wuhan_seq.split(",")'''
    #print(original_wuhan)

    x, y = dataframe[cols[0]], dataframe[cols[1]]

    #print(x)
    #print(y)

    size = len(x)

    mut = dict()
    for index, (x_seq, y_seq) in enumerate(zip(x, y)):
        #print(index)
        #print(x_seq.split(","))
        #print("-------------------------------")
        #print(y_seq.split(","))
        x_sp = x_seq.split(",")[1:]
        if gen is True:
            y_sp = y_seq.split(",")
        else:
            y_sp = y_seq.split(",")[1:]

        x_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in x_sp])
        y_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in y_sp])

        for i, (aa_x, aa_y) in enumerate(zip(x_sp, y_sp)):
            #print(i, aa_x, aa_y)
            if aa_x != aa_y: #and aa_x not in [0, "0"] and aa_y not in ["0", 0]:
                #print(i, aa_x, aa_y)
                #key = "{}>{}>{}".format(kmer_f_dict[aa_x], str(i+1), kmer_f_dict[aa_y])
                #key = "{}>{}>{}".format(aa_x, str(i+1), aa_y) 
                key = "{}>{}".format(aa_x, aa_y)
                #key = "{}".format(str(i+1))
                if key not in mut:
                    mut[key] = 0
                mut[key] += 1
    mut = {k: v for k, v in sorted(mut.items(), key=lambda item: item[1], reverse=True)}
    print(mut)
    print("----------------------------")
    return mut, size

   

def get_mat(aa_list, ct_dict, size):
    mat = np.zeros((len(aa_list), len(aa_list)))
    freq = list(ct_dict.values())
    max_freq = max(freq)
    for i, mut_y in enumerate(aa_list):
        for j, mut_x in enumerate(aa_list):
            key = "{}>{}".format(mut_y, mut_x)
            if key in ct_dict:
                norm_val = ct_dict[key] / max_freq
                #print(norm_val, ct_dict[key], max_freq)
                #if norm_val < 1.0:
                mat[i, j] = norm_val #ct_dict[key] / size
                #print(i, j, key, ct_dict[key])
    return mat    


def compute_common_muts(true_par_child, gen_par_child, te_size, gen_size):
    muts_true = len(true_par_child)
    pred_true_muts = 0
    for key in true_par_child:
        if key in gen_par_child:
            pred_true_muts += 1
            print(key, true_par_child[key] / float(te_size), gen_par_child[key] / float(gen_size))
    print(pred_true_muts / float(muts_true), pred_true_muts / float(len(gen_par_child)))
    


def plot_true_gen_dist(true_par_child, gen_par_child, te_size, gen_size):
    aa_list = list('QNKWFPYLMTEIARGHSDVC')
    par_child_mat = get_mat(aa_list, true_par_child, te_size)
    #print(par_child_mat, te_size)
    print()
    par_gen_mat = get_mat(aa_list, gen_par_child, gen_size)
    #print(par_gen_mat, gen_size)
   
    pearson_corr_te_par_child_mut = pearsonr(par_child_mat, par_gen_mat)
    print("Pearson correlation between train and test par-child mut: {}".format(str(pearson_corr_te_par_child_mut)))

    compute_common_muts(true_par_child, gen_par_child, te_size, gen_size)

    cmap = "Blues" #"RdYlBu" Spectral
    plt.rcParams.update({'font.size': 10})

    fig, axs = plt.subplots(2)

    pos_ticks = list(np.arange(0, len(aa_list)))
    pos_labels = aa_list

    interpolation = "none"

    ax0 = axs[0].imshow(par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[0].set_title("(A) Test parent-child mutation frequency")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(par_gen_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[1].set_title("(B) Test parent-gen mutation frequency")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    plt.suptitle("Mutation frequency in test and generated datasets. Pearson correlation of A & B: {}".format(str(np.round(pearson_corr_te_par_child_mut[0], 2))))
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    
    original_muts, te_size = read_dataframe(test_file, "\t", ["X", "Y"])
    gen_muts, gen_size = read_dataframe(gen_file, ",", ["20A", "Generated"], True)
    plot_true_gen_dist(original_muts, gen_muts, te_size, gen_size)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
