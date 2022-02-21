import time
import sys
import os

import pandas as pd
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr

import utils

data_path = "test_results/18_02_22_0/" #"test_results/19_10_20A_20B_unrolled_GPU/" # 08_10_one_hot_3_CPU_20A_20B
#test_file = [data_path + "test/20A_20B.csv"]
#test_file = [data_path + "test/20A_20B.csv"]
parent_clade = "20B"
test_file = [data_path + "test_future/combined_dataframe.csv"] # combined dataframe for 20B (as X) and children of 20B (as Y)
#gen_file = "model_generated_sequences/generated_seqs_20A_20B_477723_gan_train_20A.csv" # generated_seqs_20A_20B_1127915 # generated_seqs_20A_20B_302510.csv
#combined_gen_files_paths = ["model_generated_sequences/generated_seqs_20A_20B_969065_pre_train_20A.csv"]
#combined_gen_files_paths = ["model_generated_sequences/generated_seqs_20A_20B_1891906_pre_train_20B.csv"]

kmer_f_dict = utils.read_json(data_path + "kmer_f_word_dictionaries.json")
#parent_clade = "20B"
#child_clade = "20I_Alpha_20F_20D_21G_Lambda_21H" #"20B" #"20I_Alpha_20F_20D_21G_Lambda_21H"


def get_path_all_gen_files(path=data_path + "model_generated_sequences"):    
    all_gen_files = glob.glob(path + '/*.csv')
    return all_gen_files


def read_wuhan_hu_1_spike(r_dict):
    wuhan_seq = ""
    path = data_path + "wuhan-hu-1-spike-prot.txt"
    with open(path, "r") as f_seq:
        original_seq = f_seq.read()
    wuhan_seq = original_seq.split("\n")
    wuhan_seq = wuhan_seq[:len(wuhan_seq) - 1]
    wuhan_seq = "".join(wuhan_seq)
    #print(wuhan_seq, len(wuhan_seq))
    enc_wuhan_seq = [str(r_dict[aa]) for aa in wuhan_seq]
    return wuhan_seq, ",".join(enc_wuhan_seq)


def read_dataframe(file_path, sep, cols, gen=False):

    f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    r_dict = utils.read_json(data_path + "r_word_dictionaries.json")
    combined_dataframe = None
    if len(file_path) == 1:
        #print(file_path)
        combined_dataframe = pd.read_csv(file_path[0], sep=sep)
    else:
        for item_path in file_path:
            if combined_dataframe is None:
                combined_dataframe = pd.read_csv(item_path, sep=sep)
            else:
                new_pd = pd.read_csv(item_path, sep=sep)
                #print(new_pd)
                combined_dataframe = pd.concat([combined_dataframe, new_pd])

    original_wuhan_seq, enc_original_wuhan_seq = read_wuhan_hu_1_spike(r_dict)
    original_wuhan = list(original_wuhan_seq)

    x, y = combined_dataframe[cols[0]], combined_dataframe[cols[1]]

    size = len(x)

    mut = dict()
    mut_pos = dict()
    mut_parent_pos_wu = dict()
    mut_child_pos_wu = dict()

    for index, (x_seq, y_seq) in enumerate(zip(x, y)):

        x_sp = x_seq.split(",")
        y_sp = y_seq.split(",")
        x_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in x_sp])
        y_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in y_sp])

        for i, (aa_x, aa_y) in enumerate(zip(x_sp, y_sp)):
            #print(i, aa_x, aa_y)
            if aa_x != aa_y: #and aa_x not in [0, "0"] and aa_y not in ["0", 0]:

                key = "{}>{}".format(aa_x, aa_y)
                #key = "{}".format(str(i+1))
                if key not in mut:
                    mut[key] = 0
                mut[key] += 1

                key_pos = "{}>{}>{}".format(aa_x, str(i+1), aa_y)
                if key_pos not in mut_pos:
                    mut_pos[key_pos] = 0
                mut_pos[key_pos] += 1

            wu_aa = original_wuhan[i]

            if wu_aa != aa_x:
                key_wu_parent = "{}>{}>{}".format(wu_aa, str(i+1), aa_x)
                if key_wu_parent not in mut_parent_pos_wu:
                    mut_parent_pos_wu[key_wu_parent] = 0
                mut_parent_pos_wu[key_wu_parent] += 1

            if wu_aa != aa_y:
                key_wu_child = "{}>{}>{}".format(wu_aa, str(i+1), aa_y)
                if key_wu_child not in mut_child_pos_wu:
                    mut_child_pos_wu[key_wu_child] = 0
                mut_child_pos_wu[key_wu_child] += 1


    mut = {k: v for k, v in sorted(mut.items(), key=lambda item: item[1], reverse=True)}
    mut_pos = {k: v for k, v in sorted(mut_pos.items(), key=lambda item: item[1], reverse=True)}

    mut_parent_pos_wu = {k: v for k, v in sorted(mut_parent_pos_wu.items(), key=lambda item: item[1], reverse=True)}
    mut_child_pos_wu = {k: v for k, v in sorted(mut_child_pos_wu.items(), key=lambda item: item[1], reverse=True)}

    if gen == False:
        utils.save_as_json(data_path + "{}_parent_child_pos_subs.json".format(parent_clade), mut_pos)
    else:
        utils.save_as_json(data_path + "{}_parent_gen_pos_subs.json".format(parent_clade), mut_pos)
    
    print(mut)
    print("----------------------------")
    print(mut_pos)
    print("----------------------------")
    '''print("Mut Wu-Parent\n")
    print(mut_parent_pos_wu)
    print("----------------------------")
    print("Mut Wu-child\n")
    print(mut_child_pos_wu)'''
    print("================================")
    return mut, size


def get_mat(aa_list, ct_dict, size):
    mat = np.zeros((len(aa_list), len(aa_list)))
    freq = list(ct_dict.values())
    max_freq = np.sum(freq) #max(freq)
    for i, mut_y in enumerate(aa_list):
        for j, mut_x in enumerate(aa_list):
            key = "{}>{}".format(mut_y, mut_x)
            if key in ct_dict:
                norm_val = ct_dict[key] / max_freq
                #print(norm_val, ct_dict[key], max_freq)
                #if norm_val < 1.0:
                mat[i, j] = ct_dict[key] / size
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

    cmap = "Blues" #"Blues" #"RdYlBu" Spectral
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


def extract_novel_pos_subs():
    parent_pos_sub = dict()
    child_pos_sub = dict()
    gen_pos_sub = dict()
    
    past_parent_child_pos_sub = utils.read_json(data_path + "20A_parent_child_pos_subs.json")
    past_parent_gen_pos_sub = utils.read_json(data_path + "20A_parent_gen_pos_subs.json")
    future_parent_child_pos_sub = utils.read_json(data_path + "20B_parent_child_pos_subs.json")
    future_clade_parent_gen_pos_sub = utils.read_json(data_path + "20B_parent_gen_pos_subs.json")

    
    past_parent_child_keys = list(past_parent_child_pos_sub.keys())
    past_parent_gen_keys = list(past_parent_gen_pos_sub.keys())
    future_parent_child_keys = list(future_parent_child_pos_sub.keys())
    future_parent_gen_keys = list(future_clade_parent_gen_pos_sub.keys())

    unique_pc_past_keys = list(set(past_parent_child_keys).intersection(set(past_parent_gen_keys)))

    print("Past intersection of keys")
    print(unique_pc_past_keys, len(unique_pc_past_keys), len(unique_pc_past_keys) / float(len(past_parent_child_keys)), len(unique_pc_past_keys) / float(len(past_parent_gen_keys)))
    print()
    print("Distribution of past keys")
    for upk in unique_pc_past_keys:
        print(upk, past_parent_child_pos_sub[upk], past_parent_gen_pos_sub[upk])
    print()

    unique_pc_future_keys = list(set(future_parent_child_keys).intersection(set(future_parent_gen_keys)))
    print("Future intersection of keys")
    print(unique_pc_future_keys, len(unique_pc_future_keys), len(unique_pc_future_keys) / float(len(future_parent_child_keys)), len(unique_pc_future_keys) / float(len(future_parent_gen_keys)))
    
    print()

    print("Distribution of future keys")
    for ufk in unique_pc_future_keys:
        print(ufk, future_parent_child_pos_sub[ufk], future_clade_parent_gen_pos_sub[ufk])

    print("Novel muts in future not present in training for data past-parent-child")
    novel_future_keys_unseen_in_training = list(set(unique_pc_future_keys).difference(set(past_parent_child_pos_sub)))
    print(novel_future_keys_unseen_in_training, len(novel_future_keys_unseen_in_training), len(novel_future_keys_unseen_in_training) / float(len(future_parent_child_keys)))

    print()
    print("Distribution of novel future muts not seen during training")
    for key in novel_future_keys_unseen_in_training:
        print(key, future_parent_child_pos_sub[key], future_clade_parent_gen_pos_sub[key])


if __name__ == "__main__":
    start_time = time.time()
    '''all_gen_paths = get_path_all_gen_files()
    print(all_gen_paths)

    original_muts, te_size = read_dataframe(test_file, "\t", ["X", "Y"])
    gen_muts, gen_size = read_dataframe(all_gen_paths, ",", ["20A", "Generated"], True) #["20A", "Generated"] #["X", "Pred Y"]
    plot_true_gen_dist(original_muts, gen_muts, te_size, gen_size)'''


    extract_novel_pos_subs()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
