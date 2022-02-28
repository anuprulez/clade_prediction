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


aa_list = list('QNKWFPYLMTEIARGHSDVC')
seq_len = 303 #1273 #303 #1273
y_label_bin = 10
##### Best results with 18_02_22_0 22_02_22_0

data_path = "test_results/18_02_22_0/" #"test_results/19_10_20A_20B_unrolled_GPU/" # 08_10_one_hot_3_CPU_20A_20B
### 22_02_22_0


parent_clade = "20A"
# train dataset from pretrain
train_file = [data_path + "pretrain/pretrain.csv"]
test_file = [data_path + "test/20A_20B.csv"]


# combined dataframe for 20B (as X) and children of 20B (as Y)
'''parent_clade = "20B"
train_file = [data_path + "test_future/combined_dataframe.csv"]
test_file = [data_path + "test_future/combined_dataframe.csv"]'''


#gen_file = "model_generated_sequences/generated_seqs_20A_20B_477723_gan_train_20A.csv" # generated_seqs_20A_20B_1127915 # generated_seqs_20A_20B_302510.csv
#combined_gen_files_paths = ["model_generated_sequences/generated_seqs_20A_20B_969065_pre_train_20A.csv"]
#combined_gen_files_paths = ["model_generated_sequences/generated_seqs_20A_20B_1891906_pre_train_20B.csv"]


kmer_f_dict = utils.read_json(data_path + "kmer_f_word_dictionaries.json")
f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
r_dict = utils.read_json(data_path + "r_word_dictionaries.json")

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
    
    #aa_initial = dict()
    combined_dataframe = None
    if len(file_path) == 1:
        combined_dataframe = pd.read_csv(file_path[0], sep=sep)
    else:
        for item_path in file_path:
            if combined_dataframe is None:
                combined_dataframe = pd.read_csv(item_path, sep=sep)
            else:
                new_pd = pd.read_csv(item_path, sep=sep)
                combined_dataframe = pd.concat([combined_dataframe, new_pd])

    original_wuhan_seq, enc_original_wuhan_seq = read_wuhan_hu_1_spike(r_dict)
    original_wuhan = list(original_wuhan_seq)

    x, y = combined_dataframe[cols[0]], combined_dataframe[cols[1]]

    size = len(x)

    print("Dataset size: ", gen, size)

    mut = dict()
    mut_pos = dict()
    mut_parent_pos_wu = dict()
    mut_child_pos_wu = dict()
    mut_frequency_pos = dict()
    mut_frequency_mut_dist_pos = dict()
    wu_target_mut_frequency_mut_dist_pos = dict()


    for index, (x_seq, y_seq) in enumerate(zip(x, y)):

        x_sp = x_seq.split(",")
        y_sp = y_seq.split(",")
        x_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in x_sp])
        y_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in y_sp])

        for i, (aa_x, aa_y) in enumerate(zip(x_sp, y_sp)):
            if aa_x != aa_y:

                key = "{}>{}".format(aa_x, aa_y)
                if key not in mut:
                    mut[key] = 0
                mut[key] += 1

                key_pos = "{}>{}>{}".format(aa_x, str(i+1), aa_y)
                if key_pos not in mut_pos:
                    mut_pos[key_pos] = 0
                mut_pos[key_pos] += 1

                # collect per position mutation frequency
                if str(i+1) not in mut_frequency_pos:
                    mut_frequency_pos[str(i+1)] = 0
                mut_frequency_pos[str(i+1)] += 1

                # collect per position mutation frequency with its amino acid code
                if str(i+1) not in list(mut_frequency_mut_dist_pos.keys()):
                    aa_initial = dict()
                    for aa_code in aa_list:
                        aa_initial[aa_code] = 0
                    mut_frequency_mut_dist_pos[str(i+1)] = aa_initial
                mut_frequency_mut_dist_pos[str(i+1)][aa_y] += 1
                #print(index, "Parent>Gen: ", key_pos)
            wu_aa = original_wuhan[i]

            if wu_aa != aa_x:
                key_wu_parent = "{}>{}>{}".format(wu_aa, str(i+1), aa_x)
                if key_wu_parent not in mut_parent_pos_wu:
                    mut_parent_pos_wu[key_wu_parent] = 0
                mut_parent_pos_wu[key_wu_parent] += 1
                #print(index, "Wuhan>Parent: ", key_wu_parent)


            if wu_aa != aa_y:
                key_wu_child = "{}>{}>{}".format(wu_aa, str(i+1), aa_y)
                if key_wu_child not in mut_child_pos_wu:
                    mut_child_pos_wu[key_wu_child] = 0
                mut_child_pos_wu[key_wu_child] += 1


                if str(i+1) not in list(wu_target_mut_frequency_mut_dist_pos.keys()):
                    aa_initial_wu_target = dict()
                    for aa_code in aa_list:
                        aa_initial_wu_target[aa_code] = 0
                    wu_target_mut_frequency_mut_dist_pos[str(i+1)] = aa_initial_wu_target
                wu_target_mut_frequency_mut_dist_pos[str(i+1)][aa_y] += 1


    mut = {k: v for k, v in sorted(mut.items(), key=lambda item: item[1], reverse=True)}
    mut_pos = {k: v for k, v in sorted(mut_pos.items(), key=lambda item: item[1], reverse=True)}

    mut_parent_pos_wu = {k: v for k, v in sorted(mut_parent_pos_wu.items(), key=lambda item: item[1], reverse=True)}
    mut_child_pos_wu = {k: v for k, v in sorted(mut_child_pos_wu.items(), key=lambda item: item[1], reverse=True)}
    mut_frequency_pos = {k: v for k, v in sorted(mut_frequency_pos.items(), key=lambda item: int(item[0]), reverse=False)}
    mut_frequency_mut_dist_pos = {k: v for k, v in sorted(mut_frequency_mut_dist_pos.items(), key=lambda item: int(item[0]), reverse=False)}
    wu_target_mut_frequency_mut_dist_pos = {k: v for k, v in sorted(wu_target_mut_frequency_mut_dist_pos.items(), key=lambda item: int(item[0]), reverse=False)}

    if gen == "train":
        utils.save_as_json(data_path + "{}_tr_parent_child_pos_subs.json".format(parent_clade), mut_pos)
        utils.save_as_json(data_path + "wu_{}_tr_parent_pos_subs.json".format(parent_clade), mut_parent_pos_wu)
        utils.save_as_json(data_path + "wu_{}_tr_child_pos_subs.json".format(parent_clade), mut_child_pos_wu)
        utils.save_as_json(data_path + "{}_tr_parent_child_pos_mut_freq.json".format(parent_clade), mut_frequency_pos)
        utils.save_as_json(data_path + "{}_tr_parent_child_mut_dist_pos.json".format(parent_clade), mut_frequency_mut_dist_pos)
        utils.save_as_json(data_path + "wu_target_tr_parent_child_mut_dist_pos.json", wu_target_mut_frequency_mut_dist_pos)
    elif gen == "test":
        utils.save_as_json(data_path + "{}_te_parent_child_pos_subs.json".format(parent_clade), mut_pos)
        utils.save_as_json(data_path + "wu_{}_te_parent_pos_subs.json".format(parent_clade), mut_parent_pos_wu)
        utils.save_as_json(data_path + "wu_{}_te_child_pos_subs.json".format(parent_clade), mut_child_pos_wu)
        utils.save_as_json(data_path + "{}_te_parent_child_pos_mut_freq.json".format(parent_clade), mut_frequency_pos)
        utils.save_as_json(data_path + "{}_te_parent_child_mut_dist_pos.json".format(parent_clade), mut_frequency_mut_dist_pos)
        utils.save_as_json(data_path + "wu_target_te_parent_child_mut_dist_pos.json", wu_target_mut_frequency_mut_dist_pos)
    else:
        utils.save_as_json(data_path + "{}_parent_gen_pos_subs.json".format(parent_clade), mut_pos)
        utils.save_as_json(data_path + "wu_{}_gen_parent_pos_subs.json".format(parent_clade), mut_parent_pos_wu)
        utils.save_as_json(data_path + "wu_{}_gen_child_pos_subs.json".format(parent_clade), mut_child_pos_wu)
        utils.save_as_json(data_path + "{}_gen_parent_child_pos_mut_freq.json".format(parent_clade), mut_frequency_pos)
        utils.save_as_json(data_path + "{}_gen_parent_child_mut_dist_pos.json".format(parent_clade), mut_frequency_mut_dist_pos)
        utils.save_as_json(data_path + "wu_target_gen_parent_child_mut_dist_pos.json", wu_target_mut_frequency_mut_dist_pos)

    print("20A: {}\n".format(gen))
    
    print(mut)
    print("----------------------------")
    print(mut_pos)
    print("----------------------------")
    print("Mut Wu-Parent\n")
    print(mut_parent_pos_wu)
    print("----------------------------")
    print("Mut Wu-child\n")
    print(mut_child_pos_wu)
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


def compute_common_muts(true_par_child, gen_par_child, te_size, gen_size, comparison_type):
    muts_true = len(true_par_child)
    pred_true_muts = 0
    print(comparison_type + "\n")
    for key in true_par_child:
        if key in gen_par_child:
            pred_true_muts += 1
            print(key, true_par_child[key] / float(te_size), gen_par_child[key] / float(gen_size))
    print(pred_true_muts / float(muts_true), pred_true_muts / float(len(gen_par_child)))
    print()


def plot_true_gen_dist(tr_par_child, te_par_child, gen_par_child, tr_size, te_size, gen_size):
    #aa_list = list('QNKWFPYLMTEIARGHSDVC')

    tr_par_child_mat = get_mat(aa_list, tr_par_child, tr_size)
    te_par_child_mat = get_mat(aa_list, te_par_child, te_size)
    par_gen_mat = get_mat(aa_list, gen_par_child, gen_size)
    pearson_corr_tr_gen_mut = pearsonr(tr_par_child_mat, par_gen_mat)
    pearson_corr_tr_te_mut = pearsonr(tr_par_child_mat, te_par_child_mat)
    pearson_corr_te_gen_mut = pearsonr(te_par_child_mat, par_gen_mat)

    print("Pearson correlation between train and gen mut: {}".format(str(pearson_corr_tr_gen_mut)))
    print("Pearson correlation between train and test mut: {}".format(str(pearson_corr_tr_te_mut)))
    print("Pearson correlation between test and gen mut: {}".format(str(pearson_corr_te_gen_mut)))

    compute_common_muts(tr_par_child, gen_par_child, tr_size, gen_size, "train-gen")
    compute_common_muts(tr_par_child, te_par_child, tr_size, te_size, "train-test")
    compute_common_muts(te_par_child, gen_par_child, te_size, gen_size, "test-gen")

    cmap = "Blues" #"Blues" #"RdYlBu" Spectral
    plt.rcParams.update({'font.size': 10})

    fig, axs = plt.subplots(2)

    pos_ticks = list(np.arange(0, len(aa_list)))
    pos_labels = aa_list

    interpolation = "none"

    ax0 = axs[0].imshow(tr_par_child_mat, cmap=cmap, interpolation=interpolation, aspect='auto')
    axs[0].set_title("(A) Train parent-child substitution frequency")
    axs[0].set_ylabel("From source clade (20A)")
    axs[0].set_xlabel("To target clade (20B)")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    '''ax1 = axs[1].imshow(te_par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[1].set_title("(B) Test substitution frequency")
    axs[1].set_ylabel("From source clade (20A)")
    axs[1].set_xlabel("To target clade (20B)")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')'''

    ax1 = axs[1].imshow(par_gen_mat, cmap=cmap, interpolation=interpolation, aspect='auto')
    axs[1].set_title("(B) Generated (test) substitution frequency")
    axs[1].set_ylabel("From source clade (20A)")
    axs[1].set_xlabel("To target clade (20B)")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    #plt.suptitle("Substitution frequency among train, test and generated datasets. Pearson correlation of A & B: {}, Pearson correlation of B & C: {}, Pearson correlation of A & C: {}".format(str(np.round(pearson_corr_tr_te_mut[0], 2)), str(np.round(pearson_corr_te_gen_mut[0], 2)), str(np.round(pearson_corr_tr_gen_mut[0], 2))))
    plt.suptitle("Substitution frequency between train and generated datasets. Pearson correlation of A & B: {}".format(str(np.round(pearson_corr_tr_gen_mut[0], 2)), str(np.round(pearson_corr_te_gen_mut[0], 2)), str(np.round(pearson_corr_tr_gen_mut[0], 2))))
    plt.show()


def normalize_mat(D):
    return (D - np.min(D)) / ((np.max(D) - np.min(D)) + 1e-10)
    

def plot_mut_freq(tr_size, te_size, gen_size):

    # plot only frequencies with distribution of different substitutions
    
    tr_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_tr_parent_child_mut_dist_pos.json")
    #te_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_te_parent_child_mut_dist_pos.json")
    gen_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_gen_parent_child_mut_dist_pos.json")

    tr_freq = np.zeros((seq_len, len(list(aa_list))))
    gen_freq = np.zeros((seq_len, len(list(aa_list))))

    for i in range(seq_len):
        if str(i+1) in tr_parent_child_pos_mut_freq:
            # (D - D_min) / ((D_max - D_min) + 1e-10)
            tr_freq[i][:] = np.array(list(tr_parent_child_pos_mut_freq[str(i+1)].values())) #/ float(tr_size)
        else:
            tr_freq[i][:] = np.zeros(len(list(aa_list)))

        if str(i+1) in gen_parent_child_pos_mut_freq:
            gen_freq[i][:] = np.array(list(gen_parent_child_pos_mut_freq[str(i+1)].values())) #/ float(gen_size)
        else:
            gen_freq[i][:] = np.zeros(len(list(aa_list)))


    tr_freq = normalize_mat(tr_freq)
    gen_freq = normalize_mat(gen_freq)

    pearson_corr_tr_gen_mut_pos_freq = pearsonr(tr_freq, gen_freq)
    print(pearson_corr_tr_gen_mut_pos_freq)

    y_axis = list(np.arange(1, seq_len+1))
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(2)

    pos_ticks = list(np.arange(1, seq_len, y_label_bin))
    pos_labels = list(np.arange(1, seq_len, y_label_bin))

    interpolation = "none"

    for i in range(len(list(aa_list))):
        ax0 = axs[0].bar(y_axis, tr_freq[:, i])
        ax1 = axs[1].bar(y_axis, gen_freq[:, i])

    axs[0].set_title("(A) Train parent-child (20A-20B) substitution frequency")
    axs[0].set_ylabel("Total number of substitutions (normalized)")
    axs[0].set_xlabel("Spike protein genomic position (POS)")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')

    axs[1].set_title("(B) Generated parent-generated (20A-predicted/generated) substitution frequency")
    axs[1].set_ylabel("Total number of substitutions (normalized)")
    axs[1].set_xlabel("Spike protein genomic position (POS)")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    plt.suptitle("Substitution frequency in train and generated datasets per genomic position (POS), Pearson correlation (A & B): {}".format(str(np.round(pearson_corr_tr_gen_mut_pos_freq[0], 2))))
    plt.show()

    # plot only frequencies 
    tr_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_tr_parent_child_pos_mut_freq.json")
    #te_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_te_parent_child_pos_mut_freq.json")
    gen_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_gen_parent_child_pos_mut_freq.json")

    tr_freq = np.zeros(seq_len)
    #te_freq = np.zeros(seq_len)
    gen_freq = np.zeros(seq_len)

    for i in range(seq_len):
        if str(i+1) in tr_parent_child_pos_mut_freq:
            tr_freq[i] = np.array(tr_parent_child_pos_mut_freq[str(i+1)]) #/ float(tr_size)

        if str(i+1) in gen_parent_child_pos_mut_freq:
            gen_freq[i] = np.array(gen_parent_child_pos_mut_freq[str(i+1)]) #/ float(gen_size)

    #y_axis = list(np.arange(1, seq_len))

    tr_freq = normalize_mat(tr_freq)
    gen_freq = normalize_mat(gen_freq)

    pearson_corr_tr_gen_mut_pos_freq = pearsonr(tr_freq, gen_freq)
    print(pearson_corr_tr_gen_mut_pos_freq)

    plt.rcParams.update({'font.size': 10})

    fig, axs = plt.subplots(2)

    #pos_ticks = list(np.arange(0, seq_len, y_label_bin))
    #pos_labels = list(np.arange(0, seq_len, y_label_bin))

    interpolation = "none"

    ax0 = axs[0].bar(y_axis, tr_freq)
    axs[0].set_title("(A) Train parent-child substitution frequency")
    axs[0].set_ylabel("Total number of substitutions (normalized)")
    axs[0].set_xlabel("Spike protein genomic position (POS)")
    axs[0].set_xticks(pos_ticks)
    #axs[0].grid(True)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')


    ax1 = axs[1].bar(y_axis, gen_freq)
    axs[1].set_title("(B) Generated parent-generated substitution frequency")
    axs[1].set_ylabel("Total number of substitutions (normalized)")
    axs[1].set_xlabel("Spike protein genomic position (POS)")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    #axs[2].grid(True)
    plt.suptitle("Substitution frequency in train and generated datasets per genomic position (POS), Pearson correlation (A & B): {}".format(str(np.round(pearson_corr_tr_gen_mut_pos_freq[0], 2))))
    #plt.grid(True)
    plt.show()


    ##################################################################
    # --------------------------------------------------------------------
    # plot frequencies WU-Target

    tr_parent_child_pos_mut_freq = utils.read_json(data_path + "wu_target_tr_parent_child_mut_dist_pos.json")
    gen_parent_child_pos_mut_freq = utils.read_json(data_path + "wu_target_gen_parent_child_mut_dist_pos.json")

    tr_freq = np.zeros((seq_len, len(list(aa_list))))
    gen_freq = np.zeros((seq_len, len(list(aa_list))))

    for i in range(seq_len):
        if str(i+1) in tr_parent_child_pos_mut_freq:
            tr_freq[i][:] = np.array(list(tr_parent_child_pos_mut_freq[str(i+1)].values()))
        else:
            tr_freq[i][:] = np.zeros(len(list(aa_list)))

        if str(i+1) in gen_parent_child_pos_mut_freq:
            gen_freq[i][:] = np.array(list(gen_parent_child_pos_mut_freq[str(i+1)].values()))
        else:
            gen_freq[i][:] = np.zeros(len(list(aa_list)))

    tr_freq = normalize_mat(tr_freq)
    gen_freq = normalize_mat(gen_freq)

    pearson_corr_tr_gen_mut_pos_freq = pearsonr(tr_freq, gen_freq)
    print(pearson_corr_tr_gen_mut_pos_freq)

    #y_axis = list(np.arange(1, seq_len))
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(2)

    #pos_ticks = list(np.arange(0, seq_len, y_label_bin))
    #pos_labels = list(np.arange(0, seq_len, y_label_bin))

    interpolation = "none"

    for i in range(len(list(aa_list))):
        #print(i, y_axis, tr_freq[:, i])
        ax0 = axs[0].bar(y_axis, tr_freq[:, i])
        #ax1 = axs[1].bar(y_axis, te_freq[:, i])
        ax1 = axs[1].bar(y_axis, gen_freq[:, i])
        #print("--------")

    axs[0].set_title("(A) Train parent-child (Wu-20B) substitution frequency")
    axs[0].set_ylabel("Total number of substitutions (normalized)")
    axs[0].set_xlabel("Spike protein genomic position (POS)")
    axs[0].set_xticks(pos_ticks)
    #axs[0].grid(True)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')

    axs[1].set_title("(B) Generated parent-generated (Wu-generated) substitution frequency")
    axs[1].set_ylabel("Total number of substitutions (normalized)")
    axs[1].set_xlabel("Spike protein genomic position (POS)")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    #axs[2].grid(True)
    plt.suptitle("Wu-target: Substitution frequency in true target (20B) and generated target datasets per genomic position (POS), Pearson correlation (A & B):{}".format(str(np.round(pearson_corr_tr_gen_mut_pos_freq[0], 2))))
    #plt.grid(True)
    plt.show()
    


def extract_novel_pos_subs():
    parent_pos_sub = dict()
    child_pos_sub = dict()
    gen_pos_sub = dict()
    
    past_tr_parent_child_pos_sub = utils.read_json(data_path + "20A_tr_parent_child_pos_subs.json")
    past_te_parent_child_pos_sub = utils.read_json(data_path + "20A_te_parent_child_pos_subs.json")
    past_parent_gen_pos_sub = utils.read_json(data_path + "20A_parent_gen_pos_subs.json")


    future_tr_parent_child_pos_sub = utils.read_json(data_path + "20B_tr_parent_child_pos_subs.json")
    future_te_parent_child_pos_sub = utils.read_json(data_path + "20B_tr_parent_child_pos_subs.json")
    future_clade_parent_gen_pos_sub = utils.read_json(data_path + "20B_parent_gen_pos_subs.json")

    past_tr_parent_child_keys = list(past_tr_parent_child_pos_sub.keys())
    past_parent_gen_keys = list(past_parent_gen_pos_sub.keys())

    future_parent_child_keys = list(future_tr_parent_child_pos_sub.keys())
    future_parent_gen_keys = list(future_clade_parent_gen_pos_sub.keys())

    unique_pc_past_keys = list(set(past_tr_parent_child_keys).intersection(set(past_parent_gen_keys)))

    print("Past intersection of keys")
    print(unique_pc_past_keys, len(unique_pc_past_keys), len(unique_pc_past_keys) / float(len(past_tr_parent_child_keys)), len(unique_pc_past_keys) / float(len(past_parent_gen_keys)))
    print()
    print("Distribution of past keys")
    for upk in unique_pc_past_keys:
        print(upk, past_tr_parent_child_pos_sub[upk], past_parent_gen_pos_sub[upk])
    print()

    unique_pc_future_keys = list(set(future_parent_child_keys).intersection(set(future_parent_gen_keys)))
    print("Future intersection of keys")
    print(unique_pc_future_keys, len(unique_pc_future_keys), len(unique_pc_future_keys) / float(len(future_parent_child_keys)), len(unique_pc_future_keys) / float(len(future_parent_gen_keys)))
    print()

    print("Distribution of future keys")
    for ufk in unique_pc_future_keys:
        print(ufk, future_tr_parent_child_pos_sub[ufk], future_clade_parent_gen_pos_sub[ufk])

    print("Novel muts in future not present in training for data past-parent-child")
    novel_future_keys_unseen_in_training = list(set(unique_pc_future_keys).difference(set(past_tr_parent_child_pos_sub)))
    print(novel_future_keys_unseen_in_training, len(novel_future_keys_unseen_in_training), len(novel_future_keys_unseen_in_training) / float(len(future_parent_child_keys)))

    print()
    print("Distribution of novel future muts not seen during training")
    for key in novel_future_keys_unseen_in_training:
        print(key, future_tr_parent_child_pos_sub[key], future_clade_parent_gen_pos_sub[key])


def parse_mut_keys_freq(json_file, save_file_path):
    ref = list()
    pos = list()
    alt = list()
    freq = list()
    for key in json_file:
        s_key = key.split(">")
        ref.append(s_key[0])
        pos.append(int(s_key[1]))
        alt.append(s_key[2])
        freq.append(int(json_file[key]))

    pd_dataframe = pd.DataFrame(zip(ref, pos, alt, freq), columns=["REF", "POS", "ALT", "Frequency"])
    pd_dataframe = pd_dataframe.sort_values(by=["Frequency", "POS"], ascending=False)
    pd_dataframe.to_csv(data_path + save_file_path, index=None)
    
    print(pd_dataframe)
    print()
    

def create_tabular_files():
    #tr_parent_child_pos_mut_freq = utils.read_json(data_path + "20A_tr_parent_child_pos_subs.json")
    #parse_mut_keys_freq(tr_parent_child_pos_mut_freq, "20A_tr_parent_child_pos_subs.csv")

    # wu_20A_tr_child_pos_subs.json
    wu_20A_tr_child_pos_subs = utils.read_json(data_path + "wu_20A_tr_child_pos_subs.json")
    parse_mut_keys_freq(wu_20A_tr_child_pos_subs, "wu_tr_child_pos_subs.csv")

    # wu_20A_gen_child_pos_subs.json
    wu_20A_gen_child_pos_subs = utils.read_json(data_path + "wu_20A_gen_child_pos_subs.json")
    parse_mut_keys_freq(wu_20A_gen_child_pos_subs, "wu_gen_child_pos_subs.csv")

    
def create_fasta_file(parent_gen_dataset):
    cols = ["20A", "Generated"]
    parent_gen_dataset = pd.read_csv(parent_gen_dataset[0], sep=",")
    #print(parent_gen_dataset)
    x, y = parent_gen_dataset[cols[0]], parent_gen_dataset[cols[1]]

    for index, (x_seq, y_seq) in enumerate(zip(x, y)):
        x_sp = x_seq.split(",")
        y_sp = y_seq.split(",")
        x_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in x_sp])
        y_sp = utils.reconstruct_seq([kmer_f_dict[pos] for pos in y_sp])

        l_dist = utils.compute_Levenshtein_dist(x_sp, y_sp)
        randint = np.random.randint(0, len(x) - 1)
        print(randint, index)
        print("--")
        if index == 0:
            print(randint)
            print(x_sp)
            print()
            print(y_sp)
            print()
            print("Levenshtein distance: ", l_dist)

            with open(data_path + "source_protein.fasta", "w") as src:
                src.write(">Spike_src_{} \n {} \n\n".format(str(index), x_sp))

            with open(data_path + "target_protein.fasta", "w") as src:
                src.write(">Spike_target_{} \n {} \n\n".format(str(index), y_sp))
            break


if __name__ == "__main__":
    start_time = time.time()
    '''all_gen_paths = get_path_all_gen_files()
    print(all_gen_paths)
    

    tr_original_muts, tr_size = read_dataframe(train_file, "\t", ["X", "Y"], "train")
    te_original_muts, te_size = read_dataframe(test_file, "\t", ["X", "Y"], "test")
    gen_muts, gen_size = read_dataframe(all_gen_paths, ",", ["20A", "Generated"], "gen") #["20A", "Generated"] #["X", "Pred Y"]
    plot_true_gen_dist(tr_original_muts, te_original_muts, gen_muts, tr_size, te_size, gen_size)
    plot_mut_freq(tr_size, te_size, gen_size)
    #extract_novel_pos_subs()
    create_fasta_file(all_gen_paths)'''
    create_tabular_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
