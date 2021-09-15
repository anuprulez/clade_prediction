import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import glob
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr


import utils

# best results with 20A_20C_08Sept
# file name file_name_mut_ct = "true_predicted_multiple_te_x_1times.csv"
"""
20A_20C_28Aug
20A_20C_30Aug
20A_20C_31Aug
20A_20C_06Sept_20EPO
20A_20C_06Sept_10EPO
20A_20C_08Sept
20A_20C_10Sept
20A_20C_13Sept_CPU
20A_20C_14Sept_GPU

"""
clade_parent = "20C"
clade_child = "21C_Epsilon" #"21F_Iota"
results_path = "test_results/20A_20C_14Sept_CPU/" #20A_20C_06Sept_20EPO #20A_20C_14Sept_GPU
sub_path = "21C_Epsilon/"

file_name_mut_ct = "20C_21C_Epsilon/true_predicted_multiple_te_x_20C_21C_Epsilon_1times.csv"
tr_file_name = "20C_21C_Epsilon/train/20C_21C_Epsilon.csv"

#clade_end = ["20H (Beta, V2)", "20G", "21C (Epsilon)", "21F (Iota)"]
#pred_file = "true_predicted_multiple.csv" #"true_predicted_df.csv"

#c_20A = ["20B", "20C", "20E (EU1)"] #["20B", "20C", "20E (EU1)", "21A (Delta)", "21B (Kappa)", "21D (Eta)"]


        
def read_wuhan_hu_1_spike(r_dict):
    wuhan_seq = ""
    path = results_path + "wuhan-hu-1-spike-prot.txt"
    with open(path, "r") as f_seq:
        original_seq = f_seq.read()
    wuhan_seq = original_seq.split("\n")
    wuhan_seq = wuhan_seq[:len(wuhan_seq) - 1]
    wuhan_seq = "".join(wuhan_seq)
    enc_wuhan_seq = [str(r_dict[aa]) for aa in wuhan_seq]
    return ",".join(enc_wuhan_seq)

def read_json(file_path):
    with open(file_path) as file:
        return json.loads(file.read())


def write_dict(path, dic):
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    with open(path, "w") as f:
        f.write(json.dumps(dic))


def compare_mutations(parent, child, f_dict, min_pos, max_pos):
    space = 1
    mut = dict()
    for p in parent:
        p_item = p.split(",")
        p_item = np.array([int(i) for i in p_item])
        p_item = p_item[min_pos: max_pos]
        for c in child:
            for index, (p_i, c_i) in enumerate(zip(p_item, c)):
                if p_i != c_i:
                    #print(index+1, f_dict[str(p_i)], f_dict[str(c_i)])
                    if f_dict[str(p_i)] != "X" and f_dict[str(c_i)] != "X":
                        key = "{}{}{}".format(f_dict[str(p_i)], str(index+1), f_dict[str(c_i)])
                        if key not in mut:
                            mut[key] = 0
                        mut[key] += 1
    return mut


def get_frac_seq_mat(list_seq, min_pos, max_pos):
    mat = list()
    for item in list_seq:
        i_s = item.split(",")
        i_show = [int(i) for i in i_s]
        i_show = i_show[min_pos: max_pos]
        #print(i_show)
        mat.append(i_show)
    return np.array(mat)


def plot_mutation_counts():
    df_true_pred = pd.read_csv(results_path + file_name_mut_ct, sep=",")
    #df_true_pred = df_true_pred[:100]
    print(df_true_pred)
    cols = list(df_true_pred.columns)
    parent_child = dict()
    parent_gen = dict()
    child_gen = dict()
    original_gen = dict()

    f_dict = read_json(results_path + "f_word_dictionaries.json")
    r_dict = read_json(results_path + "r_word_dictionaries.json")

    enc_original_wuhan_seq = read_wuhan_hu_1_spike(r_dict)

    original_wuhan = enc_original_wuhan_seq.split(",")

    # compare differences at positions
    space = 1
    for index, row in df_true_pred.iterrows():
        true_x = row[cols[0]].split(",")
        true_y = row[cols[1]].split(",")
        pred_y = row[cols[2]].split(",")
        

        for i in range(len(true_x)):
            first = true_x[i:i+space]
            sec = true_y[i:i+space]
            third = pred_y[i:i+space]
            orin_wu = original_wuhan[i:i+space]

            first_aa = [f_dict[j] for j in first]
            sec_aa = [f_dict[j] for j in sec]
            third_aa = [f_dict[j] for j in third]
            orin_wu_aa = [f_dict[j] for j in orin_wu]
        
            first_mut = first_aa[0]
            second_mut = sec_aa[0]
            third_mut = third_aa[0]
            original_mut = orin_wu_aa[0]

            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in parent_child:
                    parent_child[key] = 0
                parent_child[key] += 1
        
            if first_mut != third_mut:
                key = "{}>{}".format(first_mut, third_mut)
                if key not in parent_gen:
                    parent_gen[key] = 0
                parent_gen[key] += 1

            if original_mut != third_mut:
                key = "{}>{}".format(original_mut, third_mut)
                if key not in original_gen:
                    original_gen[key] = 0
                original_gen[key] += 1


    write_dict(results_path + "parent_child.json", parent_child)
    write_dict(results_path + "parent_gen.json", parent_gen)
    write_dict(results_path + "original_gen.json", original_gen)

    aa_list = list('QNKWFPYLMTEIARGHSDVC')

    test_size = df_true_pred.shape[0]

    parent_child = dict(sorted(parent_child.items(), key=lambda item: item[1], reverse=True))
    print("Test: Mutation freq between parent-child: {}".format(parent_child))
    print("Test: # Mutations between parent-child: {}".format(str(len(parent_child))))
    print()

    parent_gen = dict(sorted(parent_gen.items(), key=lambda item: item[1], reverse=True))
    print("Test: Mutation freq between parent-gen: {}".format(parent_gen))
    print("Test: # Mutations between parent-child: {}".format(str(len(parent_gen))))
    print()

    parent_child = dict(sorted(original_gen.items(), key=lambda item: item[1], reverse=True))
    print("Test: Mutation freq between original-gen: {}".format(original_gen))
    print("Test: # Mutation freq between original-gen: {}".format(str(len(original_gen))))
    print()

    '''par_child_mat = get_mat(aa_list, parent_child, test_size)
    print()
    par_gen_mat = get_mat(aa_list, parent_gen, test_size)

    print("Preparing train data...")
    tr_par_child_mat, tr_parent_child  = get_train_mat()

    pearson_corr_tr_par_child_mut = pearsonr(tr_par_child_mat, par_child_mat)
    pearson_corr_tr_par_child_par_gen_mut = pearsonr(tr_par_child_mat, par_gen_mat)
    pearson_corr_te_par_child_par_gen_mut = pearsonr(par_child_mat, par_gen_mat)

    print("Pearson correlation between train and test par-child mut: {}".format(str(pearson_corr_tr_par_child_mut)))
    print("Pearson correlation between train par-child mut and test par-gen mut: {}".format(str(pearson_corr_tr_par_child_par_gen_mut)))
    print("Pearson correlation between test par-child mut and par-gen mut: {}".format(str(pearson_corr_te_par_child_par_gen_mut)))

    ## get top mut list
    n_top = 10
    tr_parent_child_top = list(tr_parent_child.items())[:n_top]
    parent_child_top = list(parent_child.items())[:n_top]
    parent_gen_top = list(parent_gen.items())[:n_top]

    print(tr_parent_child_top)
    print()
    print(parent_child_top)
    print()
    print(parent_gen_top)

    print()
    common_mutations_tr_child_gen = list()
    print("Common mutations in tr, test and gen for {}>{} branch".format(clade_parent, clade_child))
    for mut in tr_parent_child:
        if mut in parent_child and mut in parent_gen:
            print(mut, tr_parent_child[mut], parent_child[mut], parent_gen[mut])
            common_mutations_tr_child_gen.append(mut)

    utils.save_as_json(results_path + sub_path + "common_mutations_tr_child_gen.json", common_mutations_tr_child_gen)

    tr_par_child_keys = list(tr_parent_child.keys())
    te_par_child_keys = list(parent_child.keys())
    te_par_gen_keys = list(parent_gen.keys())

    print("Size of mutations - tr par-child, te par-child, te par-gen")
    print(len(tr_parent_child), len(parent_child), len(parent_gen))

    intersection_tr_par_child_te_par_child = len(list(set(tr_par_child_keys).intersection(set(te_par_child_keys)))) / float(len(tr_parent_child))
    print("% intersection between tr par-child and te par-child: {}".format(str(np.round(intersection_tr_par_child_te_par_child, 2))))

    intersection_tr_par_child_te_par_gen = len(list(set(tr_par_child_keys).intersection(set(te_par_gen_keys)))) / float(len(tr_parent_child))
    print("% intersection between tr par-child and te par-gen: {}".format(str(np.round(intersection_tr_par_child_te_par_gen, 2))))

    intersection_te_par_child_te_par_gen = len(list(set(te_par_child_keys).intersection(set(te_par_gen_keys)))) / float(len(te_par_child_keys))
    print("% intersection between te par-child and te par-gen: {}".format(str(np.round(intersection_te_par_child_te_par_gen, 2))))
    print()
    print("Common mutations in tr, test and gen for {}>{} branch".format(clade_parent, clade_child))
    for mut in tr_parent_child:
        if mut in parent_child and mut in parent_gen:
            print(mut, tr_parent_child[mut], parent_child[mut], parent_gen[mut])
    # generate plots

    cmap = "Blues" #"RdYlBu" Spectral
    plt.rcParams.update({'font.size': 8})

    fig, axs = plt.subplots(3)

    pos_ticks = list(np.arange(0, len(aa_list)))
    pos_labels = aa_list

    interpolation = "none"

    ax0 = axs[0].imshow(tr_par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[0].set_title("(A) Train parent-child mutation frequency")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[1].set_title("(B) Test parent-child mutation frequency")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    ax2 = axs[2].imshow(par_gen_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[2].set_title("(C) Test parent-generated mutation frequency")
    axs[2].set_ylabel("From")
    axs[2].set_xlabel("To")
    axs[2].set_xticks(pos_ticks)
    axs[2].set_xticklabels(pos_labels, rotation='horizontal')
    axs[2].set_yticks(pos_ticks)
    axs[2].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    plt.suptitle("Mutation frequency in test, train and generated datasets. Pearson correlation of A & B: {}, A & C: {}, B & C: {}".format(str(np.round(pearson_corr_tr_par_child_mut[0], 2)), str(np.round(pearson_corr_tr_par_child_par_gen_mut[0], 2)), str(np.round(pearson_corr_te_par_child_par_gen_mut[0], 2))))
    plt.show()

    # plot differences 

    diff_tr_par_child_te_par_child = par_child_mat - tr_par_child_mat
    diff_te_par_gen_te_par_child = par_gen_mat - par_child_mat
    diff_tr_par_child_te_par_gen = par_gen_mat - tr_par_child_mat

    cmap = "RdBu"
    fig, axs = plt.subplots(3)
    vmin = -0.08
    vmax = 0.08

    ax0 = axs[0].imshow(diff_tr_par_child_te_par_child, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=vmin, vmax=vmax) # , 
    axs[0].set_title("Test vs training")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(diff_te_par_gen_te_par_child, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title("Generated vs test")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    ax2 = axs[2].imshow(diff_tr_par_child_te_par_gen, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=vmin, vmax=vmax)
    axs[2].set_title("Generated vs training")
    axs[2].set_ylabel("From")
    axs[2].set_xlabel("To")
    axs[2].set_xticks(pos_ticks)
    axs[2].set_xticklabels(pos_labels, rotation='horizontal')
    axs[2].set_yticks(pos_ticks)
    axs[2].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    plt.suptitle("Delta of mutation frequency plots")
    plt.show()'''
    

def get_train_mat():
    
    df = pd.read_csv(results_path + tr_file_name, sep="\t")
    #df_true_pred = df_true_pred[:100]
    print(df)
    cols = list(df.columns)
    tr_parent_child = dict()

    f_dict = read_json(results_path + "f_word_dictionaries.json")

    # compare differences at positions
    space = 1
    for index, row in df.iterrows():
        true_x = row["X"].split(",")
        true_y = row["Y"].split(",")

        for i in range(len(true_x)):
            first = true_x[i:i+space]
            sec = true_y[i:i+space]

            first_aa = [f_dict[j] for j in first]
            sec_aa = [f_dict[j] for j in sec]
        
            first_mut = first_aa[0]
            second_mut = sec_aa[0]

            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in tr_parent_child:
                    tr_parent_child[key] = 0
                tr_parent_child[key] += 1

    write_dict(results_path + "tr_parent_child.json", tr_parent_child)

    aa_list = list('QNKWFPYLMTEIARGHSDVC')

    tr_parent_child = dict(sorted(tr_parent_child.items(), key=lambda item: item[1], reverse=True))
    print("Train: Mutation freq between parent-child: {}".format(tr_parent_child))
    print("Train: # Mutations between parent-child: {}".format(str(len(tr_parent_child))))
    print()

    return get_mat(aa_list, tr_parent_child, df.shape[0]), tr_parent_child


def get_mat(aa_list, ct_dict, size):
    mat = np.zeros((len(aa_list), len(aa_list)))

    for i, mut_y in enumerate(aa_list):
        for j, mut_x in enumerate(aa_list):
            key = "{}>{}".format(mut_y, mut_x)
            if key in ct_dict:
                #if ct_dict[key] > 100:
                mat[i, j] = ct_dict[key]
                #print(i, j, key, ct_dict[key])
    return mat / size


if __name__ == "__main__":
    start_time = time.time()
    '''LEN_AA = 1273
    step = 50
    for i in range(0, int(LEN_AA / float(step)) + 1):
        start = i * step
        end = start + step
        #print(start, end)
        plot_sequences(start, end)
    #plot_l_distance()'''
    plot_mutation_counts()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
