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

results_path = "test_results/20A_20C_08Sept/" #20A_20C_06Sept_20EPO
clade_parent = "20A"
clade_child = "20C"
clade_end = ["20H (Beta, V2)", "20G", "21C (Epsilon)", "21F (Iota)"]
pred_file = "true_predicted_multiple.csv" #"true_predicted_df.csv"

#c_20A = ["20B", "20C", "20E (EU1)"] #["20B", "20C", "20E (EU1)", "21A (Delta)", "21B (Kappa)", "21D (Eta)"]


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



def plot_sequences(min_pos, max_pos):
    #min_pos = 1
    #max_pos = 50
    f_dict = read_json(results_path + "f_word_dictionaries.json")

    df = pd.read_csv(results_path + "sample_clade_sequence_df.csv", sep=",")

    df_tru_gen = pd.read_csv(results_path + pred_file, sep=",")
    
    df_gen = df_tru_gen["Generated"].tolist()
 
    # parent clade
    #parent = df[df["Clade"] == clade_parent]["Sequence"].tolist()
    parent = df_tru_gen["20A"].tolist()
    mat_parent = get_frac_seq_mat(parent, min_pos, max_pos)
    print(mat_parent.shape)
    print("----")

    n_parent = mat_parent.shape[0]

    # child clade
    data_child = df_tru_gen["20C"].tolist() #df[df["Clade"] == clade_child]["Sequence"].tolist()
    mat_child = get_frac_seq_mat(data_child, min_pos, max_pos)
    print(mat_child.shape)
    print("----")

    # true child > child clades
    c_seq_list = list()

    for c in clade_end:
        u_list = list()
        df_clade = df[df["Clade"] == c]
        seq = df_clade["Sequence"]
        #print(c, seq.shape)
        c_seq_list.extend(seq.tolist())

    len_c_seq = len(c_seq_list)

    mat_true = get_frac_seq_mat(c_seq_list, min_pos, max_pos)
    print(mat_true.shape)
    print("----")

    # generated child > child clades
    #gen_sampled = random.sample(df_gen, n_parent)
    
    mat_gen_sampled = get_frac_seq_mat(df_gen, min_pos, max_pos)
    print(mat_gen_sampled.shape)
    print("----")

    '''parent = df[df["Clade"] == "19A"]["Sequence"].tolist()

    gen_mut = compare_mutations(parent, mat_gen_sampled, f_dict, min_pos, max_pos)

    print(gen_mut)

    write_dict("data/generated_files/generated_mutations_c_20A.json", gen_mut)

    true_mut = compare_mutations(parent, mat_true, f_dict, min_pos, max_pos)

    print(true_mut)

    write_dict("data/generated_files/true_mutations_c_20A.json", true_mut)'''

    cmap = "RdYlBu"
    plt.rcParams.update({'font.size': 16})
    fdict_min = 0
    f_dict_max = 21
    aa_dict = f_dict
    aa_names = list(aa_dict.values())

    fig, axs = plt.subplots(4)
    #fig.suptitle('D614G mutation in spike protein: 19A, 20A, true (20B, 20C and 20E (EU1)) and generated child amino acid (AA) sequences of 20A')
    pos_labels = list(np.arange(min_pos, max_pos))
    pos_ticks = list(np.arange(0, len(pos_labels)))
    pos_labels = [i+1 for i in pos_labels]

    color_ticks = list(np.arange(0, len(aa_dict)))
    color_tick_labels = aa_names
    interpolation = "none"

    ax0 = axs[0].imshow(mat_gen_sampled, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=fdict_min, vmax=f_dict_max)
    axs[0].set_title("Generated children of {}".format(clade_child))
    #axs[0].set_xlabel("Amino acid positions")
    axs[0].set_ylabel("AA Sequences")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(mat_true, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=fdict_min, vmax=f_dict_max)
    #fig.colorbar(ax0, ax=axs[1])
    axs[1].set_title("True children of {}".format(clade_child))
    #axs[1].set_xlabel("Amino acid positions")
    axs[1].set_ylabel("AA Sequences")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')

    ax2 = axs[2].imshow(mat_child, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=fdict_min, vmax=f_dict_max)
    #fig.colorbar(ax0, ax=axs[2])
    axs[2].set_title(clade_child)
    #axs[2].set_xlabel("Amino acid positions")
    axs[2].set_ylabel("AA Sequences")
    axs[2].set_xticks(pos_ticks)
    axs[2].set_xticklabels(pos_labels, rotation='horizontal')

    ax3 = axs[3].imshow(mat_parent, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=fdict_min, vmax=f_dict_max)
    #fig.colorbar(ax0, ax=axs[3])
    axs[3].set_title(clade_parent)
    axs[3].set_xlabel("Spike protein: AA positions")
    axs[3].set_ylabel("AA Sequences")
    axs[3].set_xticks(pos_ticks)
    axs[3].set_xticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    cbar.set_ticks(color_ticks)
    cbar.ax.set_yticklabels(color_tick_labels, rotation='0')

    plt.show()

    '''plt.ylim(0, to_show_len)
    plt.imshow(mat_gen_sampled, cmap='Reds')
    plt.title("Generated children of 20A")
    plt.colorbar()
    plt.show()

    plt.imshow(mat_true, cmap='Reds')
    plt.title("True children of 20A")
    plt.ylim(0, to_show_len)
    plt.colorbar()
    plt.show()'''


def plot_l_distance():
    file_path = "data/generated_files/filtered_l_distance.txt"
    with open(file_path, "r") as l_f:
        content = l_f.read()
    content = content.split("\n")
    content = content[:len(content) - 1]
    content = [float(i) for i in content]
    print("Mean Levenshtein distance: {}".format(str(np.mean(content))))
    plt.hist(content, density=False, bins=30)
    plt.ylabel('Count')
    plt.xlabel('Levenstein distance')
    plt.show()


######################## 

def plot_mutation_counts():
    file_name = "true_predicted_multiple_te_x_1times.csv"
    df_true_pred = pd.read_csv(results_path + file_name, sep=",")
    #df_true_pred = df_true_pred[:100]
    print(df_true_pred)
    cols = list(df_true_pred.columns)
    parent_child = dict()
    parent_gen = dict()
    child_gen = dict()

    f_dict = read_json(results_path + "f_word_dictionaries.json")

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

            first_aa = [f_dict[j] for j in first]
            sec_aa = [f_dict[j] for j in sec]
            third_aa = [f_dict[j] for j in third]
        
            first_mut = first_aa[0]
            second_mut = sec_aa[0]
            third_mut = third_aa[0]

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

    write_dict(results_path + "parent_child.json", parent_child)
    write_dict(results_path + "parent_gen.json", parent_gen)

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

    par_child_mat = get_mat(aa_list, parent_child, test_size)

    par_gen_mat = get_mat(aa_list, parent_gen, test_size)

    print("Preparing train data...")
    tr_par_child_mat, tr_parent_child  = get_train_mat()

    pearson_corr_tr_par_child_mut = pearsonr(tr_par_child_mat, par_child_mat)
    pearson_corr_tr_par_child_par_gen_mut = pearsonr(tr_par_child_mat, par_gen_mat)
    pearson_corr_te_par_child_par_gen_mut = pearsonr(par_child_mat, par_gen_mat)

    print("Pearson correlation between train and test par-child mut: {}".format(str(pearson_corr_tr_par_child_mut)))
    print("Pearson correlation between train par-child mut and test par-gen mut: {}".format(str(pearson_corr_tr_par_child_par_gen_mut)))
    print("Pearson correlation between test par-child mut and par-gen mut: {}".format(str(pearson_corr_te_par_child_par_gen_mut)))

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

    # generate plots

    cmap = "Spectral" #"RdYlBu"
    plt.rcParams.update({'font.size': 8})

    fig, axs = plt.subplots(3)

    pos_ticks = list(np.arange(0, len(aa_list)))
    pos_labels = aa_list

    interpolation = "none"

    ax0 = axs[0].imshow(tr_par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[0].set_title("Train parent-child mutation frequency")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[1].set_title("Test parent-child mutation frequency")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    ax2 = axs[2].imshow(par_gen_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[2].set_title("Test parent-generated mutation frequency")
    axs[2].set_ylabel("From")
    axs[2].set_xlabel("To")
    axs[2].set_xticks(pos_ticks)
    axs[2].set_xticklabels(pos_labels, rotation='horizontal')
    axs[2].set_yticks(pos_ticks)
    axs[2].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)

    plt.show()

    # plot differences 

    diff_tr_par_child_te_par_child = par_child_mat - tr_par_child_mat
    diff_te_par_gen_te_par_child = par_gen_mat - par_child_mat
    diff_tr_par_child_te_par_gen = par_gen_mat - tr_par_child_mat

    cmap = "RdBu"
    fig, axs = plt.subplots(3)

    ax0 = axs[0].imshow(diff_tr_par_child_te_par_child, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=-0.2, vmax=0.2)
    axs[0].set_title("Test vs training")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(diff_te_par_gen_te_par_child, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=-0.2, vmax=0.2)
    axs[1].set_title("Generated vs test")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    ax2 = axs[2].imshow(diff_tr_par_child_te_par_gen, cmap=cmap,  interpolation=interpolation, aspect='auto', vmin=-0.2, vmax=0.2)
    axs[2].set_title("Generated vs training")
    axs[2].set_ylabel("From")
    axs[2].set_xlabel("To")
    axs[2].set_xticks(pos_ticks)
    axs[2].set_xticklabels(pos_labels, rotation='horizontal')
    axs[2].set_yticks(pos_ticks)
    axs[2].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    #cbar.set_ylim(-0.2, 0.2)
    plt.show()
    

def get_train_mat():
    file_name = "train/20A_20C.csv"
    df = pd.read_csv(results_path + file_name, sep="\t")
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
