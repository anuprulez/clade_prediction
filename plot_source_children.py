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


data_path = "test_results/20A_20B_17Sept_CPU/"

clade_parent = "20B"
clade_childen = ["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"]


def read_json(file_path):
    with open(file_path) as file:
        return json.loads(file.read())


def write_dict(path, dic):
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    with open(path, "w") as f:
        f.write(json.dumps(dic))


def merge_clades():
    df_merged_true = None
    df_merged_gen = None
    true_ctr = 0
    gen_ctr = 0
    for c_clade in clade_childen:
        clade_path = "{}_{}".format(clade_parent, c_clade)
        true_path = data_path + clade_path + "/train/" + clade_path + ".csv"
        gen_path = data_path + clade_path + "/true_predicted_multiple_te_{}_x_1times.csv".format(clade_path)
        df_true = pd.read_csv(true_path, sep="\t")
        df_gen = pd.read_csv(gen_path, sep=",")
        true_ctr += len(df_true.index)
        gen_ctr += len(df_gen.index)
        if df_merged_true is None:
            df_merged_true = df_true            
        else:
            df_merged_true = pd.concat([df_merged_true, df_true])

        if df_merged_gen is None:
            df_merged_gen = df_gen
        else:
            df_merged_gen = pd.concat([df_merged_gen, df_gen])
    print(true_ctr, len(df_merged_true.index), gen_ctr, len(df_merged_gen.index))
    df_merged_true.to_csv(data_path + "df_merged_true.csv", sep="\t")
    df_merged_gen.to_csv(data_path + "df_merged_gen.csv", sep="\t")
    return df_merged_true, df_merged_gen


def get_mut_dict(dataframe, f_dict, col_idx):
    cols = list(dataframe.columns)
    mut_dict = dict()
    for index, row in dataframe.iterrows():
        x = row[cols[0]].split(",")
        y = row[cols[col_idx]].split(",")
        for i in range(len(x)):
            first = x[i:i+1]
            second = y[i:i+1]
            first_aa = [f_dict[j] for j in first]
            second_aa = [f_dict[j] for j in second]
            first_mut = first_aa[0]
            second_mut = second_aa[0]
            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in mut_dict:
                    mut_dict[key] = 0
                mut_dict[key] += 1
    return mut_dict


def plot_aa_transition_counts():

    df_true, df_gen = merge_clades()
    f_dict = read_json(data_path + "f_word_dictionaries.json")

    mut_parent_child = get_mut_dict(df_true, f_dict, 1)
    mut_parent_gen = get_mut_dict(df_gen, f_dict, 2)

    write_dict(data_path  + "merged_parent_child.json", mut_parent_child)
    write_dict(data_path + "merged_parent_gen.json", mut_parent_gen)

    aa_list = list('QNKWFPYLMTEIARGHSDVC')
    true_size = df_true.shape[0]
    gen_size = df_gen.shape[0]

    parent_child = dict(sorted(mut_parent_child.items(), key=lambda item: item[1], reverse=True))
    print("AA transition freq between parent-child: {}".format(parent_child))
    print("# AA transition between parent-child: {}".format(str(len(parent_child))))
    print()

    parent_gen = dict(sorted(mut_parent_gen.items(), key=lambda item: item[1], reverse=True))
    print("AA transition freq between parent-gen: {}".format(parent_gen))
    print("AA transition between parent-child: {}".format(str(len(parent_gen))))
    print()

    par_child_mat = get_mat(aa_list, parent_child, true_size)
    print()
    par_gen_mat = get_mat(aa_list, parent_gen, gen_size)

    par_child_keys = list(parent_child.keys())
    par_gen_keys = list(parent_gen.keys())

    print("Size of AA transitions - par-child, par-gen")
    print(len(par_child_keys), len(par_gen_keys))

    plot_matrix(aa_list, par_child_mat, par_gen_mat)

    print()
    print("Common AA transitions in true and gen for {}>{} branch".format(clade_parent, ",".join(clade_childen)))
    common_muts = list()
    for mut in parent_child:
        if mut in parent_gen:
            print(mut, parent_child[mut], parent_gen[mut])
            common_muts.append(mut)
    
    # plot mut that are common in true and generated datasets
    common_parent_child = dict()
    common_parent_gen = dict()
    for mut in common_muts:
        common_parent_child[mut] = parent_child[mut]
        common_parent_gen[mut] = parent_gen[mut]

    print(common_parent_child)
    print()
    print(common_parent_gen)

    common_par_child_mat = get_mat(aa_list, common_parent_child, true_size)
    common_par_gen_mat = get_mat(aa_list, common_parent_gen, gen_size)

    plot_matrix(aa_list, common_par_child_mat, common_par_gen_mat)


def plot_matrix(aa_list, par_child_mat, par_gen_mat):

    pearson_corr_te_par_child_par_gen_mut = pearsonr(par_child_mat, par_gen_mat)
    print("Pearson correlation between true par-child mut and true par-gen mut: {}".format(str(pearson_corr_te_par_child_par_gen_mut)))

    # generate plots
    cmap = "Blues"
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2)
    pos_ticks = list(np.arange(0, len(aa_list)))
    pos_labels = aa_list
    interpolation = "none"

    ax0 = axs[0].imshow(par_child_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[0].set_title("(A) Parent-child AA transition frequency")
    axs[0].set_ylabel("From")
    axs[0].set_xlabel("To")
    axs[0].set_xticks(pos_ticks)
    axs[0].set_xticklabels(pos_labels, rotation='horizontal')
    axs[0].set_yticks(pos_ticks)
    axs[0].set_yticklabels(pos_labels, rotation='horizontal')

    ax1 = axs[1].imshow(par_gen_mat, cmap=cmap,  interpolation=interpolation, aspect='auto')
    axs[1].set_title("(B) Parent-gen AA transition frequency")
    axs[1].set_ylabel("From")
    axs[1].set_xlabel("To")
    axs[1].set_xticks(pos_ticks)
    axs[1].set_xticklabels(pos_labels, rotation='horizontal')
    axs[1].set_yticks(pos_ticks)
    axs[1].set_yticklabels(pos_labels, rotation='horizontal')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(ax0, cax=cbar_ax)
    plt.suptitle("AA transition frequency in true and generated datasets. Parent: {}, children: {}. Pearson correlation of A & B: {}".format(clade_parent, ",".join(clade_childen), str(np.round(pearson_corr_te_par_child_par_gen_mut[0], 2))))
    plt.show()


def get_mat(aa_list, ct_dict, size):
    mat = np.zeros((len(aa_list), len(aa_list)))

    for i, mut_y in enumerate(aa_list):
        for j, mut_x in enumerate(aa_list):
            key = "{}>{}".format(mut_y, mut_x)
            if key in ct_dict:
                mat[i, j] = ct_dict[key]
                #print(i, j, key, ct_dict[key])
    return mat / 1.0 #size


if __name__ == "__main__":
    start_time = time.time()
    plot_aa_transition_counts()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
