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


import utils

results_path = "test_results/20A_20C_30Aug/"
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

if __name__ == "__main__":
    start_time = time.time()
    LEN_AA = 1273
    step = 50
    for i in range(0, int(LEN_AA / float(step)) + 1):
        start = i * step
        end = start + step
        #print(start, end)
        plot_sequences(start, end)
    #plot_l_distance()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
