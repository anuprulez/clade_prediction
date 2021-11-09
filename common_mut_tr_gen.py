import time
import numpy as np
import utils



results_path = "test_results/02_11_20A_20B_GPU_5EPO/"
clade_parent = "20A"
clade_child = "20B"
clade_future = "20I_Alpha_20F_20D_21G_Lambda_21H" #"20I_Alpha_20F_20D_21G_Lambda_21H" 20G_21C_Epsilon_21F_Iota


'''results_path = "test_results/19_10_20A_20B_unrolled_GPU/"
clade_parent = "20A"
clade_child = "20C"
clade_future = "20G_21C_Epsilon_21F_Iota"'''


def find_pred_mut():
    mut_tr = utils.read_json(results_path + "tr_parent_child_pos_{}_{}.json".format(clade_parent, clade_child))
    mut_te = utils.read_json(results_path + "te_parent_child_pos_{}_{}.json".format(clade_parent, clade_child))

    mut_future_true = utils.read_json(results_path + "parent_child_pos_{}_{}.json".format(clade_child, clade_future))
    mut_future_gen = utils.read_json(results_path + "parent_gen_pos_{}_{}.json".format(clade_child, clade_future))

    novel_mut = list()
    novel_mut_orig = list()
    present_in_tr_mut = list()
    for key in mut_future_gen:
        if key not in mut_tr and key not in mut_te and key in mut_future_true:
            print(key, mut_future_gen[key], mut_future_true[key])
            s_key = key.split(">")
            s_key = "".join(s_key)
            novel_mut_orig.append(s_key)
            novel_mut.append(key)
    print("novel mut share: {}, {}, {}".format(str(len(novel_mut) / float(len(mut_future_true))), str(len(novel_mut)), str(len(mut_future_true))))
    utils.save_as_json(results_path + "predicted_novel_mutations_in_c_{}.json".format(clade_child), novel_mut)
    utils.save_as_json(results_path + "predicted_novel_mutations_in_c_{}_original.json".format(clade_child), novel_mut_orig)

    print("---")
    for key in mut_future_gen:
        if key in mut_future_true and key in mut_tr:
            print(key, mut_future_gen[key], mut_future_true[key], mut_tr[key])
            present_in_tr_mut.append(key)

    print("--")
    tr_pos = get_POS(mut_tr)
    print(tr_pos)
    true_pos = get_POS(mut_future_true)
    print()
    print(true_pos)
    gen_pos = get_POS(mut_future_gen)
    print()
    print(gen_pos)
    novel_pos = list()
    present_in_tr_pos = list()
    for pos in gen_pos:
        if pos in true_pos and pos not in tr_pos:
            novel_pos.append(pos)
        if pos in tr_pos:
            present_in_tr_pos.append(pos)
    print()
    print("% gen mut present in tr: {}".format(str(float(len(present_in_tr_pos))/len(tr_pos))))
    print()
    print("% novel mut pos: {}".format(str(float(len(novel_pos))/len(true_pos))))
    print()
    print(novel_pos)


def get_POS(mut_dict):
    keys = list(mut_dict.keys())
    keys_POS = [int(item.split(">")[1]) for item in keys]
    return sorted(list(set(keys_POS)))
        

if __name__ == "__main__":
    start_time = time.time()
    find_pred_mut()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
