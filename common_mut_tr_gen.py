import time
import numpy as np
import utils


results_path = "test_results/20A_20C_14Sept_CPU/"
clade_parent = "20A"
clade_child = "20B"

def find_pred_mut():
    mut_tr = utils.read_json(results_path + "tr_parent_child_pos.json")
    mut_future_true = utils.read_json(results_path + "mut_pos_parent_child.json")
    mut_future_gen = utils.read_json(results_path + "mut_pos_parent_gen.json")

    novel_mut = list()
    present_in_tr_mut = list()
    for key in mut_future_gen:
        if key not in mut_tr and key in mut_future_true:
            print(key, mut_future_gen[key], mut_future_true[key])
            novel_mut.append(key)
    print("novel mut share: {}, {}, {}".format(str(len(novel_mut) / float(len(mut_future_true))), str(len(novel_mut)), str(len(mut_future_true))))
    print("---")
    for key in mut_future_gen:
        if key in mut_future_true and key in mut_tr:
            print(key, mut_future_gen[key], mut_future_true[key], mut_tr[key])
            present_in_tr_mut.append(key)

if __name__ == "__main__":
    start_time = time.time()
    find_pred_mut()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
