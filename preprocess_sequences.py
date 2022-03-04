import sys
import pandas as pd
from Bio import SeqIO
import datetime

import utils

l_dist_name = "levenshtein_distance"
PATH_SAMPLES_CLADES = "data/ncov_global/sample_clade_sequence_df.csv"
PATH_F_DICT = "data/ncov_global/f_word_dictionaries.json"
PATH_R_DICT = "data/ncov_global/r_word_dictionaries.json"
PATH_ALL_SAMPLES_CLADES = "data/ncov_global/samples_clades.json"
PATH_CLADE_EMERGENCE_DATES = "data/ncov_global/clade_emergence_dates.tsv"
amino_acid_codes = "QNKWFPYLMTEIARGHSDVC"


def get_galaxy_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    samples_clades = dict()
    for idx in range(len(ncov_global_df)):
        sample_row = ncov_global_df.take([idx])
        s_name = sample_row["seqName"].values[0]
        clade_name = sample_row["clade"].values[0]
        if sample_row["qc.overallStatus"].values[0] and sample_row["qc.overallStatus"].values[0] == "good":
            clade_name = utils.format_clade_name(clade_name)
            samples_clades[s_name] = clade_name
    utils.save_as_json(PATH_ALL_SAMPLES_CLADES, samples_clades)
    return samples_clades


def preprocess_seq_galaxy_clades(fasta_file, samples_clades, LEN_AA):
    encoded_samples = list()
    aa_chars = utils.get_all_possible_words(amino_acid_codes)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(aa_chars)
    all_sample_names = list(samples_clades.keys()) 
    for sequence_obj in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence_obj.id
        sequence = str(sequence_obj.seq)
        sequence = sequence.replace("*", '')
        if "X" not in sequence and all_sample_names.count(seq_id) > 0 and len(sequence) == LEN_AA:
            row.append(seq_id)
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            seq_chars = list(sequence)
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            joined_indices_kmers = ','.join(indices_chars)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv(PATH_SAMPLES_CLADES, index=None)
    utils.save_as_json(PATH_F_DICT, f_word_dictionaries)
    utils.save_as_json(PATH_R_DICT, r_word_dictionaries)
    return sample_clade_sequence_df, f_word_dictionaries, r_word_dictionaries


def filter_samples_clades(dataframe):
    f_df = list()
    for index, item in dataframe.iterrows():
        sample_name = item["SampleName"]
        s_name = sample_name.split("|")[1]
        split_s_name = s_name.split("/")
        # filter out incomplete/duplicate samples
        if len(split_s_name) > 2:
            f_df.append(item.tolist())
    new_df = pd.DataFrame(f_df, columns=list(dataframe.columns))
    return new_df


def filter_by_country():
    print()


def date_to_date(date):
    return datetime.datetime(int(date[0]), int(date[1]), int(date[2]))


def filter_by_date(dataframe, clade_name, start_date=None, buffer_days=180):
    f_df = list()
    emer_dates = pd.read_csv(PATH_CLADE_EMERGENCE_DATES, sep="\t")
    clade_emer_date = emer_dates[emer_dates["Nextstrain_clade"] == clade_name]
    if start_date is None:
        start_date = date_to_date(clade_emer_date.iloc[0]["first_sequence"].split("-"))
    else:
        start_date = date_to_date(start_date)
    max_date_range = start_date + datetime.timedelta(days=buffer_days)
    print(start_date, max_date_range)
    for index, item in dataframe.iterrows():
        try:
            sample_name = item["SampleName"].split("/")
            row_date = sample_name[3].split("|")
            sample_date = date_to_date(row_date[1].split("-"))
            if sample_date >= start_date and sample_date <= max_date_range:
                f_df.append(item.tolist())
        except:
            continue
    new_df = pd.DataFrame(f_df, columns=list(dataframe.columns))
    return new_df, max_date_range


def make_cross_product(clade_in_clade_out, dataframe, len_aa_subseq, start_token, collection_start_month, train_size=0.8, edit_threshold=3, random_size=200, replace=False, unrelated=False):
    total_samples = 0

    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)

    all_clades = dataframe["Clade"].tolist()
    print("All uniques clades: ", list(set(all_clades)))

    for in_clade in clade_in_clade_out:
        # get df for parent clade
        in_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == in_clade]
        print("Clade name: {}".format(in_clade))
        print(in_clade_df)
        print("Filtering by date...")
        in_clade_df, max_parent_range = filter_by_date(in_clade_df, in_clade, None)
        print(in_clade_df)
        print("Remove duplicate sequences...")
        in_clade_df = in_clade_df.drop_duplicates(subset=['Sequence'])
        print(in_clade_df)
        try:
            in_clade_df = in_clade_df.sample(n=random_size, replace=False)
        except:
            in_clade_df = in_clade_df.sample(n=random_size, replace=True)
        in_len = len(in_clade_df.index)
        print("Size of clade {}: {}".format(in_clade, str(in_len)))
        in_clade_df.to_csv("data/generated_files/in_clade_df.csv")
        in_clade_seq = in_clade_df["Sequence"]
        u_in_clade = in_clade_seq.drop_duplicates()
        u_in_clade = u_in_clade.tolist()
        print("Unique in clade size: ", len(u_in_clade))
        print(clade_in_clade_out[in_clade])
        out_clades = clade_in_clade_out[in_clade]
        for out_clade in out_clades:
            if not out_clade in all_clades:
                print("{} not present in dataframe".format(out_clade))
                continue
            if unrelated is False:
                te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
                tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            else:
               te_filename = "data/te_unrelated/{}_{}.csv".format(in_clade, out_clade)
               tr_filename = "data/tr_unrelated/{}_{}.csv".format(in_clade, out_clade)
            out_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == out_clade]
            print(out_clade_df)
            print(max_parent_range, max_parent_range.strftime("%Y-%m-%d"))
            print("Filtering by date...")
            out_clade_df, _ = filter_by_date(out_clade_df, out_clade, max_parent_range.strftime("%Y-%m-%d").split("-"))
            print(out_clade_df)
            print("Remove duplicate sequences...")
            out_clade_df = out_clade_df.drop_duplicates(subset=['Sequence'])
            print(out_clade_df)
            out_clade_df.to_csv("data/generated_files/out_clade_df.csv")
            try:
                out_clade_df = out_clade_df.sample(n=random_size, replace=False)
            except:
                out_clade_df = out_clade_df.sample(n=random_size, replace=True)
            out_len = len(out_clade_df.index)
            print("Size of clade {}: {}".format(out_clade, str(out_len)))
            out_clade_seq = out_clade_df["Sequence"]
            u_out_clade = out_clade_seq.drop_duplicates()
            u_out_clade = u_out_clade.tolist()
            #u_filtered_x_y, kmer_f_dict, kmer_r_dict = utils.generate_cross_product(u_in_clade, u_out_clade, edit_threshold, len_aa_subseq, forward_dict, start_token, unrelated=unrelated)
            u_filtered_x_y, kmer_f_dict, kmer_r_dict = utils.generate_cross_product(u_in_clade, u_out_clade, edit_threshold, len_aa_subseq, forward_dict, rev_dict, start_token, unrelated=unrelated)
            print("Unique size of clade combination {}_{}: {}".format(in_clade, out_clade, str(len(u_filtered_x_y.index))))
            total_samples += len(u_filtered_x_y.index)
            train_df = u_filtered_x_y.sample(frac=train_size, random_state=200)
            test_df = u_filtered_x_y.drop(train_df.index)
            # convert to original seq and then to Kmers
            #print("Converting to Kmers...")
            #train_df = utils.ordinal_to_kmer(train_df, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, s_kmer)
            #test_df = utils.ordinal_to_kmer(test_df, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict, s_kmer)

            train_df.to_csv(tr_filename, sep="\t", index=None)
            test_df.to_csv(te_filename, sep="\t", index=None)
            print("train size: {}".format(len(train_df.index)))
            print("test size: {}".format(len(test_df.index)))
            print()
    print()
    print("Total number of samples: {}".format(str(total_samples)))

####################### OLD preprocessing #######################


'''def get_samples_clades(path_seq_clades):
    ncov_global_df = pd.read_csv(path_seq_clades, sep="\t")
    print(ncov_global_df)
    samples_clades = dict()
    for idx in range(len(ncov_global_df)):
        sample_row = ncov_global_df.take([idx])
        s_name = sample_row["strain"].values[0]
        clade_name = sample_row["Nextstrain_clade"].values[0]
        clade_name = utils.format_clade_name(clade_name)
        samples_clades[s_name] = clade_name
    utils.save_as_json("data/generated_files/samples_clades.json", samples_clades)
    return samples_clades


def preprocess_seq(fasta_file, samples_clades):
    encoded_samples = list()
    amino_acid_codes = "QNKWFPYLMTEIARGHSDVC" #"ARNDCQEGHILKMFPSTWYV"
    max_seq_size = LEN_AA
    aa_chars = utils.get_all_possible_words(amino_acid_codes)
    f_word_dictionaries, r_word_dictionaries = utils.get_words_indices(aa_chars)
    u_list = list()
    for sequence in SeqIO.parse(fasta_file, "fasta"):
        row = list()
        seq_id = sequence.id.split("|")[1]
        sequence = str(sequence.seq)
        sequence = sequence.replace("*", '')
       
        if "X" not in sequence and seq_id in samples_clades and len(sequence) == LEN_AA:
            row.append(seq_id)
            clade_name = samples_clades[seq_id]
            clade_name = utils.format_clade_name(clade_name)
            row.append(clade_name)
            seq_chars = list(sequence) #[char for char in sequence]
            indices_chars = [str(r_word_dictionaries[i]) for i in seq_chars]
            joined_indices_kmers = ','.join(indices_chars)
            row.append(joined_indices_kmers)
            encoded_samples.append(row)
    sample_clade_sequence_df = pd.DataFrame(encoded_samples, columns=["SampleName", "Clade", "Sequence"])
    sample_clade_sequence_df.to_csv("data/generated_files/sample_clade_sequence_df.csv", index=None)
    utils.save_as_json("data/generated_files/f_word_dictionaries.json", f_word_dictionaries)
    utils.save_as_json("data/generated_files/r_word_dictionaries.json", r_word_dictionaries)
    return sample_clade_sequence_df, f_word_dictionaries, r_word_dictionaries

def make_cross_product(clade_in_clade_out, dataframe, train_size=0.8, edit_threshold=4):
    total_samples = 0
    merged_train_df = None
    merged_test_df = None
    
    for in_clade in clade_in_clade_out:
        # get df for parent clade
        in_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == in_clade]
        in_len = len(in_clade_df.index)
        print("Size of clade {}: {}".format(in_clade, str(in_len)))
        # get df for child clades
        for out_clade in clade_in_clade_out[in_clade]:
            out_clade_df = dataframe[dataframe["Clade"].replace("/", "_") == out_clade]
            out_len = len(out_clade_df.index)
            # add tmp key to obtain cross join and then drop it
            in_clade_df["_tmpkey"] = np.ones(in_len)
            out_clade_df["_tmpkey"] = np.ones(out_len)
            cross_joined_df = pd.merge(in_clade_df, out_clade_df, on="_tmpkey").drop("_tmpkey", 1)
            print("Size of clade {}: {}".format(out_clade, str(out_len)))
            merged_size = in_len * out_len
            print("Merged raw size ({} * {}) : {}".format(str(in_len), str(out_len), merged_size))

            cross_joined_df = cross_joined_df.sample(frac=1)
            cross_columns = list(cross_joined_df.columns)

            filtered_rows = list()
            l_distance = list()
            filtered_l_distance = list()
            parent = list()
            child = list()
            print("Filtering sequences...")

            for index, item in cross_joined_df.iterrows():
                x = item["Sequence_x"]
                y = item["Sequence_y"]
                l_dist = utils.compute_Levenshtein_dist(x, y)
                
                l_distance.append(l_dist)
                if l_dist > 0 and l_dist < edit_threshold:
                    parent.append(x)
                    child.append(y)
                    filtered_l_distance.append(l_dist)
            print(filtered_l_distance)
            u_p = list(set(parent))
            u_c = list(set(child))

            print("Unique parents: {}".format(str(len(u_p))))
            print("Unique children: {}".format(str(len(u_c))))

            tr_data, te_data = make_u_combinations(u_p, u_c, train_size)

            # make cross product of unique parent and children
            test_x_true_y = list(itertools.product(u_p, u_c))
            print()

            train_df = make_dataframes(tr_data)
            test_df = make_dataframes(te_data)

            np.savetxt("data/generated_files/l_distance.txt", l_distance)
            np.savetxt("data/generated_files/filtered_l_distance.txt", filtered_l_distance)

            print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
            print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))

            train_x = train_df["Sequence_x"].tolist()
            train_y = train_df["Sequence_y"].tolist()

            merged_train_df = pd.DataFrame(list(zip(train_x, train_y)), columns=["X", "Y"])
            tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
            merged_train_df.to_csv(tr_filename, sep="\t", index=None)

            test_x = test_df["Sequence_x"].tolist()
            test_y = test_df["Sequence_y"].tolist()

            te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
            merged_test_df = pd.DataFrame(list(zip(test_x, test_y)), columns=["X", "Y"])
            merged_test_df = merged_test_df.drop_duplicates()
            merged_test_df.to_csv(te_filename, sep="\t", index=None)
            print()
            break
    print()

def divide_list_tr_te(seq_list, size):
    random.shuffle(seq_list)
    tr_num = int(size * len(seq_list))
    tr_list = seq_list[:tr_num]
    te_list = seq_list[tr_num:]
    print(len(tr_list), len(te_list))
    return tr_list, te_list


def make_dataframes(l_tuples):
    filtered_test_x = list()
    filtered_true_y = list()
    for item in l_tuples:
        filtered_test_x.append(item[0])
        filtered_true_y.append(item[1])
    dataframe = pd.DataFrame(list(zip(filtered_test_x, filtered_true_y)), columns=["Sequence_x", "Sequence_y"])
    return dataframe


def make_u_combinations(u_p_list, u_c_list, size):
    p_tr, p_te = divide_list_tr_te(u_p_list, size)
    c_tr, c_te = divide_list_tr_te(u_c_list, size)
    tr_data = list(itertools.product(p_tr, c_tr))
    te_data = list(itertools.product(p_te, c_te))
    print(len(tr_data), len(te_data))
    return tr_data, te_data'''

##############################################
