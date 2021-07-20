import itertools
import json


def make_kmers(seq, size):
    # remove all letters other than A,C,G and T
    #list(filter(lambda ch: ch in 'ACGT', kmers))
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]


def reconstruct_seq(kmers):
    reconstructed_seq = []
    for i, km in enumerate(kmers):
         if i < len(kmers) - 1:
             reconstructed_seq.append(km[0])
         else:
             reconstructed_seq.append(km)
    return "".join(reconstructed_seq)


def get_all_possible_words(kmer_size=3, vocab="AGCT"):
    return [''.join(x) for x in itertools.product(vocab, repeat=kmer_size)]


def get_words_indices(word_list):
    forward_dictionary = {i + 1: word_list[i] for i in range(0, len(word_list))}
    reverse_dictionary = {word_list[i]: i + 1  for i in range(0, len(word_list))}
    return forward_dictionary, reverse_dictionary


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content
        

def format_clade_name(c_name):
    return c_name.replace("/", "_")
    
    
def embedding_info(dict_json, X):
    return len(dict_json) + 1, len(X.loc[0]["Sequence_x"])
