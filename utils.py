import itertools

def make_kmers(seq, size):
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]
 

def get_all_possible_words(kmer_size=3, vocab="AGCT")
    return [''.join(x) for x in itertools.product(vocab, repeat=kmer_size)]
    
    
def get_words_indices(word_list):
    forward_dictionary = {i: word_list[i] for i in range(0, len(word_list))}
    reverse_dictionary = {word_list[i]: i  for i in range(0, len(word_list))}
    return forward_dictionary, reverse_dictionary
