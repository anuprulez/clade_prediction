from Bio import SeqIO
import pandas as pd

DATA_PATH = "/data/users/backup/anup/gis_datasets/"
FASTA_PATH = "allnuc0815.fasta" #"data/generated_files/samples_clades_S.json"


def read_spike_fasta():
    sample_names = list()
    fasta_custom = ""
    i = 0
    clade_sample_count = dict()
    for sequence_obj in SeqIO.parse(FASTA_PATH, "fasta"):
        seq_id = sequence_obj.id
        sequence = str(sequence_obj.seq)
        if "Spike" in seq_id:
            print(seq_id)
            sample_names.append(seq_id)
        #fasta_custom += seq_id
        #fasta_custom += "\n\n"
        #fasta_custom += sequence
        i += 1
        if i == 10000000:
            break
    #print(fasta_custom)
    #fasta_file_name = "spike_nuc_{}.fasta".format(str(i))
    #with open(DATA_PATH + fasta_file_name, "w") as f:
    #    f.write(fasta_custom)
    df = pd.DataFrame(sample_names)
    df.to_csv(DATA_PATH + "sample_ids.csv")
    
if __name__ == "__main__":
    read_spike_fasta()
