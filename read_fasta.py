FASTA_PATH = "data/generated_files/samples_clades_S.json"

def read_spike_fasta():
    sample_names = list()
    fasta_custom = ""
    clade_sample_count = dict()
    with open(FASTA_PATH, "r") as f_fasta_file:
        for i, line in enumerate(f_fasta_file):
            #print(line)
            fasta_custom += line
            #print()
            '''if ">" in line:
                s_line = line.split(">")[1]
                s_sline = s_line.split("|")[1]
                sample_names.append(s_sline)
                #sample_df = ncov_global_df[ ncov_global_df["strain"] == s_line.strip() ]
                if len(sample_df) > 0:
                    clade_dict = sample_df["Nextstrain_clade"].to_dict()
                    clade_name = list(clade_dict.values())[0]
                    clade_name = clade_name.replace("/", "_")
                    if clade_name not in clade_sample_count:
                        clade_sample_count[clade_name] = 0
                    clade_sample_count[clade_name] += 1'''
            if i == 100:
                break
    print(fasta_custom)

    
if __name__ == "__main__":
    read_spike_fasta()
