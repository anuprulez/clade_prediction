import json
import pandas as pd

CLADES_PATH = "data/generated_files/samples_clades_S.json"
GISAID = "data/ncov_global/hcov_global.json"
SAMPLE_CLADE_MUTATION = "data/generated_files/sample_clade_mutation.csv"


# Test sample names: 
# BurkinaFaso/4307/2020
# Pakistan/Gilgit1/2020
# PapuaNewGuinea/10/2020
# PapuaNewGuinea/7/2020
# Ecuador/USFQ-556/2020
# India/MH-1-27/2020

search_key = "S"


def to_tabular(clade_info, all_sample_names):
    sample_names = list()
    clade_names = list()
    mutations = list()
    for item in clade_info:
        sample_names.append(item)
        clade_names.append(clade_info[item][0]["value"])
        mutations.append(",".join(clade_info[item][1][search_key]))
    sample_clade_mutation = pd.DataFrame(list(zip(sample_names, clade_names, mutations)), columns=["Samples", "Nextstrain clades", "Mutations"])
    #sample_clade_mutation = sample_clade_mutation.sort_values(by="Nextstrain clades")
    sample_clade_mutation.to_csv(SAMPLE_CLADE_MUTATION)
    u_samples = list(set(all_sample_names))
    sname_fasta = read_spike_fasta()


def read_spike_fasta():
    fasta_file = "data/ncov_global/spikeprot0815.fasta"
    sample_names = list()
    fasta_custom = ""
    clade_sample_count = dict()
    with open(fasta_file, "r") as f_fasta_file:
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
    #print(len(sample_names), sample_names)
    #print(len(sample_names), len(list(set(sample_names))))
    #print(sample_names)
    #return list(set(sample_names))
    
    

def read_phylogenetic_data(json_file=GISAID):
    with open(json_file, "r") as fp:
        data = json.loads(fp.read())
    tree = data["tree"]
    clade_info = dict()
    all_sample_names = list()
    recursive_branch(tree, clade_info, all_sample_names)
    to_tabular(clade_info, all_sample_names)
    with open(CLADES_PATH, "w") as fread:
        fread.write(json.dumps(clade_info))


def recursive_branch(obj, clade_info, all_sample_names):
    all_s_names = list()
    if "branch_attrs" in obj:
        branch_attr = obj["branch_attrs"]
        if "children" in obj:
            children = obj["children"]
            for item in children:
                if "children" in item:
                    recursive_branch(item, clade_info, all_sample_names)
                else:
                    # retrieve nuc from branch if available
                    branch_nuc = list()
                    if "branch_attrs" in obj:
                        if "mutations" in obj["branch_attrs"]:
                            if search_key in obj["branch_attrs"]["mutations"]:
                                branch_nuc = obj["branch_attrs"]["mutations"][search_key]
                    all_sample_names.append(item["name"])
                    #all_s_names.append(item["name"])
                    #if item["name"] == "hCoV-19/USA/MD-MDH-0380/2020":
                    if search_key in item["branch_attrs"]["mutations"]:
                        print(item["name"], item["branch_attrs"]["mutations"][search_key])
                    if search_key in item["branch_attrs"]["mutations"]:
                        sample_name = item["name"]
                        all_sample_names.append(item["name"])
                        if sample_name not in clade_info:
                            clade_info[sample_name] = list()
                        # add nuc from branch to all children if available
                        branch_nuc.extend(item["branch_attrs"]["mutations"][search_key])
                        branch_nuc = list(set(branch_nuc))
                        clade_info[sample_name].append(item["node_attrs"]["clade_membership"])
                        #clade_info[sample_name].append({search_key: item["branch_attrs"]["mutations"][search_key]})
                        clade_info[sample_name].append({search_key: branch_nuc})
    else:
        return None


def get_item(item):
    ref_pos_alt = [char for char in item]
    ref = ref_pos_alt[0]
    alt = ref_pos_alt[-1]
    pos = int("".join(ref_pos_alt[1:len(ref_pos_alt) - 1]))
    return ref, alt, pos
    
def combine_mutation(ch_ref, ch_pos, ch_alt):
    #print(ch_ref, ch_pos, ch_alt)
    #print("{}>{}>{}".format(ch_ref, ch_pos, ch_alt))
    #print()
    return "{}{}{}".format(ch_ref, ch_pos, ch_alt) #"{}>{}>{}".format(ch_ref, ch_pos, ch_alt)


def nuc_parser(lst_nuc):
    parsed_nuc = list()
    repeated_nuc = list()
    chained_ref = list()
    chained_alt = list()
    chained_pos = list()
    for i, item in enumerate(lst_nuc):
        c_ref, c_alt, c_pos = get_item(item)
        if c_ref != "-" and c_alt != "-":
            p_nuc = combine_mutation(c_ref, c_pos, c_alt)
            parsed_nuc.append(p_nuc)
            if len(repeated_nuc) > 0:
               parsed_nuc.append(repeated_nuc[-1])
               repeated_nuc.clear()
               chained_ref.clear()
               chained_alt.clear()
               chained_pos.clear()
        else:
            chained_ref.append(c_ref)
            chained_alt.append(c_alt)
            chained_pos.append(str(c_pos))
            p_nuc = combine_mutation("".join(chained_ref), "".join(chained_pos[0]), "".join(chained_alt))
            repeated_nuc.append(p_nuc)
            if i < len(lst_nuc) - 1:
                next_item = lst_nuc[i+1]
                n_ref, n_alt, n_pos = get_item(next_item)
                if n_pos - 1 == c_pos:
                    repeated_nuc.append(p_nuc)
                else:
                    parsed_nuc.append(repeated_nuc[-1])
                    repeated_nuc.clear()
                    chained_ref.clear()
                    chained_alt.clear()
                    chained_pos.clear()
            else:
                if len(repeated_nuc) > 0:
                    parsed_nuc.append(repeated_nuc[-1])
                else:
                    parsed_nuc.append(p_nuc)
    return list(set(parsed_nuc))


def get_nuc_clades():
    with open(CLADES_PATH, "r") as cf:
        clades = json.loads(cf.read())
    parsed_nuc_clades = dict()
    all_clades_mutations = list()
    for key in clades:
        clade_name = clades[key][0]["value"]
        clade_nuc = clades[key][1][search_key]
        if clade_name not in parsed_nuc_clades:
            parsed_nuc_clades[clade_name] = list()
        #print(clade_name, clade_nuc)
        parsed_nuc = clade_nuc #nuc_parser(clade_nuc)
        parsed_nuc_clades[clade_name].extend(parsed_nuc)
        all_clades_mutations.extend(parsed_nuc)
    u_all_clades_mutations = list(set(all_clades_mutations))
    with open("data/generated_files/parsed_S_clades.json", "w") as fread:
        fread.write(json.dumps(parsed_nuc_clades))
    with open("data/generated_files/u_all_clades_S_mutations.json", "w") as fread:
        fread.write(json.dumps(u_all_clades_mutations))
    return u_all_clades_mutations
    
    
if __name__ == "__main__":
    read_phylogenetic_data()
    get_nuc_clades()
