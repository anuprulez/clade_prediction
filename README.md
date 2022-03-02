# SARS-COV2 sequence generation
SARS-COV2 sequence generation by applying deep learning techniques on amino acid sequences (sequence to sequence encoder-decoder, GAN)

# Spike protein reference
### NCBI Reference Sequence: YP_009724390.1
https://www.ncbi.nlm.nih.gov/protein/YP_009724390.1

# Resources
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8056063/
https://www.mdpi.com/1420-3049/26/9/2622/pdf
https://www.nature.com/articles/s42003-021-01754-6
https://www.nature.com/articles/s41401-020-0485-4
https://www.nature.com/articles/s41586-020-2895-3
https://www.sciencedirect.com/science/article/pii/S2405844021006757
https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
https://arxiv.org/pdf/2008.11790.pdf
https://www.nature.com/articles/s41579-021-00573-0#Sec3
https://pubmed.ncbi.nlm.nih.gov/33423311/

https://www.mdpi.com/2076-2607/9/5/1035/pdf
https://virological.org/t/recent-evolution-and-international-transmission-of-sars-cov-2-clade-19b-pango-a-lineages/711
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7709189/
https://www.ecdc.europa.eu/sites/default/files/documents/SARS-CoV-2-variant-multiple-spike-protein-mutations-United-Kingdom.pdf
https://www.cell.com/cell/pdf/S0092-8674(20)30820-5.pdf
https://covariants.org/variants
https://www.nature.com/articles/s41598-021-96950-z
https://www.nature.com/articles/s41467-020-19843-1
https://journals.asm.org/doi/10.1128/mBio.01188-21
http://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html
### Structured noise injection
https://openaccess.thecvf.com/content_CVPR_2020/papers/Alharbi_Disentangled_Image_Generation_Through_Structured_Noise_Injection_CVPR_2020_paper.pdf
### F5L, F12S, D614G
https://www.frontiersin.org/articles/10.3389/fimmu.2021.725240/full

### Unrolled GAN
https://github.com/MarisaKirisame/unroll_gan/blob/master/main.py
https://github.com/andrewliao11/unrolled-gans/blob/master/unrolled_gan.ipynb

1. conda create --name clade_pred python=3.9
2. conda activate clade_pred
3. pip install tensorflow-gpu pandas matplotlib bio h5py scikit-learn nltk python-Levenshtein

### Cross batch statefullness

https://www.tensorflow.org/guide/keras/rnn

### Teacher forcing

https://aclanthology.org/2020.lrec-1.576.pdf
https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in21_paper_12.pdf
https://en.wikipedia.org/wiki/Gibbs_sampling
http://people.csail.mit.edu/andyyuan/docs/interspeech-16.audio2vec.paper.pdf
https://arxiv.org/pdf/2108.11992.pdf

### Acknowledgements

Data from this analysis have been downloaded from GISAID (https://www.gisaid.org/help/publish-with-data-from-gisaid/)

- Khare, S., et al (2021) GISAID’s Role in Pandemic Response. China CDC Weekly, 3(49): 1049-1051. doi: 10.46234/ccdcw2021.255  PMCID: 8668406
- Elbe, S. and Buckland-Merrett, G. (2017) Data, disease and diplomacy: GISAID’s innovative contribution to global health. Global Challenges, 1:33-46. doi:10.1002/gch2.1018  PMCID: 31565258
- Shu, Y. and McCauley, J. (2017)  GISAID: from vision to reality. EuroSurveillance, 22(13) doi:10.2807/1560-7917.ES.2017.22.13.30494  PMCID: PMC5388101
