import msprime
import tskit
import pandas as pd
import numpy as np
import pybedtools






def find_max_match_rate(df):
    """
    Function to find the maximum match_rate and corresponding src_sample for each unique combination of chrom, start, and end.

    Parameters:
    df (pandas.DataFrame): The input dataframe with columns 'chrom', 'start', 'end', 'match_rate', 'src_sample'.

    Returns:
    pandas.DataFrame: A dataframe with columns 'chrom', 'start', 'end', 'max_match_rate', 'src_sample'.
    """
    # Sort by 'chrom', 'start', 'end', and 'match_rate' in descending order
    df_sorted = df.sort_values(by=['sample','chrom', 'start', 'end', 'match_rate'], ascending=[True, True, True, True, False])

    # Drop duplicates keeping the first (which is the max match_rate after sorting)
    df_max_match_rate = df_sorted.drop_duplicates(subset=['chrom', 'start', 'end', 'sample']).copy()

    # Select only the required columns
    result_df = df_max_match_rate[['chrom', 'start', 'end', 'sample', 'match_rate', 'src_sample']]

    return result_df



def process_skovhmm_output_new(in_file, out_file, cutoff = 0.5, src_id = 'Archaic'):
    """
    Description:
        Helper function for converting output from SkovHMM to BED format given a cutoff.

    Arguments:
        in_file str: Name of the input file.
        out_file str: Name of the output file.
        cutoff float: Cutoff of posterior probablity for assigning an introgressed fragments.
        win_len int: Window length for detecting introgressed framgents.
        src_id str: Name of the population donated introgressed fragments.
    """
    df = pd.read_csv(in_file, sep="\t")
    df = df[df["state"] == src_id]
    df["mean_prob"] = pd.to_numeric(df["mean_prob"], errors="coerce")
    df = df[df["mean_prob"] > cutoff]
    cols = ['chrom', 'start', 'end']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)

def process_skovhmm_output_2src_new(in_file, out_file, out_file1, out_file2, cutoff=0.5):
    import pandas as pd

    df = pd.read_csv(in_file, sep="\t", header=0)

    # 保留 state == 'Archaic' 且 mean_prob > cutoff
    df = df[df["state"] == 'Archaic'].copy()
    df["mean_prob"] = pd.to_numeric(df["mean_prob"], errors="coerce")
    df = df[df["mean_prob"] > cutoff]

    # 选出 tsk_* 列
    tsk_cols = [col for col in df.columns if col.startswith('tsk')]

    last_tsk_col = tsk_cols[-1] if tsk_cols else None  # 最后一列 tsk_ 名称

    def assign_state(row):
        values = row[tsk_cols]
        max_val = values.max()
        max_cols = values[values == max_val].index.tolist()  # 取所有最大值的列名
        if len(max_cols) != 1:
            return None  # 多个最大值 → 无法确定
        if max_cols[0] == last_tsk_col:
            return 'src2'
        else:
            return 'src1'

    df['assigned_state'] = df.apply(assign_state, axis=1)

    # 分组
    df1 = df[df['assigned_state'] == 'src1']
    df2 = df[df['assigned_state'] == 'src2']

    # 输出
    cols = ['chrom', 'start', 'end']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)   # 所有 archaic (未分类)
    df1.to_csv(out_file1, columns=cols, sep="\t", header=False, index=False) # src1
    df2.to_csv(out_file2, columns=cols, sep="\t", header=False, index=False) # src2


# def process_skovhmm_output_2src(in_file, src1_out_file, src2_out_file, cutoff, src_id, src1_name_file, src2_name_file):
#     # Read input file
#     df = pd.read_csv(in_file, sep="\t")
#     df = df[df["state"] == src_id]
#     df = df[df["mean_prob"] > cutoff]

#     df["end"] = df["end"] + 1000

#     if len(df) == 0:
#         open(src1_out_file, 'w').close()
#         open(src2_out_file, 'w').close()
#         return

#     # Read source names
#     with open(src1_name_file) as f:
#         src1_names = f.read().splitlines()
#     with open(src2_name_file) as f:
#         src2_names = f.read().splitlines()
    
#     # Filter columns for source names
#     src_columns = src1_names + src2_names
#     df_src = df[src_columns]

#     # Find max value and corresponding column name for each row
#     max_value_col = df_src.idxmax(axis=1)

#     # Split data into two DataFrames based on source names
#     df1 = df[max_value_col.isin(src1_names)]
#     df2 = df[max_value_col.isin(src2_names)]

#     # Columns to output
#     cols = ['chrom', 'start', 'end']

#     # Write to output files
#     df1.to_csv(src1_out_file, columns=cols, sep="\t", header=False, index=False)
#     df2.to_csv(src2_out_file, columns=cols, sep="\t", header=False, index=False)


def combine_segs(segs, get_segs=True):
    merged = np.empty([0, 2])
    if len(segs) == 0:
        if get_segs:
            return ([])
        else:
            return (0)
    sorted_segs = segs[np.argsort(segs[:, 0]), :]
    for higher in sorted_segs:
        if len(merged) == 0:
            merged = np.vstack([merged, higher])
        else:
            lower = merged[-1, :]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1, :] = (lower[0], upper_bound)
            else:
                merged = np.vstack([merged, higher])
    if get_segs:
        return (merged)
    else:
        return (np.sum(merged[:, 1] - merged[:, 0]) / len(merged))

def run_simulation(demography=None, samples=None, sequence_length=20*10**6, mut_rate=1.29e-8, recomb_rate=1e-8, random_seed=None):
    """
    Description:
        Simulates tree-sequences using msprime.

    Arguments:
        demography msprime.Demography: Demographic model for simulation.
        samples list: List of sample sets.
        sequence_length int: Length of the simulated sequence.
        mut_rate float: Mutation rate per base per generation.
        recomb_rate float: Recombinate rate per base per generation.
        random_seed int: Random seed.

    Returns:
        ts tskit.TreeSequence: Simulated tree-squeuences.
    """
    if (demography is None) or (samples is None):
        print("No simulation is performed, because either the demographic model or the sample set is not available.")
        return None

    ts = msprime.sim_ancestry(
        recombination_rate=recomb_rate,
        sequence_length=sequence_length,
        samples = samples,
        demography = demography,
        record_migrations=True,  # Needed for tracking segments.
        random_seed=random_seed,
    )

    ts = msprime.sim_mutations(ts, rate=mut_rate, random_seed=random_seed)

    return ts


def get_introgressed_tracts(ts, chr_name, src_name, tgt_name, output):
    """
    Description:
        Outputs true introgressed tracts from a tree-sequence into a BED file.

    Arguments:
        ts tskit.TreeSequence: Tree-sequence containing introgressed tracts.
        chr_name int: Name of the chromosome.
        src_name str: Name of the source population.
        tgt_name str: Name of the target population.
        output string: Name of the output file.
    """
    source_id = [p.id for p in ts.populations() if p.metadata['name'] == src_name][0]
    target_id = [p.id for p in ts.populations() if p.metadata['name'] == tgt_name][0]

    de_seg = {i: [] for i in ts.get_samples(target_id)}

    for mr in ts.migrations():
        if mr.dest == source_id:
            for tree in ts.trees(leaf_lists=True):
                if mr.left > tree.get_interval()[0]:
                    continue
                if mr.right <= tree.get_interval()[0]:
                    break
                for l in tree.leaves(mr.node):
                    if l in de_seg.keys():
                        de_seg[l].append(tree.get_interval())

    true_de_segs = [combine_segs(np.array(de_seg[i]), True) for i in sorted(de_seg.keys())]
    with open(output, 'w') as o:
        for haplotype, archaic_segments in enumerate(true_de_segs):
            for archaic_segment in archaic_segments:
                o.write(
                    '{}\t{}\t{}\t{}\n'.format(chr_name, int(archaic_segment[0]), int(archaic_segment[1]), haplotype))

def get_introgressed_tracts_as2(ts, chr_name, src_name, tgt_name, output):
    """
    Description:
        Outputs true introgressed tracts from a tree-sequence into a BED file.

    Arguments:
        ts tskit.TreeSequence: Tree-sequence containing introgressed tracts.
        chr_name int: Name of the chromosome.
        src_name str: Name of the source population.
        tgt_name str: Name of the target population.
        output string: Name of the output file.
    """
    source_id = [p.id for p in ts.populations() if p.metadata['name'] == src_name][0]
    target_id = [p.id for p in ts.populations() if p.metadata['name'] == tgt_name][0]

    de_seg = {i: [] for i in ts.get_samples(target_id)}

    for mr in ts.migrations():
        if mr.dest == source_id and mr.source == target_id:
            for tree in ts.trees(leaf_lists=True):
                if mr.left > tree.get_interval()[0]:
                    continue
                if mr.right <= tree.get_interval()[0]:
                    break
                for l in tree.leaves(mr.node):
                    if l in de_seg.keys():
                        de_seg[l].append(tree.get_interval())

    true_de_segs = [combine_segs(np.array(de_seg[i]), True) for i in sorted(de_seg.keys())]
    with open(output, 'w') as o:
        for haplotype, archaic_segments in enumerate(true_de_segs):
            for archaic_segment in archaic_segments:
                o.write(
                    '{}\t{}\t{}\t{}\n'.format(chr_name, int(archaic_segment[0]), int(archaic_segment[1]), haplotype))

def process_sprime_output(in_file, out_file):
    """
    Description:
        Helper function for converting output from SPrime to BED format.

    Arguments:
        in_file str: Name of the input file.
        out_file str: Name of the output file.
    """
    # read in the data - the SPrime output
    df = pd.read_csv(in_file, delimiter="\t")

    # drop columns that are not needed
    df2 = df.drop(['ID', 'REF', 'ALT', 'ALLELE'], axis=1)

    # add columns START and END with the highest ans lowest position of each chromosome, segment and score
    df2['START'] = df2.groupby(['CHROM', 'SCORE', 'SEGMENT'])['POS'].transform(min)
    df2['END'] = df2.groupby(['CHROM', 'SCORE', 'SEGMENT'])['POS'].transform(max)

    # group by chromosome, segment and score - drop the column position
    df3 = df2.loc[df2.groupby(["CHROM", "SCORE", "SEGMENT"])["POS"].idxmax()]
    df4 = df3.drop(['POS'], axis=1)

    # get the right order (for the bed file)
    df_final = df4[['CHROM','START','END','SEGMENT','SCORE']].sort_values(by=['START', 'SEGMENT'])

    np.savetxt(out_file, df_final.values, fmt='%s', delimiter='\t')


def process_sprime_match_rate(in_file, src1_out_file, src2_out_file):
    """
    """
    df = pd.read_csv(in_file, sep=" ").dropna()
    src1_out = open(src1_out_file, 'w')
    src2_out = open(src2_out_file, 'w')
    for i in range(len(df)):
        if (df.iloc[i]['src1'] > df.iloc[i]['src2']): src1_out.write(f'{int(df.iloc[i]["chr"])}\t{int(df.iloc[i]["from"])}\t{int(df.iloc[i]["to"])}\n')
        if (df.iloc[i]['src1'] < df.iloc[i]['src2']): src2_out.write(f'{int(df.iloc[i]["chr"])}\t{int(df.iloc[i]["from"])}\t{int(df.iloc[i]["to"])}\n')


def process_skovhmm_output(in_file, out_file, cutoff, win_len, src_id):
    """
    Description:
        Helper function for converting output from SkovHMM to BED format given a cutoff.

    Arguments:
        in_file str: Name of the input file.
        out_file str: Name of the output file.
        cutoff float: Cutoff of posterior probablity for assigning an introgressed fragments.
        win_len int: Window length for detecting introgressed framgents.
        src_id str: Name of the population donated introgressed fragments.
    """
    df = pd.read_csv(in_file, sep="\t")
    df = df[df[src_id] > cutoff]
    df['end'] = df['start'] + win_len
    cols = ['chrom', 'start', 'end']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)


def process_archaicseeker2_output(in_file, out_file1, out_file2, src1_id, src2_id):
    """
    Description:
        Helper function for converting output from ArchaicSeeker 2.0 to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file1 str: Name of the output file for src1.
        out_file2 str: Name of the output file for src2.
        src1_id str: Name of the source population 1.
        src2_id str: Name of the source population 2.
    """
    df = pd.read_csv(in_file, sep="\t")
    df_src1 = df[df['BestMatchedPop'].str.contains(src1_id, na=False)]
    df_src2 = df[df['BestMatchedPop'].str.contains(src2_id, na=False)]
    cols = ['Contig', 'Start(bp)', 'End(bp)']
    df_src1.to_csv(out_file1, columns=cols, sep="\t", header=False, index=False)
    df_src2.to_csv(out_file2, columns=cols, sep="\t", header=False, index=False)


def process_archaicseeker2_1src_output(in_file, out_file):
    """
    Description:
        Helper function for converting output from ArchaicSeeker 2.0 to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file1 str: Name of the output file for src1.
        out_file2 str: Name of the output file for src2.
        src1_id str: Name of the source population 1.
        src2_id str: Name of the source population 2.
    """
    df = pd.read_csv(in_file, sep="\t")
    cols = ['Contig', 'Start(bp)', 'End(bp)']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)


def process_archaicseeker3_output(in_file, out_file1, out_file2,cutoff):
    """
    Description:
        Helper function for converting output from ArchaicSeeker 3.0 to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file1 str: Name of the output file for src1.
        out_file2 str: Name of the output file for src2.
        src1_id str: Name of the source population 1.
        src2_id str: Name of the source population 2.
    """
    df = pd.read_csv(in_file, sep=r"\s+", engine="python", header=None)
    df.columns = ['Chr', 'Start', 'End', 'Index', 'Archaic', '#SNP', 'Score']
    df = df[df['Score'] > cutoff]
    df_src1 = df[df['Archaic'] == 1]
    df_src2 = df[df['Archaic'] == 2]
    cols = ['Chr', 'Start', 'End']
    df_src1.to_csv(out_file1, columns=cols, sep="\t", header=False, index=False)
    df_src2.to_csv(out_file2, columns=cols, sep="\t", header=False, index=False)

def process_archaicseeker3_1src_output(in_file, out_file,cutoff):
    """
    Description:
        Helper function for converting output from ArchaicSeeker 3.0 to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file1 str: Name of the output file for src1.
        out_file2 str: Name of the output file for src2.
        src1_id str: Name of the source population 1.
        src2_id str: Name of the source population 2.
    """
    df = pd.read_csv(in_file, sep=r"\s+", engine="python", header=None)
    df.columns = ['Chr', 'Start', 'End', 'Index', 'Archaic', '#SNP', 'Score']
    df = df[df['Score'] > cutoff]
    cols = ['Chr', 'Start', 'End']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)


def make_skovhmm_input(out_chrfile, out_mutfile):
    """
    Description:
        Function to create two files - the chromosome textfile and the mutation rates textfile.

    Arguments:
        out_chrfile str: Name of the chromosome output file.
        out_mutfile str: Name of the mutrates output file.
    """
    #create list with values from 0 to 200000000 bp
    #and convert to dataframe
    a = list(range(0, 200001000, 1000))
    df = pd.DataFrame(a)
    df.rename( columns={0 :'bp'}, inplace=True )

    #add column chr with values 1
    df["chr"] = 1

    #add column perc with values 1.0
    df["perc"] = 1.0

    #reorder columns
    df = df[['chr', 'bp', 'perc']]

    #save files
    df.to_csv(out_chrfile, sep='\t', index=False, header=False)
    df.to_csv(out_mutfile, sep='\t', index=False, header=False)


def ms2vcf(ms_file, vcf_file, nsamp, seq_len, ploidy=2):
    """
    Description:
        Converts ms output files into the VCF format.

    Arguments:
        ms_file str: Name of the ms file (input).
        vcf_file str: Name of the VCF file (output).
        nsamp int: Number of haploid genomes.
        seq_len int: Sequence length.
        ploidy int: Ploidy of each individual.
    """
    data = []
    i = -1
    header = "##fileformat=VCFv4.2\n"
    header += "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(['ms_' + str(i) for i in range(int(nsamp/ploidy))])

    with open(ms_file, 'r') as f:
        f.readline()
        f.readline()
        for l in f.readlines():
            if l.startswith('//'):
                i += 1
                data.append({})
                data[i]['pos'] = []
                data[i]['geno'] = []
            elif l.startswith('positions'):
                data[i]['pos'] = l.rstrip().split(" ")[1:]
            elif l.startswith('0') or l.startswith('1'):
                data[i]['geno'].append(l.rstrip())

    shift = 0
    with open(vcf_file, 'w') as o:
        o.write(header+"\n")
        for i in range(len(data)):
            for j in range(len(data[i]['pos'])):
                pos = int(seq_len * float(data[i]['pos'][j])) + shift
                genotypes = "".join([data[i]['geno'][k][j] for k in range(len(data[i]['geno']))])
                genotypes = "\t".join([a+'|'+b for a,b in zip(genotypes[0::ploidy],genotypes[1::ploidy])])
                o.write(f"1\t{pos}\t.\tA\tT\t100\tPASS\t.\tGT\t{genotypes}\n")
            shift += seq_len


# def cal_accuracy(true_tracts, inferred_tracts):
#     """
#     Description:
#         Helper function for calculating accuracy.

#     Arguments:
#         true_tracts str: Name of the BED file containing true introgresssed tracts.
#         inferred_tracts str: Name of the BED file containing inferred introgressed tracts.

#     Returns:
#         precision float: Amount of true introgressed tracts detected divided by amount of inferred introgressed tracts.
#         recall float: Amount ot true introgressed tracts detected divided by amount of true introgressed tracts.
#     """
#     truth_tracts = pybedtools.BedTool(true_tracts).sort().merge()
#     inferred_tracts =  pybedtools.BedTool(inferred_tracts).sort().merge()

#     total_inferred_tracts = sum([x.stop - x.start for x in (inferred_tracts)])
#     total_true_tracts =  sum([x.stop - x.start for x in (truth_tracts)])
#     true_positives = sum([x.stop - x.start for x in inferred_tracts.intersect(truth_tracts)])

#     if float(total_inferred_tracts) == 0: precision = np.nan
#     else: precision = true_positives / float(total_inferred_tracts) * 100
#     if float(total_true_tracts) == 0: recall = np.nan
#     else: recall = true_positives / float(total_true_tracts) * 100
#     return precision, recall

# def cal_accuracy_tgt1(true_tracts, inferred_tracts):
#     """
#     Description:
#         Helper function for calculating accuracy.

#     Arguments:
#         true_tracts str: Name of the BED file containing true introgresssed tracts.
#         inferred_tracts str: Name of the BED file containing inferred introgressed tracts.

#     Returns:
#         precision float: Amount of true introgressed tracts detected divided by amount of inferred introgressed tracts.
#         recall float: Amount ot true introgressed tracts detected divided by amount of true introgressed tracts.
#     """
#     truth_tracts = pybedtools.BedTool(true_tracts).sort().merge()
#     inferred_tracts =  pybedtools.BedTool(inferred_tracts).sort().merge()

#     total_inferred_tracts = sum([x.stop - x.start for x in (inferred_tracts)])
#     total_true_tracts =  sum([x.stop - x.start for x in (truth_tracts)])
#     true_positives = sum([x.stop - x.start for x in inferred_tracts.intersect(truth_tracts)])

#     if float(total_inferred_tracts) == 0: precision = np.nan
#     else: precision = true_positives / float(total_inferred_tracts) * 100
#     if float(total_true_tracts) == 0: recall = np.nan
#     else: recall = true_positives / float(total_true_tracts) * 100
    
#     if precision + recall != 0:
#         f1 = 2 * precision * recall / (precision + recall)
#     else: f1 = 0
#     return precision, recall, f1

# def cal_accuracy_tgt10(true_tracts, inferred_tracts):
    # """
    # Description:
    #     Helper function for calculating accuracy.

    # Arguments:
    #     true_tracts str: Name of the BED file containing true introgresssed tracts.
    #     inferred_tracts str: Name of the BED file containing inferred introgressed tracts.

    # Returns:
    #     precision float: Amount of true introgressed tracts detected divided by amount of inferred introgressed tracts.
    #     recall float: Amount ot true introgressed tracts detected divided by amount of true introgressed tracts.
    # """
    # truth_tracts = pybedtools.BedTool(true_tracts).sort().merge()
    # inferred_tracts =  pybedtools.BedTool(inferred_tracts).sort().merge()

    # total_inferred_tracts = sum([x.stop - x.start for x in (inferred_tracts)])
    # total_true_tracts =  sum([x.stop - x.start for x in (truth_tracts)])
    # true_positives = sum([x.stop - x.start for x in inferred_tracts.intersect(truth_tracts)])

    # if float(total_inferred_tracts) == 0: precision = np.nan
    # else: precision = true_positives / float(total_inferred_tracts) * 100
    # if float(total_true_tracts) == 0: recall = np.nan
    # else: recall = true_positives / float(total_true_tracts) * 100
    
    # if precision + recall != 0:
    #     f1 = 2 * precision * recall / (precision + recall)
    # else: f1 = 0
    # return precision, recall, f1

def process_ibdmix_output(in_file, out_file, cutoff):
    """
    Description:
        Helper function for converting output from IBDmix to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file1 str: Name of the output file for src1.
        out_file2 str: Name of the output file for src2.
        src1_id str: Name of the source population 1.
        src2_id str: Name of the source population 2.
    """
    df = pd.read_csv(in_file, sep="\t", header=0)
    df = df[df['slod'] > cutoff]
    cols = ['chrom', 'start', 'end']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)


def _calc_accuracy(truth_bt, infer_bt):
    truth_bt  = truth_bt.sort().merge()
    infer_bt  = infer_bt.sort().merge()

    total_inf  = sum(iv.length for iv in infer_bt)
    total_true = sum(iv.length for iv in truth_bt)
    tp_len     = sum(iv.length for iv in infer_bt.intersect(truth_bt))

    precision = np.nan if total_inf  == 0 else tp_len / total_inf  * 100
    recall    = np.nan if total_true == 0 else tp_len / total_true * 100
    f1 = 0 if (precision + recall) == 0 else 2*precision*recall/(precision+recall)
    return precision, recall, f1


# ---------- ② 旧版：所有样本片段合在一起 ----------
def cal_accuracy_tgt1(true_tracts_path: str, inferred_tracts_path: str):
    """
    与原函数一致：将所有样本片段合并后整体评估。
    """
    truth_bt   = pybedtools.BedTool(true_tracts_path)
    infer_bt   = pybedtools.BedTool(inferred_tracts_path)
    return _calc_accuracy(truth_bt, infer_bt)


# ---------- ③ 新版：按样本分别评估再取平均 ----------
def cal_accuracy_tgt10(true_tracts_path: str,
                       inferred_tracts_path: str,
                       sample_col: int = 3):
    """
    - true_tracts_path / inferred_tracts_path:  含列  chr start end sampleID ...
    - sample_col:  样本 ID 在第几列（0-based），默认 3 即第 4 列
    返回:
        mean_precision, mean_recall, mean_f1,  per_sample_dict
    """
    truth_all  = pybedtools.BedTool(true_tracts_path)
    infer_all  = pybedtools.BedTool(inferred_tracts_path)

    # 获取同时存在于真值和预测中的样本集合
    samples_truth = {iv.fields[sample_col] for iv in truth_all}
    samples_pred  = {iv.fields[sample_col] for iv in infer_all}
    samples = sorted(samples_truth & samples_pred)

    results = {}
    for samp in samples:
        t_bt = truth_all.filter(lambda x: x.fields[sample_col] == samp).saveas()
        i_bt = infer_all.filter(lambda x: x.fields[sample_col] == samp).saveas()
        results[samp] = _calc_accuracy(t_bt, i_bt)

    # 计算宏平均（忽略 NaN）
    arr = np.array(list(results.values()), dtype=float)  # shape: (n_sample, 3)
    mean_prec, mean_rec, mean_f1 = np.nanmean(arr, axis=0)

    return mean_prec, mean_rec, mean_f1, results
