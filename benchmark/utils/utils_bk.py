import msprime
import tskit
import pandas as pd
import numpy as np
import pybedtools


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
    introgressed_tracts = []

    for p in ts.populations():
        source_id = [p.id for p in ts.populations() if p.metadata['name']==src_name][0]
        target_id = [p.id for p in ts.populations() if p.metadata['name']==tgt_name][0]

    for m in ts.migrations():
        if m.dest == source_id: introgressed_tracts.append((int(m.left), int(m.right)))

    introgressed_tracts.sort(key=lambda x:x[0])
    with open(output, 'w') as o:
        for t in introgressed_tracts:
            o.write(f'1\t{t[0]}\t{t[1]}\n')


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


def process_sstar_1src_output(in_file, out_file):
    """
    Description:
        Helper function for converting output from sstar to BED format files.

    Arguments:
        in_file str: Name of the input file.
        out_file str: Name of the output file.
    """
    df = pd.read_csv(in_file, sep="\t")
    df = df[df['significant'] == True]
    cols = ['chrom', 'start', 'end']
    df.to_csv(out_file, columns=cols, sep="\t", header=False, index=False)


def process_sstar_2src_output(src1_in_file, src2_in_file, threshold_file, src1_out_file, src2_out_file):
    """
    """
    src1_df = pd.read_csv(src1_in_file, sep="\t")
    src2_df = pd.read_csv(src2_in_file, sep="\t")
    threshold_df = pd.read_csv(threshold_file, sep="\t")
    threshold_df = threshold_df[threshold_df['significant'] == True]
    src1_sig_df = pd.merge(src1_df, threshold_df, on=['chrom', 'start', 'end', 'sample'])
    src2_sig_df = pd.merge(src2_df, threshold_df, on=['chrom', 'start', 'end', 'sample'])
    src1_out = open(src1_out_file, 'w')
    src2_out = open(src2_out_file, 'w')
    for i in range(len(src1_sig_df)):
        if (src1_sig_df.iloc[i]['match_pct'] > src2_sig_df.iloc[i]['match_pct']):
            src1_out.write(f'{src1_sig_df.iloc[i]["chrom"]}\t{src1_sig_df.iloc[i]["start"]}\t{src1_sig_df.iloc[i]["end"]}\n')
        if (src1_sig_df.iloc[i]['match_pct'] < src2_sig_df.iloc[i]['match_pct']):
            src2_out.write(f'{src2_sig_df.iloc[i]["chrom"]}\t{src2_sig_df.iloc[i]["start"]}\t{src2_sig_df.iloc[i]["end"]}\n')


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
    df_src1 = df[df['BestMatchedPop'] == src1_id]
    df_src2 = df[df['BestMatchedPop'] == src2_id]
    cols = ['Contig', 'Start(bp)', 'End(bp)']
    df_src1.to_csv(out_file1, columns=cols, sep="\t", header=False, index=False)
    df_src2.to_csv(out_file2, columns=cols, sep="\t", header=False, index=False)

    
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


def cal_accuracy(true_tracts, inferred_tracts):
    """
    Description:
        Helper function for calculating accuracy.

    Arguments:
        true_tracts str: Name of the BED file containing true introgresssed tracts.
        inferred_tracts str: Name of the BED file containing inferred introgressed tracts.

    Returns:
        precision float: Amount of true introgressed tracts detected divided by amount of inferred introgressed tracts.
        recall float: Amount ot true introgressed tracts detected divided by amount of true introgressed tracts.
    """
    truth_tracts = pybedtools.BedTool(true_tracts).sort().merge()
    inferred_tracts =  pybedtools.BedTool(inferred_tracts).sort().merge()

    total_inferred_tracts = sum([x.stop - x.start for x in (inferred_tracts)])
    total_true_tracts =  sum([x.stop - x.start for x in (truth_tracts)])
    true_positives = sum([x.stop - x.start for x in inferred_tracts.intersect(truth_tracts)])

    if float(total_inferred_tracts) == 0: precision = np.nan
    else: precision = true_positives / float(total_inferred_tracts) * 100
    if float(total_true_tracts) == 0: recall = np.nan
    else: recall = true_positives / float(total_true_tracts) * 100

    return precision, recall
