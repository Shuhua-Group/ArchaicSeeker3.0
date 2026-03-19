#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import subprocess
import pybedtools
import msprime
import tskit
import demes
import pandas as pd
import numpy as np

# ---------------- 参数解析 ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate chr19 under Human-Neanderthal-Denisovan-Papuan model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for msprime simulation",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory to save results",
    )
    return parser.parse_args()

############# function #############

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

# Output path

def output_path(outdir):
    ts_path = outdir / "sim.trees"
    vcf_path = outdir / "sim.vcf"
    vcf_gz_path = outdir / "sim.vcf.gz"
    biallelic_vcf_path = outdir / "sim.biallelic.vcf.gz"
    Nean_bed_path = outdir / "Nean.sim.bed"
    Den_bed_path = outdir / "Den.sim.bed"
    Den1_bed_path = outdir / "Den1.sim.bed"
    Den2_bed_path = outdir / "Den2.sim.bed"
    # Archaic_bed_path = outdir / "Archaic.sim.bed"
    target_list = outdir / "target.list"
    target_vcf_path = outdir / "target.vcf.gz"
    ref_list = outdir / "ref.list"
    ref_map = outdir / "ref.map"
    ref_vcf_path = outdir / "ref.vcf.gz"
    return ts_path, vcf_path, vcf_gz_path, biallelic_vcf_path, Nean_bed_path, Den_bed_path, Den1_bed_path, Den2_bed_path, target_list, target_vcf_path, ref_list, ref_map, ref_vcf_path

# Samples
def load_samples(ref_list, target_list, ref_map, nref= 50, ntgt = 1,nNean = 3,nDen1 = 1,nDen2 = 1):

    T_Den1 = 2058
    T_Den2 = 2058
    T_Nea1 = 2612

    samples = [
        msprime.SampleSet(nref, ploidy=2, population="YRI"),
        msprime.SampleSet(ntgt, ploidy=2, population="Papuan"),
        msprime.SampleSet(nNean, ploidy=2, population="Nea1", time=T_Nea1),
        msprime.SampleSet(nDen1, ploidy=2, population="Den1", time=T_Den1),
        msprime.SampleSet(nDen2, ploidy=2, population="Den2", time=T_Den2),
    ]

    with open(ref_list, "w") as o:
        for i in range(nref):
            o.write(f"tsk_{i}\n")
        for i in range(nref+ntgt, nref + ntgt + nNean + nDen1 + nDen2):
            o.write(f"tsk_{i}\n")
    with open(target_list, "w") as o:
        for i in range(nref, nref + ntgt):
            o.write(f"tsk_{i}\n")
    with open(ref_map, "w") as o:
        for i in range(nref):
            o.write(f"tsk_{i}\tAFR\n")
        for i in range(nref+ntgt, nref + ntgt + nNean):
            o.write(f"tsk_{i}\tNEAN\n")
        for i in range(nref+ntgt+nNean, nref + ntgt + nNean + nDen1 + nDen2):
            o.write(f"tsk_{i}\tDEN\n")

    return samples

# ---------------- 主逻辑 ---------------- #
def main():
    args = parse_args()
    seed = args.seed
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 模拟设置

    ts_path, vcf_path, vcf_gz_path, biallelic_vcf_path, Nean_bed_path, Den_bed_path, Den1_bed_path, Den2_bed_path, target_list, target_vcf_path, ref_list, ref_map, ref_vcf_path = output_path(outdir)

    samples = load_samples(ref_list, target_list, ref_map, nref= 50, ntgt = 1,nNean = 3,nDen1 = 1,nDen2 = 1)

    graph = demes.load("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/10J19/HumanNeanderthalDenisovan_PapuansOutOfAfrica_10J19.yaml")
    demography = msprime.Demography.from_demes(graph)
    mutation_rate = 1.4e-8
    recombination_rate = 1.83848e-8
    seq_len = 58585793

    print("-------------------------------- Simulation Settings --------------------------------")
    print(f"seed: {seed}")
    print(f"samples: {samples}")
    print(f"demography: {demography}")
    print(f"mutation_rate: {mutation_rate}")
    print(f"recombination_rate: {recombination_rate}")
    print(f"seq_len: {seq_len}")

    # 2. 模拟并保存

    if not os.path.exists(ts_path):
        ts = run_simulation(
            demography=demography,
            samples=samples,
            mut_rate=mutation_rate,
            sequence_length=seq_len,
            recomb_rate=recombination_rate,
        random_seed=seed
        )
        ts.dump(ts_path)
    else:
        ts = tskit.load(ts_path)

    with open(vcf_path, "w") as o:
        ts.write_vcf(o)

    print(f"[Simulation] Done seed={seed}, ts={ts_path}, vcf={vcf_path}")
    
    # 3.VCF处理

    # 双等位基因
    cmd_bial = (
        f"bcftools view {vcf_path} -m 2 -M 2 | "
        "awk 'BEGIN{OFS=\"\\t\";ORS=\"\"}"
        "{if($0~/^#/){print $0\"\\n\"}"
        "else{print $1,$2,$3,\"A\",\"T\\t\";"
        "for(i=6;i<NF;i++){print $i\"\\t\"};"
        "print $NF\"\\n\"}}' | "
        f"bgzip -c > {biallelic_vcf_path}"
    )
    subprocess.run(cmd_bial, shell=True, check=True)


    # 压缩原始 vcf -> sim2src.vcf.gz，并删除未压缩 vcf
    cmd_vcf_gz = f"bgzip -c {vcf_path} > {vcf_gz_path}"
    subprocess.run(cmd_vcf_gz, shell=True, check=True)
    os.remove(vcf_path)

    print(f"[biallelic] Done seed={seed}, biallelic_vcf={biallelic_vcf_path}, vcf_gz={vcf_gz_path}")

    # 提取Target样本
    cmd_target = (
        f"bcftools view {biallelic_vcf_path} -S {target_list} "
        f"| bgzip -c > {target_vcf_path}"
    )
    subprocess.run(cmd_target, shell=True, check=True)

    subprocess.run(f"bcftools index -t {target_vcf_path}", shell=True)

    # 提取Ref样本
    cmd_ref = (
        f"bcftools view {biallelic_vcf_path} -S {ref_list} "
        f"| bgzip -c > {ref_vcf_path}"
    )
    subprocess.run(cmd_ref, shell=True, check=True)

    subprocess.run(f"bcftools index -t {ref_vcf_path}", shell=True)

    print(f"[VCF] Done seed={seed}, biallelic_vcf={biallelic_vcf_path}, target_vcf={target_vcf_path}, ref_vcf={ref_vcf_path}")

    # 4. 提取 introgressed tracts

    get_introgressed_tracts(ts, chr_name=1, src_name="Den1", tgt_name="Papuan", output=Den1_bed_path)
    get_introgressed_tracts(ts, chr_name=1, src_name="Den2", tgt_name="Papuan", output=Den2_bed_path)
    get_introgressed_tracts(ts, chr_name=1, src_name="Nea1", tgt_name="Papuan", output=Nean_bed_path)
    
    print(f"[introgressed tracts] Done seed={seed}, Nean_bed={Nean_bed_path}, Den1_bed={Den1_bed_path}, Den2_bed={Den2_bed_path}")

if __name__ == "__main__":
    main()
