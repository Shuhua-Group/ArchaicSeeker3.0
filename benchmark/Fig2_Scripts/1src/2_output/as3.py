import pandas as pd
import os
import subprocess
# ====== 输入文件列表 ======
list_file = "/home/linhuanyu/00_TEST/Stats/full_1src_list.txt"

# 读取每一行
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

def Filter(AS3_raw_bed,AS3_filtered_bed,min_length,min_score):
    df = pd.read_csv(AS3_raw_bed, sep="\t", header=None)
    df = df[df[6] >= min_score]
    df = df[df[2] - df[1] >= min_length * 1000]
    df.to_csv(AS3_filtered_bed, sep="\t", header=None, index=False)
    return df

def Filter_Score_Length(Archaic_infered_bed, out_dir, score_cutoff, length_cutoff, prefix):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df = df[df[6] >= score_cutoff]
    df = df[df[2] - df[1] >= length_cutoff * 1000]
    # prefix = f'PostFilter_score_{score_cutoff}_length_{length_cutoff}kb'
    prefix = f"{prefix}"

    df.to_csv(out_dir / f"Archaic.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 1].to_csv(out_dir / f"Den.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 2].to_csv(out_dir / f"Nean.{prefix}.bed", sep="\t", index=False, header=None)
    return prefix

# 遍历每一行
for line in lines:
    pop, nref, ntgt, seed = line.split()

    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"

    input_bed = os.path.join(dir_path, "introgression_prediction.raw.bk.bed")
    input_snp = os.path.join(dir_path, "introgression_prediction.raw.snps.bk.gz")
    input_bed_tmp = os.path.join(dir_path, "introgression_prediction.raw.bed")
    input_snp_tmp = os.path.join(dir_path, "introgression_prediction.raw.snps.gz")
    output_raw_bed = os.path.join(dir_path, "Merged.raw.bed")
    output_src_bed = os.path.join(dir_path, "AS3_1208.bed")

    # 检查文件是否存在
    if not (os.path.exists(input_bed)):
        print(f"[跳过] 缺文件: {dir_path}")
        subprocess.run(f"cp {input_bed_tmp} {input_bed}", shell=True)
        subprocess.run(f"cp {input_snp_tmp} {input_snp}", shell=True)
        continue

    # ====== 处理 ======
    Filter(input_bed, input_bed_tmp, min_length=5, min_score=0)
    cmd = (
        f"/home/linhuanyu/02_Software/ArchaicSeeker3_memFast_TwoStage/merge_bed_segments.py "
        f"-i {input_bed_tmp} "
        f"-o {output_raw_bed} -d 10000"
    )
    subprocess.run(cmd, shell=True, check=True)


    Filter_Score_Length(output_raw_bed, dir_path, score_cutoff=0.6, length_cutoff=10, prefix="AS3_1208")
    print(f"[完成] {output_src_bed}")

