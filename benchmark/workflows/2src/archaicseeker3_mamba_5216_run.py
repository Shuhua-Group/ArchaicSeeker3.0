from multiprocessing import Pool, Semaphore, Manager
import os
import subprocess
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

# === 加载配置 ===
with open("config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
np.random.seed(config["seed"])
seed_list = np.random.randint(1, 2**31, replicates)
demo_model_list = config["demo_models"]
nref_list = [10, 50]
ntgt_list = [1, 10]

log_file = Path("pipeline_progress.log")
log_file.write_text(f"Parallel pipeline started at {datetime.now()}\n")
debug_script = Path("debug_failed_jobs.sh")
debug_script.write_text("#!/bin/bash\n\n")

# 获取所有 GPU ID 并为每张卡创建并发锁（限制每卡最多两个任务）
def get_gpu_ids():
    result = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv,noheader", shell=True)
    return [int(x) for x in result.decode().strip().split("\n")]

gpu_ids = get_gpu_ids()
gpu_locks = {gpu_id: Semaphore(3) for gpu_id in gpu_ids}

def run_one(params):
    demog, nref, ntgt, seed = params
    out_path = Path(f"{output_dir}inference/ArchaicSeeker3.1/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
    output_bed = out_path / "AS3_Mamba_Smoother_4096_2048_5216.bed"

    if output_bed.exists():
        print(f"⏩ Skipped: {output_bed}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    ref_vcf = out_path / "sim.ref.vcf.gz"
    tgt_vcf = out_path / "sim.tgt.vcf.gz"
    map_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/reference_map.txt"
    merge_distance = 5216

    # 获取空闲 GPU，并申请并发锁
    gpu_cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -nk2 | head -n1 | cut -d, -f1"
    gpu_id = int(subprocess.check_output(gpu_cmd, shell=True).decode().strip())

    with gpu_locks[gpu_id]:
        stdout_log = out_path / "as3.stdout.log"
        stderr_log = out_path / "as3.stderr.log"
        as3_exec = "/share/apps/gene/ArchaicSeeker-mamba-smoother/ArchaicSeeker3-mamba-smoother"

        shell_cmd = f"""
        export CUDA_VISIBLE_DEVICES={gpu_id}
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        module load anaconda3/2022.10
        source activate ArchaicSeeker3
        cd {out_path}
        /usr/bin/time -v {as3_exec} \\
            -t {tgt_vcf} -r {ref_vcf} -m {map_file} -o {out_path} \\
            --merge {merge_distance} \\
            1> {stdout_log} 2> {stderr_log}
        """

        success = False
        for attempt in range(3):
            try:
                subprocess.run(["bash", "-c", shell_cmd], check=True)
                success = True
                break
            except subprocess.CalledProcessError:
                print(f"Retry {attempt+1}/3 for task {params}")
                time.sleep(30)

        if not success:
            err_msg = f"❌ Error for: {params} | GPU={gpu_id} | {datetime.now()}"
            print(err_msg)
            with open(log_file, "a") as log:
                log.write(err_msg + "\n")
            with open(debug_script, "a") as dbg:
                dbg.write(f"# FAILED: {params}\ncd {out_path}\n{shell_cmd.strip()}\n\n")
            return

    subprocess.run(f"bcftools query -l {tgt_vcf} > {out_path}/tgt_samples.txt", shell=True)

    hapmap_path = out_path / "hapmap.txt"
    with open(out_path / "tgt_samples.txt") as fin, open(hapmap_path, "w") as fout:
        for i, line in enumerate(fin):
            name = line.strip()
            fout.write(f"{i*2} {name}_1\n")
            fout.write(f"{i*2+1} {name}_2\n")

    def process_bed(raw_file, out_file):
        subprocess.run(f"awk 'NR==FNR {{map[$1]=$2; next}} {{$4 = map[$4]; print}}' {hapmap_path} {raw_file} > {out_file}", shell=True)

    intro_raw = out_path / "introgression_prediction.bed"
    intro_bed = output_bed

    process_bed(intro_raw, intro_bed)

    msg = f"✅ Finished: {demog} | nref={nref}, ntgt={ntgt}, seed={seed} | GPU={gpu_id} | {datetime.now()}"
    print(msg)
    with open(log_file, "a") as log:
        log.write(msg + "\n")

# 所有参数组合
all_combos = [(d, nref, ntgt, seed) for d in demo_model_list for nref in nref_list for ntgt in ntgt_list for seed in seed_list]

if __name__ == "__main__":
    with Pool(processes=20) as pool:
        pool.map(run_one, all_combos)
