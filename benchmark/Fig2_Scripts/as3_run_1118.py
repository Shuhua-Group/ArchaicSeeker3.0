from multiprocessing import Pool, Semaphore
import os
import subprocess
import numpy as np
import yaml
import time
from datetime import datetime
from pathlib import Path
import signal
import sys

# === 加载配置 ===
with open("/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
np.random.seed(config["seed"])
seed_list = np.random.randint(1, 2**31, replicates)
demo_model_list = ['AncientEurasia']#config["demo_models"]
nref_list = [10]
ntgt_list = [10]

# 日志
log_file = Path("pipeline_progress.log")
log_file.write_text(f"Parallel pipeline started at {datetime.now()}\n")
debug_script = Path("debug_failed_jobs.sh")
debug_script.write_text("#!/bin/bash\n\n")

# 获取所有GPU
def get_gpu_ids():
    result = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv,noheader", shell=True)
    return [int(x) for x in result.decode().strip().split("\n")]

gpu_ids = get_gpu_ids()
gpu_locks = {gpu_id: Semaphore(1) for gpu_id in gpu_ids}

# 全局子进程记录，Ctrl+C时清理
active_processes = []

# 运行单个任务
def run_one(args):
    params, gpu_id = args
    demog, nref, ntgt, seed = params

    out_path = Path(f"{output_dir}inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
    output_bed = out_path / "AS3_Mamba_Smoother_Nov22.bed"

    if output_bed.exists():
        print(f"⏩ Skipped: {output_bed}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    ref_vcf = out_path / "sim.ref.vcf.gz"
    tgt_vcf = out_path / "sim.tgt.vcf.gz"
    map_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/reference_map.txt"
    # merge_distance = 5000

    with gpu_locks[gpu_id]:
        stdout_log = out_path / "as3.stdout.log"
        stderr_log = out_path / "as3.stderr.log"
        as3_exec = "/share/home/linhuanyu/02_Software/ArchaicSeeker3_mem_update/AS3_dev/ArchaicSeeker3.1-mamba"

        shell_cmd = f"""
        export CUDA_VISIBLE_DEVICES={gpu_id}
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        module load ArchaicSeeker/3.0

        cd {out_path}
        /usr/bin/time -v {as3_exec} -t {tgt_vcf} -r {ref_vcf} -m {map_file} -o {out_path} --merge 5000
        """

        success = False
        for attempt in range(3):
            try:
                p = subprocess.Popen(
                    ["bash", "-c", shell_cmd],
                    stdout=open(stdout_log, "w"),
                    stderr=open(stderr_log, "w"),
                    start_new_session=True,
                )
                active_processes.append(p)
                p.communicate(timeout=7200)
                active_processes.remove(p)
                if p.returncode == 0:
                    success = True
                    break
                else:
                    raise subprocess.CalledProcessError(p.returncode, shell_cmd)

            except subprocess.TimeoutExpired:
                print(f"⚠️ 超时，强制杀掉 {params}")
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                p.wait()
                time.sleep(10)

            except subprocess.CalledProcessError:
                print(f"⚠️ 程序内部出错，重试 {attempt+1}/3 for {params}")
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                p.wait()
                time.sleep(10)

        if not success:
            err_msg = f"❌ Error after retries: {params} | GPU={gpu_id} | {datetime.now()}"
            print(err_msg)
            with open(log_file, "a") as log:
                log.write(err_msg + "\n")
            with open(debug_script, "a") as dbg:
                dbg.write(f"# FAILED: {params}\ncd {out_path}\n{shell_cmd.strip()}\n\n")
            return

    # 后处理
    subprocess.run(f"bcftools query -l {tgt_vcf} > {out_path}/tgt_samples.txt", shell=True)

    hapmap_path = out_path / "hapmap.txt"
    with open(out_path / "tgt_samples.txt") as fin, open(hapmap_path, "w") as fout:
        for i, line in enumerate(fin):
            name = line.strip()
            fout.write(f"{i*2} {name}_1\n")
            fout.write(f"{i*2+1} {name}_2\n")

    intro_raw = out_path / "introgression_prediction.bed"
    intro_bed = output_bed

    subprocess.run(f"awk 'NR==FNR {{map[$1]=$2; next}} {{$4 = map[$4]; print}}' {hapmap_path} {intro_raw} > {intro_bed}", shell=True)

    if intro_raw.exists():
        intro_raw.unlink()

    msg = f"✅ Finished: {demog} | nref={nref}, ntgt={ntgt}, seed={seed} | GPU={gpu_id} | {datetime.now()}"
    print(msg)
    with open(log_file, "a") as log:
        log.write(msg + "\n")

# 所有参数组合
all_combos = [(d, nref, ntgt, seed) for d in demo_model_list for nref in nref_list for ntgt in ntgt_list for seed in seed_list]

# 按顺序分配GPU
assigned_gpus = []
for idx in range(len(all_combos)):
    gpu = gpu_ids[idx % len(gpu_ids)]
    assigned_gpus.append(gpu)

# Ctrl+C时清理
def clean_up(sig, frame):
    print("🛑 Ctrl+C detected, killing all running subprocesses...")
    for p in active_processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    sys.exit(1)

signal.signal(signal.SIGINT, clean_up)

if __name__ == "__main__":
    task_list = list(zip(all_combos, assigned_gpus))
    with Pool(processes=len(gpu_ids)) as pool:
        pool.map(run_one, task_list)

