#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Pool, Semaphore
import os
import signal
import subprocess
import time
from statistics import mean
from datetime import datetime
from pathlib import Path
import sys

# ===========================
# 全局日志与调试脚本
# ===========================
log_file = Path("pipeline_progress.log")
# 不覆盖历史，追加写
with log_file.open("a") as f:
    f.write(f"Parallel pipeline started at {datetime.now()}\n")

debug_script = Path("debug_failed_jobs.sh")
debug_script.write_text("#!/bin/bash\n\n")
debug_script.chmod(0o755)

# ===========================
# GPU 相关
# ===========================
def get_gpu_ids():
    """获取所有可用 GPU ID 列表。"""
    result = subprocess.check_output(
        "nvidia-smi --query-gpu=index --format=csv,noheader",
        shell=True,
    )
    return [int(x) for x in result.decode().strip().split("\n")]

gpu_ids = get_gpu_ids()
gpu_locks = {gpu_id: Semaphore(1) for gpu_id in gpu_ids}

# 全局子进程记录，Ctrl+C 时清理
active_processes = []

# ===========================
# 解析 /usr/bin/time -v 输出
# ===========================
def parse_time_resource_usage(stderr_log_path):
    """
    解析 /usr/bin/time -v 输出的资源使用信息。
    返回字典，例如：
        {
            "user_time": float,
            "sys_time": float,
            "elapsed_sec": float,
            "cpu_percent": float,
            "max_rss_kb": int,
            "effective_cores": float,
            "core_hours": float,
        }
    解析失败或文件不存在时返回 None。
    """
    stderr_log_path = Path(stderr_log_path)
    if not stderr_log_path.exists():
        return None

    user_time = None
    sys_time = None
    elapsed_str = None
    cpu_percent = None
    max_rss = None

    with stderr_log_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("User time (seconds):"):
                try:
                    user_time = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("System time (seconds):"):
                try:
                    sys_time = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Elapsed (wall clock) time"):
                # Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    elapsed_str = parts[1].strip()
            elif line.startswith("Percent of CPU this job got:"):
                # Percent of CPU this job got: 97%
                try:
                    val = line.split(":", 1)[1].strip()
                    cpu_percent = float(val.strip("%").strip())
                except ValueError:
                    pass
            elif line.startswith("Maximum resident set size (kbytes):"):
                try:
                    max_rss = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    # 把 elapsed_str 转成秒
    elapsed_sec = None
    if elapsed_str is not None:
        try:
            time_parts = [float(x) for x in elapsed_str.split(":")]
            if len(time_parts) == 3:
                h, m, s = time_parts
                elapsed_sec = h * 3600 + m * 60 + s
            elif len(time_parts) == 2:
                m, s = time_parts
                elapsed_sec = m * 60 + s
        except Exception:
            elapsed_sec = None

    # 计算总 CPU 时间（秒） = 核时（秒），以及有效核数
    core_hours = None
    effective_cores = None
    if user_time is not None or sys_time is not None:
        total_cpu_sec = (user_time or 0.0) + (sys_time or 0.0)
        core_hours = total_cpu_sec / 3600.0
        if elapsed_sec and elapsed_sec > 0:
            effective_cores = total_cpu_sec / elapsed_sec

    if (
        user_time is None and sys_time is None
        and elapsed_sec is None and max_rss is None
    ):
        return None

    return {
        "user_time": user_time,
        "sys_time": sys_time,
        "elapsed_sec": elapsed_sec,
        "cpu_percent": cpu_percent,
        "max_rss_kb": max_rss,
        "effective_cores": effective_cores,  # 平均用了多少核
        "core_hours": core_hours,            # 核时（小时）
    }

# ===========================
# GPU 监控相关
# ===========================
def start_gpu_monitor(gpu_id, log_path):
    """
    启动一个后台 nvidia-smi 进程，定期采样 GPU 利用率与显存占用。
    返回该监控进程的 Popen 对象。
    """
    cmd = (
        "nvidia-smi "
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
        "memory.used,memory.total "
        f"--format=csv,noheader,nounits -i {gpu_id} -l 5"
    )
    log_f = open(log_path, "w")
    # start_new_session=True 方便后面用 killpg 一次性杀掉
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=log_f,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc

def stop_gpu_monitor(proc):
    """停止 GPU 监控进程。"""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass

def parse_gpu_usage(log_path):
    """
    解析 nvidia-smi 采样日志，返回：
        {
            "max_gpu_util": float,
            "avg_gpu_util": float,
            "max_gpu_mem": float,
            "avg_gpu_mem": float,
        }
    单位：util 为 %，mem 为 MiB。
    如果日志为空或不存在，返回 None。
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    gpu_utils = []
    gpu_mems = []

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 预期列格式：
            # timestamp, index, utilization.gpu, utilization.memory, memory.used, memory.total
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                continue

            try:
                util_gpu = float(parts[2])  # 如 "35"
                mem_used = float(parts[4])  # MiB
            except ValueError:
                continue

            gpu_utils.append(util_gpu)
            gpu_mems.append(mem_used)

    if not gpu_utils:
        return None

    return {
        "max_gpu_util": max(gpu_utils),
        "avg_gpu_util": mean(gpu_utils),
        "max_gpu_mem": max(gpu_mems),
        "avg_gpu_mem": mean(gpu_mems),
    }

# ===========================
# 单个任务运行函数
# ===========================
def run_one(args):
    params, gpu_id = args
    seed = params

    output_dir = Path(
        "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/"
    )
    out_path = output_dir / str(seed)

    out_path.mkdir(parents=True, exist_ok=True)

    ref_vcf = out_path / "ref.vcf.gz"
    tgt_vcf = out_path / "target.vcf.gz"
    map_file = out_path / "ref.map"
    merge_distance = 15000
    prefix = "AS3_Merge_15000"
    output_bed = out_path / f"{prefix}.bed"

    if output_bed.exists():
        print(f"⏩ Skipped: {output_bed}")
        return

    with gpu_locks[gpu_id]:
        stdout_log = out_path / "as3.stdout.log"
        stderr_log = out_path / "as3.stderr.log"
        gpu_log = out_path / "gpu_usage.log"

        shell_cmd = f"""
        export CUDA_VISIBLE_DEVICES={gpu_id}
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        module load ArchaicSeeker/3.0
        cd {out_path}
        /usr/bin/time -v /share/home/linhuanyu/02_Software/ArchaicSeeker3_mem_update/AS3_dev/ArchaicSeeker3.1-mamba \\
            -t {tgt_vcf} -r {ref_vcf} -m {map_file} \\
            -o {out_path} --merge {merge_distance}
        """

        success = False

        for attempt in range(3):
            gpu_monitor = None
            try:
                # 启动 GPU 监控
                gpu_monitor = start_gpu_monitor(gpu_id, gpu_log)

                # 启动 AS3
                p = subprocess.Popen(
                    ["bash", "-c", shell_cmd],
                    stdout=open(stdout_log, "w"),
                    stderr=open(stderr_log, "w"),
                    start_new_session=True,
                )
                active_processes.append(p)
                p.communicate(timeout=7200)
                active_processes.remove(p)

                # 停止 GPU 监控
                stop_gpu_monitor(gpu_monitor)

                if p.returncode == 0:
                    success = True
                    break
                else:
                    raise subprocess.CalledProcessError(p.returncode, shell_cmd)

            except subprocess.TimeoutExpired:
                print(f"⚠️ 超时，强制杀掉 seed={seed}")
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                p.wait()
                time.sleep(10)
                stop_gpu_monitor(gpu_monitor)

            except subprocess.CalledProcessError:
                print(f"⚠️ 程序内部出错，重试 {attempt+1}/3 for seed={seed}")
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                p.wait()
                time.sleep(10)
                stop_gpu_monitor(gpu_monitor)

            except Exception as e:
                print(f"⚠️ 未知异常: {e} for seed={seed}")
                stop_gpu_monitor(gpu_monitor)
                break

        if not success:
            err_msg = f"❌ Error after retries: seed={seed} | GPU={gpu_id} | {datetime.now()}"
            print(err_msg)
            with log_file.open("a") as log:
                log.write(err_msg + "\n")
            with debug_script.open("a") as dbg:
                dbg.write(f"# FAILED: seed={seed}\ncd {out_path}\n{shell_cmd.strip()}\n\n")
            return

    intro_raw = out_path / "introgression_prediction.bed"

    subprocess.run(f"mv {intro_raw} {output_bed}", shell=True)

    # =======================
    # 解析 CPU / 时间指标
    # =======================
    usage = parse_time_resource_usage(stderr_log)

    # 解析 GPU 指标
    gpu_usage = parse_gpu_usage(gpu_log)

    # 写入资源使用汇总
    usage_file = out_path / f"{prefix}_resource_usage.tsv"

    with usage_file.open("w") as uf:
        uf.write(
            "seed\tgpu_id\tfinish_time\t"
            "user_time_sec\tsys_time_sec\telapsed_sec\tcpu_percent\t"
            "effective_cores\tcore_hours\tmax_rss_kb\t"
            "max_gpu_util\tavg_gpu_util\tmax_gpu_mem\tavg_gpu_mem\n"
        )
        uf.write(
            f"{seed}\t{gpu_id}\t{datetime.now()}"
            f"\t{usage.get('user_time') if usage else ''}"
            f"\t{usage.get('sys_time') if usage else ''}"
            f"\t{usage.get('elapsed_sec') if usage else ''}"
            f"\t{usage.get('cpu_percent') if usage else ''}"
            f"\t{usage.get('effective_cores') if usage else ''}"
            f"\t{usage.get('core_hours') if usage else ''}"
            f"\t{usage.get('max_rss_kb') if usage else ''}"
            f"\t{gpu_usage.get('max_gpu_util') if gpu_usage else ''}"
            f"\t{gpu_usage.get('avg_gpu_util') if gpu_usage else ''}"
            f"\t{gpu_usage.get('max_gpu_mem') if gpu_usage else ''}"
            f"\t{gpu_usage.get('avg_gpu_mem') if gpu_usage else ''}"
            "\n"
        )


    msg = f"✅ Finished: seed={seed} | GPU={gpu_id} | {datetime.now()}"
    print(msg)
    with log_file.open("a") as log:
        log.write(msg + "\n")

# ===========================
# 读取 seed 列表 & 分配 GPU
# ===========================
seed_file = Path(
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"
)
with seed_file.open() as f:
    seed_list = [int(line.strip()) for line in f if line.strip()]

all_combos = [seed for seed in seed_list]

assigned_gpus = []
for idx in range(len(all_combos)):
    gpu = gpu_ids[idx % len(gpu_ids)]
    assigned_gpus.append(gpu)

# ===========================
# Ctrl+C 清理
# ===========================
def clean_up(sig, frame):
    print("🛑 Ctrl+C detected, killing all running subprocesses...")
    for p in active_processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    sys.exit(1)

signal.signal(signal.SIGINT, clean_up)

# ===========================
# main
# ===========================
if __name__ == "__main__":
    task_list = list(zip(all_combos, assigned_gpus))
    with Pool(processes=len(gpu_ids)) as pool:
        pool.map(run_one, task_list)
