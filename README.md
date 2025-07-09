# ArchaicSeeker3.1-mamba

This project is developed and maintained by the **[Shuhua Xu's Research Group](https://pog.fudan.edu.cn/)**, School of Life Sciences, Fudan University.

## About

`ArchaicSeeker3.1-mamba` is an algorithm for detecting archaic introgression segments (e.g., from Neanderthals and Denisovans) in modern human genomes. It is based on the Mamba (SSM-Mamba) architecture, designed for accurate and efficient analysis of large-scale genomic data.

This repository provides the core software and an example script demonstrating how to use it for parallel analysis on multi-GPU systems.

## Citation

If you use `ArchaicSeeker3.1-mamba` in your research, please cite our relevant publications. For a list of publications, please visit our group's website: [POG Fudan Publications](https://pog.fudan.edu.cn/pog-publications/).

*(Please replace this with the specific citation for the paper)*
> **Example Citation:**
> Author A, Author B, ..., Xu S. (2025). Title of the paper. *Journal*, Volume(Issue), pages.

## License

This project is licensed under the **ArchaicSeeker Academic Use License**.
-   **For Academic Users:** Free to use, modify, and distribute for non-commercial research purposes.
-   **For Commercial Users:** A separate commercial license is required.

Please see the `LICENSE` file for detailed terms.

## Installation

We provide two methods for installation. The automated script is recommended for most users.

### Prerequisites

* A Linux-based operating system.
* **Conda** or **Miniforge/Mamba** installed.
* For GPU acceleration: An **NVIDIA GPU** with the appropriate **CUDA Toolkit** and drivers installed.

---

### Method 1: Recommended Installation via Script

This method uses the provided `install.sh` script to automatically create a conda environment and handle all dependencies, including complex ones.

1.  **Run the installation script:**
    Give the script execution permissions and run it.
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    The script will create a new conda environment named `as3_mamba`.

2.  **Activate the environment:**
    Once the installation is complete, activate the new environment to use the software.
    ```bash
    conda activate as3_mamba
    ```

> **📦 Offline or Difficult Installations:**
> The `mamba-ssm` and `causal-conv1d` packages can sometimes be difficult to build from source. For convenience, you can **pre-download their `.whl` files** that match your system (Python 3.9, CUDA version) and place them in the root directory of this project. The `install.sh` script will automatically detect and install them, skipping the build-from-source process.

---

### Method 2: Installation from `environment.yml`

This method is a standard alternative for users familiar with conda.

1.  **Create the environment from the YAML file:**
    This command creates the `as3_mamba` environment and installs all listed dependencies in a single step.
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment:**
    ```bash
    conda activate as3_mamba
    ```
> **Note:** If you encounter errors with this method, they are likely related to building `mamba-ssm` or `causal-conv1d`. We recommend using **Method 1** in such cases.

## Usage

You can run `ArchaicSeeker3.1-mamba` either by executing the main Python script directly (recommended for single runs and customization) or by using the provided shell script for parallel analysis.

### Direct Execution via `main.py`

This method gives you full control over all parameters.

**Basic Command:**
```bash
python ArchaicSeeker3.1-mamba \
    -t <path/to/target.vcf.gz> \
    -r <path/to/reference.vcf.gz> \
    -m <path/to/map.txt> \
    -o <path/to/output_folder> \
    [OPTIONS]
```

**Command-Line Arguments:**

| Argument | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| **`--test-mixed`** | `-t` | **Required.** Path to the phased VCF file of the target samples. | `None` |
| **`--reference`** | `-r` | **Required.** Path to the reference panel VCF file (archaic & African). | `None` |
| **`--map`** | `-m` | **Required.** Path to the reference map file. | `None` |
| **`--out-folder`** | `-o` | **Required.** Path to the folder where results will be saved. | `None` |
| `--base-model-cp`| | Path to the base model checkpoint (`.pth`). | Defaults to `./exp/Basemodel.../best_model.pth` |
| `--smoother-model-cp`| | Path to the smoother model checkpoint (`.pth`). | Defaults to `./exp/Smoother.../best_model.pth` |
| `--stride` | | The stride of the sliding window for model inference. | `512` |
| `--merge` | | The distance threshold (bp) for merging adjacent introgressed segments. | `5000` |
| `--anc` | | Archaic parameter setting for analysis. | `0` |
| `--target-chunk-size`| | Process target samples in chunks of this size to reduce memory usage. `None` means all at once. | `None` |
| `--base-model-args`| | Path to the base model's arguments file (`.pckl`). If `None`, auto-detected. | `None` |
| `--smoother-model-args`| | Path to the smoother model's arguments file (`.pckl`). If `None`, auto-detected. | `None` |


### Parallel Analysis via Shell Script

For convenience, we provide an example script `run_analysis.sh` to parallelize the analysis of a whole genome (chromosomes 1-22) across multiple GPUs.

1.  **Configure the script**:
    Open `run_analysis.sh` and modify the variables at the top: `wk_path`, `aseek` (which should point to your `main.py` script), and `gpus`.
2.  **Run the script**:
    ```bash
    bash run_analysis.sh
    ```

## Output Format

The primary output files are `introgression_prediction.bed` and `introgression_prediction.txt`.

### 1. `introgression_prediction.bed`
This file lists the predicted archaic introgression segments.

| Column | Description |
| :--- | :--- |
| **Chr** | Chromosome |
| **Start** | Start position of the segment (0-based) |
| **End** | End position of the segment (0-based) |
| **Haplotype** | Haplotype index relative to the start of a processed chunk. |
| **Archaic** | Predicted source: `1`=Denisovan, `2`=Neanderthal, `3`=Mosaic |
| **#SNP** | Number of SNPs within the segment |
| **Score** | Mean score of all SNPs in the segment. A higher score indicates higher confidence. A score > 0.4 is recommended. |
| **#SNP_Archaic1** | Number of SNPs supporting Archaic Source 1 |
| **#SNP_Archaic2** | Number of SNPs supporting Archaic Source 2 |
| **SampleID_HapID** | A globally unique identifier combining the original Sample ID and haplotype (1 or 2). |

### 2. `introgression_prediction.txt`
This file provides SNP-level prediction results.

-   **Rows**: Haplotype indices.
-   **Columns**: Variant positions.
-   **Values**: Predicted ancestry: `0`=African (non-introgressed), `1`=Denisovan, `2`=Neanderthal.

## Contact

For questions, bug reports, or collaboration inquiries, please visit our lab website:
**[https://pog.fudan.edu.cn/](https://pog.fudan.edu.cn/)**