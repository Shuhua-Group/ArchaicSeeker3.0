# ArchaicSeeker3.1-mamba

This project is developed and maintained by the **[Shuhua Xu's Research Group](https://pog.fudan.edu.cn/)**, School of Life Sciences, Fudan University.

## About

`ArchaicSeeker3.1-mamba` is an algorithm for detecting archaic introgression segments (e.g., from Neanderthals and Denisovans) in modern human genomes. It is based on the Mamba (SSM-Mamba) architecture, designed for accurate and efficient analysis of large-scale genomic data.

This repository provides the core software and example scripts to demonstrate how to use it for parallel analysis on multi-GPU systems.

---

## Citation

If you use `ArchaicSeeker3.1-mamba` in your research, please cite our relevant publications. For a list of publications, please visit our group's website: [POG Fudan Publications](https://pog.fudan.edu.cn/#/article).

> **Citation:**
> unpublished

---

## License

This project is licensed under the **ArchaicSeeker Academic Use License**.
-   **For Academic Users:** Free to use, modify, and distribute for non-commercial research purposes.
-   **For Commercial Users:** A separate commercial license is required.

Please see the `LICENSE` file for detailed terms.

---

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
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    The script will create a new conda environment named `as3_mamba`.

2.  **Activate the environment:**
    ```bash
    conda activate as3_mamba
    ```

> **📦 Offline or Difficult Installations:**
> The `mamba-ssm` and `causal-conv1d` packages can be difficult to build from source. For convenience, you can **pre-download their `.whl` files** that match your system (Python 3.9, CUDA version) and place them in the root directory of this project. The `install.sh` script will automatically detect and install them.

---

### Method 2: Installation from `environment.yml`

This is an alternative for users familiar with conda.

1.  **Create the environment from the YAML file:**
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment:**
    ```bash
    conda activate as3_mamba
    ```
> **Note:** If you encounter errors with this method, they are likely related to building `mamba-ssm` or `causal-conv1d`. We recommend using **Method 1** in such cases.

---

## Data Preparation

For your convenience, we provide pre-processed archaic reference data panels.

* Download the data for either **GRCh38** or **CHM13** from our Zenodo record:
    **[https://zenodo.org/records/14552025](https://zenodo.org/records/14552025)**

---

## Usage Workflow

The analysis is divided into two main steps, managed by two separate scripts.

### Step 1: Pre-processing Raw VCFs (Optional)

This step is only necessary if you want to process your own raw VCF files instead of using our pre-processed data.

1.  **Configure the script:** Open `00.preprocess.sh` and set the paths to your input data and desired output directories.
2.  **Run the script:**
    ```bash
    bash 00.preprocess.sh
    ```
This script will normalize and filter your VCF files, generating the `Final_Target_VCFs`, `Final_Ref_VCFs`, and `reference.map` files required for the next step.

> **Note:** If you downloaded the pre-processed data from Zenodo, you can **skip this step**.

### Step 2: Running ArchaicSeeker3 Analysis

This script runs the main ArchaicSeeker3 analysis in parallel across all chromosomes.

1.  **Configure the script:** Open `01.run_archaicseeker3.mamba.all.chr.sh` and set the paths to your input data (from Step 1 or Zenodo), the desired output directory, and your GPU configuration.
2.  **Run the script:**
    ```bash
    bash 01.run_archaicseeker3.mamba.all.chr.sh
    ```

### Direct Execution (Advanced)

For debugging or running on a single file, you can execute the main program directly.

**Basic Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python ArchaicSeeker3.1-mamba \
    -t <path/to/target.vcf.gz> \
    -r <path/to/reference.vcf.gz> \
    -m <path/to/map.txt> \
    -o <path/to/output_folder> \
    [OPTIONS]