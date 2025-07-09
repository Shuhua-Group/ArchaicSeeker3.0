#!/bin/bash

# Enhanced Mamba Installation Script
#
# Features:
# - Automatically detects and uses mamba or conda.
# - Prioritizes pre-downloaded .whl files for offline installation.
# - Git Clone Fallback: Tries SSH first, falls back to HTTPS if it fails.
# - Robust pre-flight checks for dependencies (conda, git, pip, requirements.txt).
# - Automatic cleanup of cloned repositories on script exit (success or failure).
# - Retry mechanism for package installation from source.

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.

# Environment and package settings
readonly ENV_NAME="as3_mamba"
readonly PYTHON_VERSION="3.9"

# Git repository URLs
readonly MAMBA_REPO_SSH="git@github.com:state-spaces/mamba.git"
readonly MAMBA_REPO_HTTPS="https://github.com/state-spaces/mamba.git"
readonly CONV1D_REPO_SSH="git@github.com:Dao-AILab/causal-conv1d.git"
readonly CONV1D_REPO_HTTPS="https://github.com/Dao-AILab/causal-conv1d.git"

# Temporary directories for cloning
readonly MAMBA_DIR="mamba"
readonly CONV1D_DIR="causal-conv1d"

# --- Helper Functions ---

# A function to print informational messages.
log_info() {
    echo "INFO: $1"
}

# A function to print error messages and exit.
log_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# A function to print warning messages.
log_warn() {
    echo "WARNING: $1"
}

# Cleanup function to be called on script exit.
cleanup() {
    log_info "Cleaning up cloned repositories..."
    rm -rf "$MAMBA_DIR"
    rm -rf "$CONV1D_DIR"
}

# Set up trap to automatically call cleanup() on EXIT, INT, or TERM signals.
trap cleanup EXIT INT TERM

# A function to clone a git repository with SSH, with a fallback to HTTPS.
git_clone_with_fallback() {
    local repo_dir="$1"
    local repo_ssh="$2"
    local repo_https="$3" # <-- FIX: Corrected syntax error here.

    # Clean up any previous directory.
    if [ -d "$repo_dir" ]; then
        log_info "Removing existing directory '$repo_dir'..."
        rm -rf "$repo_dir"
    fi

    log_info "Attempting to clone '$repo_dir' using SSH..."
    if git clone "$repo_ssh" "$repo_dir"; then
        log_info "Successfully cloned using SSH."
    else
        log_warn "SSH clone failed. Falling back to HTTPS..."
        if git clone "$repo_https" "$repo_dir"; then
            log_info "Successfully cloned using HTTPS."
        else
            log_error "Failed to clone repository from both SSH and HTTPS sources."
        fi
    fi
}

# A function to install a package from a local .whl file or from source.
install_package() {
    local pkg_name="$1"
    local whl_pattern="$2"
    local repo_dir="$3"
    local repo_ssh="$4"
    local repo_https="$5"

    log_info "--- Step: Installing $pkg_name ---"

    # First, check for a pre-downloaded .whl file in the current directory.
    local whl_file
    whl_file=$(find . -maxdepth 1 -name "$whl_pattern" | head -n 1)

    if [ -f "$whl_file" ]; then
        log_info "Found pre-downloaded wheel file: $whl_file. Installing directly."
        if pip install "$whl_file"; then
            log_info "$pkg_name installed successfully from .whl file."
            return 0
        else
            log_warn "Installation from .whl file failed. Will attempt to build from source."
        fi
    fi

    # If no .whl file or installation from it failed, clone and install from source.
    log_info "No local $pkg_name wheel file found or installation failed. Cloning from source..."
    git_clone_with_fallback "$repo_dir" "$repo_ssh" "$repo_https"

    pushd "$repo_dir" > /dev/null # Enter the repo directory quietly

    local install_success=false
    for i in $(seq 1 3); do
        log_info "Attempting to install $pkg_name from source (Attempt $i/3)..."
        # --no-build-isolation is often needed for packages with complex C++/CUDA dependencies.
        if pip install . --no-build-isolation; then
            install_success=true
            break
        elif [ $i -lt 3 ]; then
            log_warn "Attempt $i failed. Retrying in 10 seconds..."
            sleep 10
        fi
    done

    popd > /dev/null # Return to the original directory

    if [ "$install_success" = false ]; then
        log_error "Failed to install $pkg_name after 3 attempts. Please check the build logs."
    fi
    log_info "$pkg_name installed successfully from source."
}


# --- Main Script ---
log_info "Starting the smart installation script."

# --- Step 1: Pre-flight checks ---
log_info "--- Step 1/5: Running pre-flight checks ---"

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    log_info "Found 'mamba'. Using it as the package manager."
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    log_info "Found 'conda'. Using it as the package manager."
else
    log_error "No Conda or Mamba installation found. Please install Miniconda or Miniforge first."
fi

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    log_error "'requirements.txt' not found in the current directory."
fi

# --- Step 2: Create Conda Environment ---
log_info "--- Step 2/5: Creating/Verifying Conda environment '$ENV_NAME' ---"
if $CONDA_CMD info --envs | grep -q "^${ENV_NAME}\s"; then
    log_info "Environment '$ENV_NAME' already exists. Skipping creation."
else
    log_info "Creating minimal base environment '$ENV_NAME'..."
    # Install python, pip, and git to ensure they are available.
    $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" pip git -c conda-forge -y
fi

# --- Step 3: Activate Environment and Install Dependencies ---
log_info "--- Step 3/5: Activating environment and installing dependencies ---"

# Activate the environment
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Verify that pip and git are available in the new environment
if ! command -v pip &> /dev/null || ! command -v git &> /dev/null; then
    log_error "pip or git command not found after activating the environment. Installation failed."
fi

log_info "Installing packages from requirements.txt..."
pip install -r requirements.txt --no-cache-dir

# --- Step 4: Install mamba-ssm and causal-conv1d ---
install_package "mamba-ssm" "mamba_ssm*.whl" "$MAMBA_DIR" "$MAMBA_REPO_SSH" "$MAMBA_REPO_HTTPS"
install_package "causal-conv1d" "causal_conv1d*.whl" "$CONV1D_DIR" "$CONV1D_REPO_SSH" "$CONV1D_REPO_HTTPS"

# --- Step 5: Finalization ---
# The cleanup function will run automatically on exit, so no need to call it here.
log_info "--- Step 5/5: Finalizing installation ---"
echo ""
echo "===================================================================="
echo "SUCCESS! The environment '$ENV_NAME' is ready."
echo "To activate it, run:"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "===================================================================="


