#!/bin/bash
# Array index encodes (replicate, exp_id):
#   IDX = replicate * NUM_EXP_IDS + exp_idx
# EXP_IDS and NUM_EXP_IDS are derived at runtime from vae.yaml via env.sh.
# The --array upper bound must match (NUM_REPLICATES * NUM_EXP_IDS - 1).
# submit_all.sh computes this automatically; for manual sbatch, adjust --array below.
#SBATCH --job-name=xpop_ld_tgtrain
#SBATCH --array=0-9
#SBATCH --output=logs/ld_tgtrain_%A_%a.out
#SBATCH --error=logs/ld_tgtrain_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -eo pipefail

module --ignore_cache purge || true
module --ignore_cache load cuda/11.8

source ~/miniforge3/etc/profile.d/conda.sh
conda activate PRS

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

source "/projects/kernlab/akapoor/XPopVAE/bash_scripts/env.sh"

REPLICATE=$(( SLURM_ARRAY_TASK_ID / NUM_EXP_IDS ))
EXP_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_EXP_IDS ))
EXP_ID="${EXP_IDS[$EXP_IDX]}"

TARGET="experiments/${MODEL}/vae/${SIM_NUMBER}/rep${REPLICATE}/${EXP_ID}/diagnostics/ld_decay_target_train/ld_decay_truth_vs_reconstructed.png"
ABS_TARGET="${ROOT}/${TARGET}"

if [[ -f "$ABS_TARGET" ]]; then
    echo "SKIP: $ABS_TARGET already exists"
    exit 0
fi

echo "Array $SLURM_ARRAY_TASK_ID → sim=$SIM_NUMBER rep=$REPLICATE exp=$EXP_ID"

snakemake --snakefile "$SNAKEFILE" \
          --directory  "$ROOT"      \
          --nolock                  \
          --latency-wait 120        \
          --rerun-incomplete        \
          --rerun-triggers mtime    \
          -j "$SLURM_CPUS_PER_TASK" \
          "$TARGET"

echo "Array task $SLURM_ARRAY_TASK_ID finished."
