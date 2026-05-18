#!/bin/bash
#SBATCH --job-name=xpop_gblup
#SBATCH --array=0-4
#SBATCH --output=logs/gblup_%A_%a.out
#SBATCH --error=logs/gblup_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate PRS

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPLICATE=$SLURM_ARRAY_TASK_ID

TARGET="experiments/${MODEL}/gblup/${SIM_NUMBER}/rep${REPLICATE}/summary.json"
ABS_TARGET="${ROOT}/${TARGET}"

if [[ -f "$ABS_TARGET" ]]; then
    echo "SKIP: $ABS_TARGET already exists"
    exit 0
fi

echo "Array $SLURM_ARRAY_TASK_ID → sim=$SIM_NUMBER rep=$REPLICATE"

snakemake --snakefile "$SNAKEFILE" \
          --directory  "$ROOT"      \
          --nolock                  \
          --latency-wait 120        \
          --rerun-incomplete        \
          --rerun-triggers mtime    \
          -j "$SLURM_CPUS_PER_TASK" \
          "$TARGET"

echo "Array task $SLURM_ARRAY_TASK_ID finished."
