#!/bin/bash
# Submit the full XPopVAE pipeline with SLURM dependencies.
# Array sizes are computed dynamically from the config files.
#
# Usage (from the cluster):
#   cd /projects/kernlab/akapoor/XPopVAE/bash_scripts
#   bash submit_all.sh
#
# Pipeline order:
#   run_simulation
#     └── build_genotypes
#           ├── run_gwas ──► run_standard_prs
#           ├── run_gblup
#           ├── run_baselines
#           └── train_vae
#                 ├── ld_decay_discovery
#                 ├── ld_decay_target_train
#                 ├── ld_decay_target_held_out
#                 └── diagnose_af_ld

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

# Load shared config (ROOT, MODEL, NUM_REPLICATES, NUM_EXP_IDS, EXP_IDS, ...)
source "$SCRIPT_DIR/env.sh"

# Array bounds (SLURM --array is 0-indexed inclusive)
REP_ARRAY="0-$(( NUM_REPLICATES - 1 ))"
REP_EXP_ARRAY="0-$(( NUM_REPLICATES * NUM_EXP_IDS - 1 ))"

echo "Config:"
echo "  NUM_REPLICATES = $NUM_REPLICATES"
echo "  NUM_EXP_IDS    = $NUM_EXP_IDS"
echo "  EXP_IDS        = ${EXP_IDS[*]}"
echo "  rep array      = $REP_ARRAY"
echo "  rep×exp array  = $REP_EXP_ARRAY"
echo ""

# 1. Simulations
JID_SIM=$(sbatch --parsable --array="$REP_ARRAY" run_simulation.sh)
echo "run_simulation           → job $JID_SIM  (array $REP_ARRAY)"

# 2. Build genotypes (after simulations)
JID_GENO=$(sbatch --parsable --array="$REP_ARRAY" --dependency=afterok:$JID_SIM build_genotypes.sh)
echo "build_genotypes          → job $JID_GENO  (array $REP_ARRAY)"

# 3. GWAS, gBLUP, baselines, VAE training (all after genotypes, run in parallel)
JID_GWAS=$(sbatch --parsable --array="$REP_ARRAY" --dependency=afterok:$JID_GENO run_gwas.sh)
echo "run_gwas                 → job $JID_GWAS  (array $REP_ARRAY)"

JID_GBLUP=$(sbatch --parsable --array="$REP_ARRAY" --dependency=afterok:$JID_GENO run_gblup.sh)
echo "run_gblup                → job $JID_GBLUP  (array $REP_ARRAY)"

JID_BL=$(sbatch --parsable --array="$REP_ARRAY" --dependency=afterok:$JID_GENO run_baselines.sh)
echo "run_baselines            → job $JID_BL  (array $REP_ARRAY)"

JID_VAE=$(sbatch --parsable --array="$REP_EXP_ARRAY" --dependency=afterok:$JID_GENO train_vae.sh)
echo "train_vae                → job $JID_VAE  (array $REP_EXP_ARRAY)"

# 4. Standard PRS (after GWAS)
JID_PRS=$(sbatch --parsable --array="$REP_ARRAY" --dependency=afterok:$JID_GWAS run_standard_prs.sh)
echo "run_standard_prs         → job $JID_PRS  (array $REP_ARRAY)"

# 5. Diagnostics (after VAE training)
JID_LD_DISC=$(sbatch --parsable --array="$REP_EXP_ARRAY" --dependency=afterok:$JID_VAE ld_decay_discovery.sh)
echo "ld_decay_discovery       → job $JID_LD_DISC  (array $REP_EXP_ARRAY)"

JID_LD_TGT=$(sbatch --parsable --array="$REP_EXP_ARRAY" --dependency=afterok:$JID_VAE ld_decay_target_train.sh)
echo "ld_decay_target_train    → job $JID_LD_TGT  (array $REP_EXP_ARRAY)"

JID_LD_HELD=$(sbatch --parsable --array="$REP_EXP_ARRAY" --dependency=afterok:$JID_VAE ld_decay_target_held_out.sh)
echo "ld_decay_target_held_out → job $JID_LD_HELD  (array $REP_EXP_ARRAY)"

JID_AF=$(sbatch --parsable --array="$REP_EXP_ARRAY" --dependency=afterok:$JID_VAE diagnose_af_ld.sh)
echo "diagnose_af_ld           → job $JID_AF  (array $REP_EXP_ARRAY)"

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
