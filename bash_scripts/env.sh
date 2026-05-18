#!/bin/bash
# Source this file in every sbatch script.
# All parameters are read from the project config files — edit those, not this file.

ROOT="/projects/kernlab/akapoor/XPopVAE"
SNAKEFILE="$ROOT/Snakefile"
MODEL="OOA"

SIM_CFG="$ROOT/config_files/simulation_config_OOA.json"
VAE_YAML="$ROOT/config_files/model_hyperparams/vae.yaml"

NUM_DRAWS=$(python -c "import json; print(json.load(open('$SIM_CFG'))['num_draws'])")
NUM_REPLICATES=$(python -c "import json; print(json.load(open('$SIM_CFG'))['num_replicates'])")
SIM_NUMBER=0  # index within NUM_DRAWS; if num_draws > 1 this must be parameterized

# EXP_IDS derived from vae.yaml grid — matches Snakefile's sorted(EXPERIMENTS.keys())
mapfile -t EXP_IDS < <(python "$ROOT/bash_scripts/get_exp_ids.py" "$VAE_YAML")
NUM_EXP_IDS=${#EXP_IDS[@]}
