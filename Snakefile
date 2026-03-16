# Snakefile

import json
from pathlib import Path
import yaml

##############################################################################
# VAE YAML + experiment config
##############################################################################
BASE_VAE_YAML = Path("config_files/model_hyperparams/vae.yaml")
BASE_VAE = yaml.safe_load(BASE_VAE_YAML.read_text())

EXP_CFG = "config_files/experiment_config_IM_symmetric.json"
CFG = json.loads(Path(EXP_CFG).read_text())
MODEL = CFG["demographic_model"]

##############################################################################
# Scripts
##############################################################################
TRAIN_VAE_SCRIPT = "snakemake_scripts/train_vae_wrapper.py"

##############################################################################
# Directories
##############################################################################
VAE_BASEDIR = Path(f"experiments/{MODEL}/vae")
VAE_BASEDIR.mkdir(parents=True, exist_ok=True)

##############################################################################
# Data inputs
##############################################################################
DISCOVERY_TRAIN = f"experiments/{MODEL}/processed_data/discovery_train.npy"
DISCOVERY_VAL = f"experiments/{MODEL}/processed_data/discovery_val.npy"
TARGET = f"experiments/{MODEL}/processed_data/target.npy"

##############################################################################
# Main targets
##############################################################################
rule all:
    input:
        VAE_BASEDIR / "vae_outputs/checkpoints/best_model.pt",
        VAE_BASEDIR / "vae_outputs/checkpoints/final_model.pt",
        VAE_BASEDIR / "vae_outputs/training_history.npz"


rule train_vae:
    input:
        vae_yaml=BASE_VAE_YAML,
        training_data=DISCOVERY_TRAIN,
        validation_data=DISCOVERY_VAL,
        target_data=TARGET,
        script=TRAIN_VAE_SCRIPT,
    output:
        best_model=VAE_BASEDIR / "vae_outputs/checkpoints/best_model.pt",
        final_model=VAE_BASEDIR / "vae_outputs/checkpoints/final_model.pt",
        history=VAE_BASEDIR / "vae_outputs/training_history.npz",
    params:
        outdir=VAE_BASEDIR,
    shell:
        r"""
        python {input.script} \
            --vae-config {input.vae_yaml} \
            --training-data {input.training_data} \
            --validation-data {input.validation_data} \
            --target-data {input.target_data} \
            --outputs {params.outdir}
        """