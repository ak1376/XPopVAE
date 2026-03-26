import copy
import itertools
import json
import re
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
def set_nested(d, path, value):
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value

def get_nested(d, path):
    keys = path.split(".")
    cur = d
    for key in keys:
        cur = cur[key]
    return cur

def sanitize_tag_value(x):
    s = str(x)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s

def make_exp_id(prefix, sep, assignments, dims):
    parts = [prefix] if prefix else []
    for dim, value in zip(dims, assignments):
        tag = dim.get("tag", dim["path"].replace(".", "_"))
        parts.append(f"{tag}{sanitize_tag_value(value)}")
    return sep.join(parts) if parts else "default"

def build_experiment_grid(base_cfg):
    grid_cfg = base_cfg.get("grid", {})
    enabled = bool(grid_cfg.get("enabled", False))
    dims = grid_cfg.get("dims", [])

    if not enabled or len(dims) == 0:
        return {
            "default": {
                "config": copy.deepcopy(base_cfg),
                "assignments": {}
            }
        }

    prefix = grid_cfg.get("name", {}).get("prefix", "experiment")
    sep = grid_cfg.get("name", {}).get("sep", "__")

    value_lists = [dim["values"] for dim in dims]
    combos = itertools.product(*value_lists)

    experiments = {}
    for combo in combos:
        cfg = copy.deepcopy(base_cfg)
        assignments = {}

        for dim, value in zip(dims, combo):
            set_nested(cfg, dim["path"], value)
            assignments[dim["path"]] = value

        exp_id = make_exp_id(prefix, sep, combo, dims)
        experiments[exp_id] = {
            "config": cfg,
            "assignments": assignments,
        }

    return experiments

# -----------------------------------------------------------------------------
# Load config files
# -----------------------------------------------------------------------------
EXP_CFG_PATH = Path("/sietch_colab/akapoor/XPopVAE/config_files/experiment_config_IM_symmetric.json")
VAE_YAML_PATH = Path("/sietch_colab/akapoor/XPopVAE/config_files/model_hyperparams/vae.yaml")

EXP_CFG = json.loads(EXP_CFG_PATH.read_text())
VAE_CFG = yaml.safe_load(VAE_YAML_PATH.read_text())

EXPERIMENTS = build_experiment_grid(VAE_CFG)
EXP_IDS = sorted(EXPERIMENTS.keys())

print("Resolved VAE experiments:")
for exp_id, spec in EXPERIMENTS.items():
    print(f"  {exp_id}: {spec['assignments']}")

# -----------------------------------------------------------------------------
# Core settings from experiment config
# -----------------------------------------------------------------------------
MODEL = EXP_CFG["demographic_model"]
NUM_DRAWS = int(EXP_CFG.get("num_draws", 1))
NUM_REPLICATES = int(EXP_CFG.get("num_replicates", 1))
MAF_THRESHOLD = float(EXP_CFG.get("maf_threshold", 0.0))
DISCOVERY_POP = str(EXP_CFG.get("discovery", "CEU"))

SUBSET_SNPS = int(EXP_CFG.get("subset_snps", 10000))
SUBSET_BP = EXP_CFG.get("subset_bp", None)
SUBSET_MODE = str(EXP_CFG.get("subset_mode", "random"))
SUBSET_SEED = int(EXP_CFG.get("subset_seed", 295))
VAL_FRAC = float(EXP_CFG.get("val_frac", 0.2))
SPLIT_SEED = int(EXP_CFG.get("split_seed", 42))

# -----------------------------------------------------------------------------
# Optional: inspect VAE YAML grid
# -----------------------------------------------------------------------------
VAE_GRID_CFG = VAE_CFG.get("grid", {})
VAE_GRID_ENABLED = bool(VAE_GRID_CFG.get("enabled", False))
VAE_GRID_DIMS = VAE_GRID_CFG.get("dims", [])

print("Loaded experiment config:")
print(f"  MODEL={MODEL}")
print(f"  NUM_DRAWS={NUM_DRAWS}")
print(f"  NUM_REPLICATES={NUM_REPLICATES}")
print(f"  MAF_THRESHOLD={MAF_THRESHOLD}")
print(f"  DISCOVERY_POP={DISCOVERY_POP}")

print("Loaded VAE config:")
print(f"  grid.enabled={VAE_GRID_ENABLED}")
if VAE_GRID_ENABLED:
    for dim in VAE_GRID_DIMS:
        print(f"  grid dim: path={dim.get('path')} values={dim.get('values')} tag={dim.get('tag')}")

# -----------------------------------------------------------------------------
# Scripts
# -----------------------------------------------------------------------------
SIM_SCRIPT = "snakemake_scripts/run_simulation.py"
BUILD_GT_SCRIPT = "snakemake_scripts/run_build_genotypes.py"
TRAIN_VAE_SCRIPT = "snakemake_scripts/train_vae_wrapper.py"
COMPARE_LD_SCRIPT = "snakemake_scripts/compare_ld_decay.py"
DIAGNOSE_AF_LD_SCRIPT = "snakemake_scripts/diagnose_allelefreq_vs_ld.py"
BASELINE_SCRIPT = "snakemake_scripts/baseline_predictors.py"

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
SIM_BASEDIR = Path(f"experiments/{MODEL}/simulations")
PROC_BASEDIR = Path(f"experiments/{MODEL}/processed_data")
VAE_BASEDIR = Path(f"experiments/{MODEL}/vae")

SIM_BASEDIR.mkdir(parents=True, exist_ok=True)
PROC_BASEDIR.mkdir(parents=True, exist_ok=True)
VAE_BASEDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Wildcard values from config
# -----------------------------------------------------------------------------
SIM_NUMBERS = [str(i) for i in range(NUM_DRAWS)]
REPLICATES = [str(i) for i in range(NUM_REPLICATES)]

# -----------------------------------------------------------------------------
# Use sim 0 / rep0 processed data for VAE training
# -----------------------------------------------------------------------------
DISCOVERY_TRAIN = PROC_BASEDIR / "0/rep0/discovery_train.npy"
DISCOVERY_VAL = PROC_BASEDIR / "0/rep0/discovery_val.npy"
TARGET = PROC_BASEDIR / "0/rep0/target.npy"

# Define the phenotype paths as well
DISCOVERY_TRAIN_PHENO = PROC_BASEDIR / "0/rep0/discovery_train_pheno.npy"
DISCOVERY_VAL_PHENO = PROC_BASEDIR / "0/rep0/discovery_val_pheno.npy"
TARGET_PHENO = PROC_BASEDIR / "0/rep0/target_pheno.npy"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def sim_dir(wc):
    return SIM_BASEDIR / wc.sim_number / f"rep{wc.replicate}"

def proc_dir(wc):
    return PROC_BASEDIR / wc.sim_number / f"rep{wc.replicate}"

def exp_dir(wc):
    return VAE_BASEDIR / wc.exp_id

def exp_checkpoint_dir(wc):
    return exp_dir(wc) / "vae_outputs/checkpoints"

def exp_ld_diag_dir(wc):
    return exp_dir(wc) / "diagnostics/ld_decay_discovery_val"

def exp_af_ld_diag_dir(wc):
    return exp_dir(wc) / "diagnostics/allelefreq_vs_ld_discovery_val"

def exp_config_path(wc):
    return exp_dir(wc) / "resolved_vae_config.yaml"

# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
def set_nested(d, path, value):
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value

def get_nested(d, path):
    keys = path.split(".")
    cur = d
    for key in keys:
        cur = cur[key]
    return cur

def sanitize_tag_value(x):
    s = str(x)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s

def make_exp_id(prefix, sep, assignments, dims):
    parts = [prefix] if prefix else []
    for dim, value in zip(dims, assignments):
        tag = dim.get("tag", dim["path"].replace(".", "_"))
        parts.append(f"{tag}{sanitize_tag_value(value)}")
    return sep.join(parts) if parts else "default"

def build_experiment_grid(base_cfg):
    grid_cfg = base_cfg.get("grid", {})
    enabled = bool(grid_cfg.get("enabled", False))
    dims = grid_cfg.get("dims", [])

    if not enabled or len(dims) == 0:
        return {
            "default": {
                "config": copy.deepcopy(base_cfg),
                "assignments": {}
            }
        }

    prefix = grid_cfg.get("name", {}).get("prefix", "experiment")
    sep = grid_cfg.get("name", {}).get("sep", "__")

    value_lists = [dim["values"] for dim in dims]
    combos = itertools.product(*value_lists)

    experiments = {}
    for combo in combos:
        cfg = copy.deepcopy(base_cfg)
        assignments = {}
        for dim, value in zip(dims, combo):
            set_nested(cfg, dim["path"], value)
            assignments[dim["path"]] = value

        exp_id = make_exp_id(prefix, sep, combo, dims)
        experiments[exp_id] = {
            "config": cfg,
            "assignments": assignments,
        }

    return experiments

# -----------------------------------------------------------------------------
# Final targets
# -----------------------------------------------------------------------------
rule all:
    input:
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_train.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_val.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/target.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),

        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_train_pheno.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_val_pheno.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/target_pheno.npy",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),

        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/meta.pkl",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),
        expand(PROC_BASEDIR / "{sim_number}/rep{replicate}/hap_meta.pkl",
               sim_number=SIM_NUMBERS, replicate=REPLICATES),

        expand(VAE_BASEDIR / "{exp_id}/resolved_vae_config.yaml",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/best_model.pt",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/final_model.pt",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/vae_outputs/training_history.npz",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/ld_decay_truth_vs_reconstructed.png",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/ld_decay_summary.txt",
               exp_id=EXP_IDS),

        expand(VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/ld_decay_truth_vs_reconstructed.png",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/ld_decay_summary.txt",
               exp_id=EXP_IDS),
        expand(VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/diagnostic_summary.txt",
               exp_id=EXP_IDS),
        PROC_BASEDIR / "0/rep0/baselines/baseline_results.txt",

# -----------------------------------------------------------------------------
# 1. Run one simulation
# -----------------------------------------------------------------------------
rule run_simulation:
    input:
        script=SIM_SCRIPT,
        experiment_config=EXP_CFG_PATH,
    output:
        tree=SIM_BASEDIR / "{sim_number}/rep{replicate}/tree_sequence.trees",
        phenotype=SIM_BASEDIR / "{sim_number}/rep{replicate}/phenotype.pkl",
        effect_sizes=SIM_BASEDIR / "{sim_number}/rep{replicate}/effect_sizes.pkl",
        sfs=SIM_BASEDIR / "{sim_number}/rep{replicate}/SFS.pkl",
        sampled_params=SIM_BASEDIR / "{sim_number}/rep{replicate}/sampled_params.pkl",
        demes_png=SIM_BASEDIR / "{sim_number}/rep{replicate}/demes.png",
    params:
        simulation_dir=SIM_BASEDIR,
        output_dir=lambda wc: SIM_BASEDIR / wc.sim_number / f"rep{wc.replicate}",
        model_type=MODEL,
    shell:
        r"""
        python {input.script} \
            --simulation-dir {params.simulation_dir} \
            --experiment-config {input.experiment_config} \
            --model-type {params.model_type} \
            --simulation-number {wildcards.sim_number} \
            --replicate {wildcards.replicate} \
            --output-dir {params.output_dir}
        """

# -----------------------------------------------------------------------------
# 2. Build genotype arrays from one simulation
# -----------------------------------------------------------------------------
rule build_genotypes:
    input:
        script=BUILD_GT_SCRIPT,
        tree=SIM_BASEDIR / "{sim_number}/rep{replicate}/tree_sequence.trees",
        phenotype=SIM_BASEDIR / "{sim_number}/rep{replicate}/phenotype.pkl",
        experiment_config=EXP_CFG_PATH,
    output:
        discovery_train=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_train.npy",
        discovery_val=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_val.npy",
        target=PROC_BASEDIR / "{sim_number}/rep{replicate}/target.npy",

        discovery_train_pheno=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_train_pheno.npy",
        discovery_val_pheno=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_val_pheno.npy",
        target_pheno=PROC_BASEDIR / "{sim_number}/rep{replicate}/target_pheno.npy",

        meta=PROC_BASEDIR / "{sim_number}/rep{replicate}/meta.pkl",
        hap_meta=PROC_BASEDIR / "{sim_number}/rep{replicate}/hap_meta.pkl",
        all_individuals=PROC_BASEDIR / "{sim_number}/rep{replicate}/all_individuals.npy",
        hap1=PROC_BASEDIR / "{sim_number}/rep{replicate}/hap1.npy",
        hap2=PROC_BASEDIR / "{sim_number}/rep{replicate}/hap2.npy",
        snp_index=PROC_BASEDIR / "{sim_number}/rep{replicate}/snp_index.npy",
        variant_positions=PROC_BASEDIR / "{sim_number}/rep{replicate}/variant_positions_bp.npy",
        variant_site_ids=PROC_BASEDIR / "{sim_number}/rep{replicate}/variant_site_ids.npy",
        ts_individual_ids=PROC_BASEDIR / "{sim_number}/rep{replicate}/ts_individual_ids.npy",
        disc_train_idx=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_train_idx.npy",
        disc_val_idx=PROC_BASEDIR / "{sim_number}/rep{replicate}/discovery_val_idx.npy",
        target_idx=PROC_BASEDIR / "{sim_number}/rep{replicate}/target_idx.npy",
        site_filter_report=PROC_BASEDIR / "{sim_number}/rep{replicate}/site_filter_report.txt",
        genotype_site_stats=PROC_BASEDIR / "{sim_number}/rep{replicate}/genotype_site_stats.txt",
        train_mono_filter_report=PROC_BASEDIR / "{sim_number}/rep{replicate}/train_mono_filter_report.txt",
    params:
        outdir=lambda wc: PROC_BASEDIR / wc.sim_number / f"rep{wc.replicate}",
        maf_threshold=MAF_THRESHOLD,
        subset_snps=SUBSET_SNPS,
        subset_mode=SUBSET_MODE,
        subset_seed=SUBSET_SEED,
        val_frac=VAL_FRAC,
        split_seed=SPLIT_SEED,
        discovery_pop=DISCOVERY_POP,
    shell:
        r"""
        python {input.script} \
            --tree {input.tree} \
            --phenotype {input.phenotype} \
            --outdir {params.outdir} \
            --experiment-config-json {input.experiment_config} \
            --maf-threshold {params.maf_threshold} \
            --subset-snps {params.subset_snps} \
            --subset-mode {params.subset_mode} \
            --subset-seed {params.subset_seed} \
            --val-frac {params.val_frac} \
            --split-seed {params.split_seed} \
            --discovery-pop {params.discovery_pop}
        """

rule train_vae:
    input:
        vae_yaml=VAE_BASEDIR / "{exp_id}/resolved_vae_config.yaml",
        training_data=DISCOVERY_TRAIN,
        validation_data=DISCOVERY_VAL,

        # THE BELOW PHENOTYPES ARE HARDCODED TO USE THE SANITY CHECK ONES. 
        training_pheno = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_train.npy',
        validation_pheno = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_val.npy',
        target_pheno = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_target.npy',
        target_data=TARGET,
        script=TRAIN_VAE_SCRIPT,
    output:
        best_model=VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/best_model.pt",
        final_model=VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/final_model.pt",
        history=VAE_BASEDIR / "{exp_id}/vae_outputs/training_history.npz",
    params:
        outdir=lambda wc: VAE_BASEDIR / wc.exp_id,
    shell:
        r"""
        python {input.script} \
            --vae-config {input.vae_yaml} \
            --training-data {input.training_data} \
            --validation-data {input.validation_data} \
            --target-data {input.target_data} \
            --training-pheno {input.training_pheno} \
            --validation-pheno {input.validation_pheno} \
            --target-pheno {input.target_pheno} \
            --outputs {params.outdir}
        """

rule compare_ld_decay_discovery:
    input:
        checkpoint=VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/best_model.pt",
        genotype_npy=DISCOVERY_VAL,
        variant_positions=PROC_BASEDIR / "0/rep0/variant_positions_bp.npy",
        script=COMPARE_LD_SCRIPT,
    output:
        reconstructed=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/reconstructed_genotypes_argmax.npy",
        curves=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/ld_decay_curves.npz",
        plot=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/ld_decay_truth_vs_reconstructed.png",
        summary=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_discovery_val/ld_decay_summary.txt",
    params:
        output_dir=lambda wc: VAE_BASEDIR / wc.exp_id / "diagnostics/ld_decay_discovery_val",
        batch_size=128,
        distance_mode="bp",
        max_bp_distance=50000,
        bp_bin_size=1000,
        label="discovery_val",
        title=lambda wc: f"LD decay: Discovery Val ({wc.exp_id})",
    shell:
        r"""
        python {input.script} \
            --checkpoint {input.checkpoint} \
            --genotype-npy {input.genotype_npy} \
            --variant-positions-npy {input.variant_positions} \
            --output-dir {params.output_dir} \
            --batch-size {params.batch_size} \
            --distance-mode {params.distance_mode} \
            --max-bp-distance {params.max_bp_distance} \
            --bp-bin-size {params.bp_bin_size} \
            --label {params.label} \
            --title "{params.title}" \
            --include-metrics-in-title
        """


rule compare_ld_decay_target:
    input:
        checkpoint=VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/best_model.pt",
        genotype_npy=TARGET,
        variant_positions=PROC_BASEDIR / "0/rep0/variant_positions_bp.npy",
        script=COMPARE_LD_SCRIPT,
    output:
        reconstructed=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/reconstructed_genotypes_argmax.npy",
        curves=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/ld_decay_curves.npz",
        plot=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/ld_decay_truth_vs_reconstructed.png",
        summary=VAE_BASEDIR / "{exp_id}/diagnostics/ld_decay_target/ld_decay_summary.txt",
    params:
        output_dir=lambda wc: VAE_BASEDIR / wc.exp_id / "diagnostics/ld_decay_target",
        batch_size=128,
        distance_mode="bp",
        max_bp_distance=50000,
        bp_bin_size=1000,
        label="target_yri",
        title=lambda wc: f"LD decay: Target/YRI ({wc.exp_id})",
    shell:
        r"""
        python {input.script} \
            --checkpoint {input.checkpoint} \
            --genotype-npy {input.genotype_npy} \
            --variant-positions-npy {input.variant_positions} \
            --output-dir {params.output_dir} \
            --batch-size {params.batch_size} \
            --distance-mode {params.distance_mode} \
            --max-bp-distance {params.max_bp_distance} \
            --bp-bin-size {params.bp_bin_size} \
            --label {params.label} \
            --title "{params.title}" \
            --include-metrics-in-title
        """
        
rule diagnose_allelefreq_vs_ld:
    input:
        checkpoint=VAE_BASEDIR / "{exp_id}/vae_outputs/checkpoints/best_model.pt",
        train_genotype_npy=DISCOVERY_TRAIN,
        eval_genotype_npy=DISCOVERY_VAL,
        script=DIAGNOSE_AF_LD_SCRIPT,
    output:
        reconstructed_eval=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/reconstructed_eval_argmax.npy",
        reconstructed_eval_shuffled=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/reconstructed_eval_argmax_shuffled_input.npy",
        reconstructed_baseline=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/reconstructed_eval_frequency_baseline.npy",
        snp_permutation=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/snp_permutation.npy",
        maf_eval=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/maf_eval.npy",
        per_snp_bal_acc_vae=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/per_snp_bal_acc_vae.npy",
        per_snp_bal_acc_baseline=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/per_snp_bal_acc_baseline.npy",
        plot=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/balanced_accuracy_vs_maf.png",
        maf_summary=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/maf_accuracy_summary.tsv",
        summary_txt=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/diagnostic_summary.txt",
        summary_npz=VAE_BASEDIR / "{exp_id}/diagnostics/allelefreq_vs_ld_discovery_val/diagnostic_summary.npz",
    params:
        output_dir=lambda wc: VAE_BASEDIR / wc.exp_id / "diagnostics/allelefreq_vs_ld_discovery_val",
        batch_size=128,
        seed=0,
        maf_bins="0 0.01 0.05 0.1 0.2 0.3 0.4 0.5",
    shell:
        r"""
        python {input.script} \
            --checkpoint {input.checkpoint} \
            --train-genotype-npy {input.train_genotype_npy} \
            --eval-genotype-npy {input.eval_genotype_npy} \
            --output-dir {params.output_dir} \
            --batch-size {params.batch_size} \
            --seed {params.seed} \
            --maf-bins {params.maf_bins}
        """

rule write_vae_config:
    output:
        config=VAE_BASEDIR / "{exp_id}/resolved_vae_config.yaml",
    run:
        exp_id = wildcards.exp_id
        cfg = copy.deepcopy(EXPERIMENTS[exp_id]["config"])

        outdir = Path(output.config).parent
        outdir.mkdir(parents=True, exist_ok=True)

        with open(output.config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

rule run_baselines:
    input:
        script      = BASELINE_SCRIPT,
        x_train     = DISCOVERY_TRAIN,
        # TODO: The current phenotype files are using the sanity check ones. 
        y_train     = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_train.npy',
        x_val       = DISCOVERY_VAL,
        y_val       = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_val.npy',
        x_test      = TARGET,
        y_test      = '/sietch_colab/akapoor/XPopVAE/phenotype_creation/simulated_phenotype_target.npy',
    output:
        results = PROC_BASEDIR / "0/rep0/baselines/baseline_results.txt",
    params:
        out_dir = PROC_BASEDIR / "0/rep0/baselines",
        h2      = float(EXP_CFG.get("h2", 0.7)),
        seed    = 42,
    shell:
        r"""
        python {input.script} \
            --x_train {input.x_train} \
            --y_train {input.y_train} \
            --x_val   {input.x_val}   \
            --y_val   {input.y_val}   \
            --x_test  {input.x_test}  \
            --y_test  {input.y_test}  \
            --out_dir {params.out_dir} \
            --h2      {params.h2}      \
            --seed    {params.seed}
        """