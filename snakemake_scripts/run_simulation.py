#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import demesdraw
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation import (
    simulation,
    create_SFS,
    calculate_fst,
    sample_params,
)


def run_simulation(
    simulation_dir: Path,
    experiment_config: Path,
    model_type: str,
    simulation_number: Optional[str] = None,
    replicate: int = 0,
    output_dir: Optional[Path] = None,
):
    cfg = json.loads(experiment_config.read_text())
    engine = cfg["engine"]

    if output_dir is not None:
        out_dir = output_dir
        if simulation_number is None:
            simulation_number = "0000"
    else:
        if simulation_number is None:
            existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
            simulation_number = f"{max(existing, default=0) + 1:04d}"
        out_dir = simulation_dir / simulation_number

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── seed ──────────────────────────────────────────────────────────────────
    base_seed = cfg.get("seed")
    if base_seed is not None:
        simulation_seed = int(base_seed) + int(simulation_number) * 1000 + int(replicate)
        print(f"• Using seed {simulation_seed} (base={base_seed}, grid={simulation_number}, rep={replicate})")
        rng = np.random.default_rng(simulation_seed)
    else:
        simulation_seed = None
        rng = np.random.default_rng()
        print("• No seed specified, using random state")

    # ── sample demographic params ─────────────────────────────────────────────
    grid_cfg = cfg.get("grid_sampling", {})
    if grid_cfg.get("enabled", False):
        fixed     = grid_cfg["fixed_params"]
        var_param = grid_cfg["varying_param"]
        min_val   = float(grid_cfg["min_value"])
        max_val   = float(grid_cfg["max_value"])
        scale     = grid_cfg.get("scale", "linear")
        num_grid  = int(grid_cfg.get("num_grid_points", 1))
        grid_idx  = int(simulation_number)

        if grid_idx < 0 or grid_idx >= num_grid:
            raise ValueError(f"Simulation number {grid_idx} out of range [0, {num_grid}).")

        grid_vals = (
            np.linspace(min_val, max_val, num_grid)
            if scale == "linear"
            else np.geomspace(min_val, max_val, num_grid)
        )
        val = float(grid_vals[grid_idx])
        sampled_params = dict(fixed)
        sampled_params[var_param] = val
        print(f"• Grid mode: {var_param}={val:.2f} (grid {grid_idx}/{num_grid-1}, rep {replicate})")
    else:
        sampled_params = sample_params(cfg["priors"], rng=rng)

    if engine != "msprime":
        raise ValueError("engine must be 'msprime'.")
    print("• engine=msprime → neutral simulation")

    # ── run simulation ────────────────────────────────────────────────────────
    sim_cfg = dict(cfg)
    if simulation_seed is not None:
        sim_cfg["seed"] = simulation_seed

    ts, g = simulation(
        sampled_params=sampled_params,
        model_type=model_type,
        experiment_config=sim_cfg,
    )

    # ── compute + save artefacts ──────────────────────────────────────────────
    sfs     = create_SFS(ts)
    fst_val = calculate_fst(ts)
    sampled_params["Fst"] = fst_val
    print(f"• Fst (YRI-CEU): {fst_val:.4f}")

    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts.dump(out_dir / "tree_sequence.trees")

    fig_path = out_dir / "demes.png"
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    try:
        rel = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_dir
    print(f"✓ simulation written to {rel}")


def main():
    cli = argparse.ArgumentParser(description="Generate one neutral simulation (msprime)")
    cli.add_argument("--simulation-dir",   type=Path, required=True)
    cli.add_argument("--experiment-config", type=Path, required=True)
    cli.add_argument("--model-type",       required=True, choices=["IM_symmetric", "OOA"])
    cli.add_argument("--simulation-number", type=str, default=None)
    cli.add_argument("--replicate",        type=int, default=0)
    cli.add_argument("--output-dir",       type=Path, default=None)
    args = cli.parse_args()

    run_simulation(
        args.simulation_dir,
        args.experiment_config,
        args.model_type,
        args.simulation_number,
        args.replicate,
        args.output_dir,
    )


if __name__ == "__main__":
    main()