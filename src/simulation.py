from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any

import demes
import numpy as np
import stdpopsim as sps
import tskit
import tstrait
import moments
import math
import sys
from pathlib import Path
import msprime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demes_models import IM_symmetric_model, OOA

# ──────────────────────────────────
# Minimal helpers
# ──────────────────────────────────

def sample_params(
    priors: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params: Dict[str, float] = {}

    for k, bounds in priors.items():
        params[k] = float(rng.uniform(*bounds))

    return params


def build_demes_graph(
    model_type: str, sampled_params: Dict[str, float]
) -> demes.Graph:
    """
    Build a Demes graph for the given model_type + sampled_params.
    Mirrors the logic inside simulation(...), but returns only the graph.
    """
    if model_type == "IM_symmetric":
        return IM_symmetric_model(sampled_params)
    elif model_type == "OOA":
        return OOA(sampled_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    

def simulation_runner(
    g: demes.Graph, experiment_config: Dict[str, Any]
) -> Tuple[tskit.TreeSequence, demes.Graph]:
    
    species = experiment_config.get("species", "HomSap")
    sp = sps.get_species(species)

    # Direct instantiation — no subclass needed
    model = sps.DemographicModel(
        id="model",
        description="model",
        long_description="model",
        model=msprime.Demography.from_demes(g),
        generation_time=1,
    )

    genome_length = experiment_config["genome_length"]
    mutation_rate = experiment_config["mutation_rate"]
    recombination_rate = experiment_config['recombination_rate']

    contig = sp.get_contig(
          chromosome=None,
          length=genome_length,
          mutation_rate=mutation_rate,
          recombination_rate=recombination_rate,
      )
    
    print(f'• Using engine: {experiment_config.get("engine")}')
    print("contig.length:", contig.length)
    print("contig.mutation_rate:", contig.mutation_rate)
    print("contig.recombination_map.mean_rate:", contig.recombination_map.mean_rate)

    samples = {
        k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()
    }
    seed = experiment_config.get("seed", None)

    eng = sps.get_engine("msprime")

    ts = eng.simulate(
        model,
        contig,
        samples,
        seed=seed
    )

    return ts, g


def simulation(
    sampled_params: Dict[str, float],
    model_type: str,
    experiment_config: Dict[str, Any],
) -> Tuple[tskit.TreeSequence, demes.Graph]:
    # Build demes graph

    g = build_demes_graph(model_type=model_type, sampled_params=sampled_params)

    return simulation_runner(
        g, experiment_config
    )



def _individual_genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return an (num_individuals, num_sites) genotype matrix with diploid genotypes.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples/nodes)
    if ts.num_individuals == 0:
        return G_hap.T  # treat haplotypes as individuals

    num_inds = ts.num_individuals
    num_sites = ts.num_sites
    G_ind = np.zeros((num_inds, num_sites), dtype=np.float32)

    for i, ind in enumerate(ts.individuals()):
        nodes = ind.nodes
        if len(nodes) > 0:
            G_ind[i] = G_hap[:, nodes].sum(axis=1)  # sum over haplotypes

    return G_ind

def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """Build a moments.Spectrum using pops that have sampled individuals."""
    sample_sets: List[np.ndarray] = []
    pop_ids: List[str] = []
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps):
            sample_sets.append(samps)
            meta = pop.metadata if isinstance(pop.metadata, dict) else {}
            pop_ids.append(meta.get("name", f"pop{pop.id}"))
    if not sample_sets:
        raise ValueError("No sampled populations found.")
    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets, mode="site", polarised=True, span_normalise=False
    )
    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs



def calculate_fst(ts: tskit.TreeSequence) -> float:
    import numpy as np

    pop_to_samps = {}
    pop_to_name = {}

    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps) == 0:
            continue
        meta = pop.metadata if isinstance(pop.metadata, dict) else {}
        name = meta.get("name", f"pop{pop.id}")
        pop_to_samps[name] = samps
        pop_to_name[pop.id] = name

    # Choose pair in preferred order
    if "YRI" in pop_to_samps and "CEU" in pop_to_samps:
        s1, s2 = pop_to_samps["YRI"], pop_to_samps["CEU"]
    elif "AFR" in pop_to_samps and "EUR" in pop_to_samps:
        s1, s2 = pop_to_samps["AFR"], pop_to_samps["EUR"]
    else:
        # fallback: first two
        if len(pop_to_samps) < 2:
            return 0.0
        names = list(pop_to_samps.keys())
        s1, s2 = pop_to_samps[names[0]], pop_to_samps[names[1]]

    try:
        fst_val = ts.Fst(
            sample_sets=[s1, s2],
            indexes=[(0, 1)],
            windows=None,
            mode="site",
            span_normalise=True,
        )
        return float(fst_val[0])
    except Exception as e:
        print(f"Warning: Fst calculation failed: {e}")
        return 0.0
