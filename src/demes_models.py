from typing import Dict, Optional
import demes


def IM_symmetric_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + symmetric migration (YRI/CEU).
    YRI is constant size throughout (ancestral + modern).
    CEU has a bottleneck: small founding size N_CEU_bottleneck expanding to N_CEU.
    """

    required_keys = ["N_YRI", "N_CEU", "N_CEU_bottleneck", "m", "T_split", "T_bottleneck"]
    for k in required_keys:
        assert k in sampled, f"Missing required key: {k}"

    N_YRI         = float(sampled["N_YRI"])
    N_CEU         = float(sampled["N_CEU"])
    N_CEU_bot     = float(sampled["N_CEU_bottleneck"])
    T_split       = float(sampled["T_split"])
    T_bot         = float(sampled["T_bottleneck"])  # generations after split, so T_split > T_bot > 0
    m             = float(sampled["m"])

    assert T_split > T_bot > 0, "Need T_split > T_bottleneck > 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    # YRI: constant size, carries the root
    b.add_deme(
        "YRI",
        epochs=[dict(start_size=N_YRI, end_time=0)],
    )

    # CEU: branches off at T_split with bottleneck size, expands to N_CEU at T_bot
    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T_split,
        epochs=[
            dict(start_size=N_CEU_bot, end_time=T_bot),   # bottleneck epoch
            dict(start_size=N_CEU,     end_time=0),        # post-bottleneck expansion
        ],
    )

    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m, start_time=T_split, end_time=0)
        b.add_migration(source="CEU", dest="YRI", rate=m, start_time=T_split, end_time=0)

    return b.resolve()