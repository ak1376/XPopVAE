from typing import Dict, Optional
import demes
import math


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

def OOA(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Two-population Out-of-Africa model (Tennessen et al. / Fu et al.)
    YRI: three epochs — deep ancestral constant, post-expansion constant, then exponential growth.
    CEU: splits from YRI at T_OOA with a founder bottleneck, then two phases of exponential growth.
    Migration rate is higher during the founder phase and drops after CEU begins expanding.
    """

    required_keys = [
        "N_anc", "N_YRI_anc", "N_CEU_founder", "N_CEU_early",
        "r_YRI", "r_CEU_early", "r_CEU",
        "T_AF", "T_OOA", "T_CEU_expand", "T_growth",
        "m_founder", "m_modern",
    ]
    for k in required_keys:
        assert k in sampled, f"Missing required key: {k}"

    N_anc        = float(sampled["N_anc"])
    N_YRI_anc    = float(sampled["N_YRI_anc"])
    N_CEU_founder= float(sampled["N_CEU_founder"])
    N_CEU_early  = float(sampled["N_CEU_early"])
    r_YRI        = float(sampled["r_YRI"])
    r_CEU_early  = float(sampled["r_CEU_early"])
    r_CEU        = float(sampled["r_CEU"])
    T_AF         = float(sampled["T_AF"])
    T_OOA        = float(sampled["T_OOA"])
    T_CEU_expand = float(sampled["T_CEU_expand"])
    T_growth     = float(sampled["T_growth"])
    m_founder    = float(sampled["m_founder"])
    m_modern     = float(sampled["m_modern"])

    assert T_AF > T_OOA > T_CEU_expand > T_growth > 0, \
        "Need T_AF > T_OOA > T_CEU_expand > T_growth > 0"

    # Derive present-day and boundary sizes from growth rates
    # N_CEU_early is the size at T_CEU_expand; grow it forward to T_growth
    N_CEU_at_Tgrowth = N_CEU_early * math.exp(r_CEU_early * (T_CEU_expand - T_growth))
    # then grow at the faster rate from T_growth to present
    N_CEU = N_CEU_at_Tgrowth * math.exp(r_CEU * T_growth)
    # YRI grows from N_YRI_anc at T_growth to present
    N_YRI = N_YRI_anc * math.exp(r_YRI * T_growth)

    b = demes.Builder(time_units="generations", generation_time=1)

    # YRI: carries the root
    b.add_deme(
        "YRI",
        description="African population (YRI)",
        epochs=[
            dict(start_size=N_anc,     end_time=T_AF,        size_function="constant"),
            dict(start_size=N_YRI_anc, end_time=T_growth,    size_function="constant"),
            dict(start_size=N_YRI_anc, end_size=N_YRI,       end_time=0, size_function="exponential"),
        ],
    )

    # CEU: splits from YRI at T_OOA
    b.add_deme(
        "CEU",
        description="European population (CEU)",
        ancestors=["YRI"],
        start_time=T_OOA,
        epochs=[
            dict(start_size=N_CEU_founder, end_time=T_CEU_expand, size_function="constant"),
            dict(start_size=N_CEU_early,   end_size=N_CEU_at_Tgrowth, end_time=T_growth, size_function="exponential"),
            dict(start_size=N_CEU_at_Tgrowth, end_size=N_CEU,   end_time=0, size_function="exponential"),
        ],
    )

    # Higher migration during founder phase
    if m_founder > 0:
        b.add_migration(demes=["YRI", "CEU"], rate=m_founder, start_time=T_OOA, end_time=T_CEU_expand)

    # Lower migration after CEU starts expanding
    if m_modern > 0:
        b.add_migration(demes=["YRI", "CEU"], rate=m_modern, start_time=T_CEU_expand, end_time=0)

    return b.resolve()