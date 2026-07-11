"""Real-data adapters for the robust / sandwich bandit experiment.

The robust-sandwich experiment is strongest when the heavy-tailed reward — and
hence the model misspecification that makes A_t = L^{-1}M L^{-⊤} ≠ I — comes from
*real data* rather than injected noise. This module casts real datasets into the
contextual-bandit form the experiment needs:

    contexts C ∈ R^{N×d},   rewards r ∈ R^{N}  (heavy-tailed),

so an off-data replay can present rows as candidate arms and observe their real
reward.

Runnable here: ``load_charges`` fetches a small real dataset with a right-skewed
reward (medical charges: skew ≈ 1.5) and real covariates, cached locally.

For your infrastructure (gated / large), ``load_kuairand`` and ``load_ipinyou``
document the exact bundling {contexts + propensities + heavy-tailed reward} and
return the same (contexts, rewards[, propensities]) interface:

  * KuaiRand — watch-time reward (heavy right skew) with a uniform-random logging
    slice (~0.37% of interactions) that gives known propensities for unbiased
    offline evaluation; user/item features as contexts.
  * iPinYou — RTB market/winning price (heavy-tailed) with auction/ad context;
    you supply the arm/propensity structure.

Nothing here is committed as data; files are fetched to a gitignored cache.
"""

from __future__ import annotations

import io
import pathlib
import urllib.request

import torch

_CACHE = pathlib.Path(__file__).resolve().parent.parent / "results" / "data_cache"
_CHARGES_URL = ("https://raw.githubusercontent.com/stedy/"
                "Machine-Learning-with-R-datasets/master/insurance.csv")


def _fetch_text(url, cache_name):
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / cache_name
    if path.exists():
        return path.read_text()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:  # network / proxy allowlist
        raise RuntimeError(
            f"Could not fetch {url} ({exc}). Download it manually to {path} "
            f"and re-run.") from exc
    path.write_text(text)
    return text


def _standardize(x):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


def load_charges(dtype=torch.float64, device=None, log_reward=False):
    """Real heavy-tailed dataset: medical charges (N≈1338, right-skewed reward).

    Contexts: age, bmi, children (standardized) + one-hot sex/smoker/region and a
    bias. Reward: `charges`, standardized (raw skew ≈ 1.5). With ``log_reward``,
    log-charges (a lighter-tailed control on the same data).

    Returns (contexts (N,d), rewards (N,), feature_names).
    """
    import csv
    text = _fetch_text(_CHARGES_URL, "insurance.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    device = device or torch.device("cpu")

    age = torch.tensor([float(r["age"]) for r in rows], dtype=dtype)
    bmi = torch.tensor([float(r["bmi"]) for r in rows], dtype=dtype)
    kids = torch.tensor([float(r["children"]) for r in rows], dtype=dtype)
    smoker = torch.tensor([1.0 if r["smoker"] == "yes" else 0.0 for r in rows], dtype=dtype)
    male = torch.tensor([1.0 if r["sex"] == "male" else 0.0 for r in rows], dtype=dtype)
    regions = ["southwest", "southeast", "northwest", "northeast"]
    reg = torch.stack([torch.tensor([1.0 if r["region"] == g else 0.0 for r in rows],
                                    dtype=dtype) for g in regions[:-1]], dim=1)  # drop one
    charges = torch.tensor([float(r["charges"]) for r in rows], dtype=dtype)

    cont = _standardize(torch.stack([age, bmi, kids], dim=1))
    bias = torch.ones(len(rows), 1, dtype=dtype)
    C = torch.cat([bias, cont, smoker[:, None], male[:, None], reg], dim=1).to(device)
    names = (["bias", "age", "bmi", "children", "smoker", "male"]
             + [f"region_{g}" for g in regions[:-1]])

    r = torch.log(charges) if log_reward else charges
    r = ((r - r.mean()) / r.std().clamp_min(1e-6)).to(device)
    return C, r, names


def load_kuairand(path, dtype=torch.float64, device=None):
    """KuaiRand adapter (bring your own download).

    Expects a parquet/csv with per-interaction rows including user/item feature
    columns, the uniform-random-exposure flag, the logging propensity, and a
    watch-time (play-time) column used as the heavy-tailed reward. Fill in the
    column names for your KuaiRand variant (Pure/1K/27K) below.

    Returns (contexts, rewards, propensities).
    """
    raise NotImplementedError(
        "KuaiRand is gated/large; download the variant you want (kuairand.com) to "
        f"{path}, then map: contexts = user/item feature cols, reward = watch/play "
        "time (standardized), propensity = logging prob on the random-exposure slice. "
        "This function is the documented hook — same (contexts, rewards, propensities) "
        "interface as load_charges.")


def load_ipinyou(path, dtype=torch.float64, device=None):
    """iPinYou RTB adapter (bring your own download).

    Expects the season logs with auction/ad context columns and the market/winning
    price column (the heavy-tailed reward/cost); you supply the arm/propensity
    structure since iPinYou is framed as bidding, not a logged bandit.
    """
    raise NotImplementedError(
        "iPinYou is ~35GB; download a season to "
        f"{path}, then map: contexts = auction/ad/user feature cols, reward = "
        "market/winning price (log-standardized), and define arms = candidate bids. "
        "Same interface as load_charges.")
