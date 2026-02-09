from typing import Optional, List, Tuple
import torch
import math

# ------------------------------------------------------
# helper functions for reweighting and interval changing
# not used (as 12/29)
# -------------------------------------------------------


def get_weights(B: int, device: torch.device, num_mask: torch.Tensor, l_eff: torch.Tensor, reweighting: Optional[str] = None) -> torch.Tensor:
    """
    compute the weights for the reweighting strategy
    normalize each weight to be integreated to 1.0
    """
    if reweighting == "none" or reweighting is None:
        return torch.ones(B, 1, device=device)
    elif reweighting == "sin":
        ratio = num_mask / l_eff.clamp(min=1.0)
        norm = (1 + 0.5 / math.pi)
        return (1.0 + 0.25 * torch.sin(ratio * math.pi)) / norm
    elif reweighting == "quadratic":
        ratio = num_mask / l_eff.clamp(min=1.0)
        norm = 7.0 / 6.0
        return (1.0 + ratio * (1.0 - ratio)) / norm
    elif reweighting == "soft_linear":
        ratio = num_mask / l_eff.clamp(min=1.0)
        norm = 1.5
        return (1.0 + ratio) / norm
    else:
        assert False, "invalid reweighting strategy"

def build_intervals_uneven_hard_coded(K: int = 6) -> List[Tuple[float, float]]:
    """
    Build uneven K ratio intervals over [0, 1] so each interval has equal total
    mass under the same shape:
        weight(mask_ratio) = mask_ratio / norm,
        where mask_ratio = 1 - unmasked_ratio
    formulation:
    """
    assert K == 6, "K must be 6 for hard-coded intervals"
    return [(0.0, 0.1), (0.1, 0.4), (0.4, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]


def build_intervals_uneven(K: int, norm: float = 1.5) -> List[Tuple[float, float]]:
    """
    Build uneven K ratio intervals over [0, 1] so each interval has equal total
    mass under the same shape:
        weight(mask_ratio) = mask_ratio / norm,
        where mask_ratio = 1 - unmasked_ratio
    formulation:
        b_j = 1 - sqrt(1 - j/K)
    """
    if K <= 0:
        return []

    boundaries = [0.0]
    for j in range(1, K):
        frac = j / K
        # numerical safety for sqrt
        boundaries.append(1.0 - math.sqrt(max(0.0, 1.0 - frac)))
    boundaries.append(1.0)

    return [(boundaries[j], boundaries[j + 1]) for j in range(K)]