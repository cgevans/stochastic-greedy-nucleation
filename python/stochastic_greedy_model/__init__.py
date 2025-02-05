from dataclasses import dataclass

@dataclass
class KTAMParams:
    """Parameters for kTAM simulations
    
    Args:
        gmc: Monomer concentration parameter
        gse: Bond energy parameter
        alpha: kTAM alpha parameter
        kf: Forward rate constant
    """
    gmc: float
    gse: float
    alpha: float
    kf: float

from .stochastic_greedy_model import *