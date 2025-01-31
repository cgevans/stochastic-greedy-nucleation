class KTAMParams:
    """Parameters for kTAM simulations"""
    def __init__(self, gmc: float, gse: float, alpha: float, kf: float):
        """
        Initialize kTAM parameters
        
        Args:
            gmc: Monomer concentration parameter
            gse: Stacking energy parameter
            alpha: Nucleation penalty parameter
            kf: Forward rate constant
        """
        self.gmc = gmc
        self.gse = gse
        self.alpha = alpha
        self.kf = kf
