import numpy as np

class OFEvaluation:
    def __init__(self, ofs, gts) -> None:
        
        # Model inference
        self.ofs= ofs

        # Ground Truth
        self.gts= gts
    
    # Average Angular Error
    def AAE(self):
        pass

    # Average Endpoint Error(AEPE)
    def AEPE(self):
        pass

    # Average Magnitude Error
    def AME(self):
        pass

    # Normalized Average Magnitude Error
    def NAME(self):
        pass

    # Error Normal to the Gradient (???)

    # Root-Mean-Square Error
    def RMS(self):
        pass

    # Normalized Interpolation Error
    def NIE(self):
        pass

    def load_data(self, path: str) -> None:
        pass

    def save_result(self):
        pass