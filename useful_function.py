import numpy as np 

def get_sr(pnl):
    return np.mean(pnl) / np.std(pnl) * np.sqrt(252)