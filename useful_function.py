import numpy as np 


def get_pne(signal, returns):
    return (signal * returns).sum(axis=1)


def get_sr(pnl):
    return np.mean(pnl) / np.std(pnl) * np.sqrt(252)


