import pickle

with open('ct_dma/ct_dma_valid.pkl', 'rb') as handle:
    ct = pickle.load(handle)

with open('ais_downloads/valid.pkl', 'rb') as handle:
    ais = pickle.load(handle)


x = 1