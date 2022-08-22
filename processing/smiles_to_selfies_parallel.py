import multiprocessing
import time
import selfies as sf
import os

mol = "all-fragments"

data = os.path.join(os.path.dirname(__file__), f"../data/")

with open(data + f"{mol}.txt", "r") as f:
    text = f.read().split('\n')[:-1]

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# text = chunks(text, 40)

selfies = []

def mp_worker(smiles):
    
    for smile in smiles:
        selfie = sf.encoder(smiles)
        if selfie is not None:
            selfies.append(sf.encoder(smiles))

def mp_handler():
    p = multiprocessing.Pool(40)
    p.map(mp_worker, text)

if __name__ == '__main__':
    mp_handler()

    with open(data + f"{mol}_selfies_1.txt", "w") as f:
        for item in selfies:
            f.write("%s\n" % item)
