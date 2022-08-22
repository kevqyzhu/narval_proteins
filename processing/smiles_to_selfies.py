import selfies as sf
import os

mol = "all-fragments"

data = os.path.join(os.path.dirname(__file__), f"../data/")

with open(data + f"{mol}.txt", "r") as f:
    text = f.read().split('\n')[:-1]

with open(data + f"{mol}_selfies.txt", "w") as f:
    for item in text:
        selfies = sf.encoder(item)
        if selfies != None:
            f.write("%s\n" % selfies)

