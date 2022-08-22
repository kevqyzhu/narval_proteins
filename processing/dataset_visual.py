from rdkit import Chem
import random
import rdkit.Chem.Draw as Draw
from PIL import Image
from PyPDF2 import PdfFileMerger
import os
import shutil

mol = "largestfragments"

data = os.path.join(os.path.dirname(__file__), f"../../data/")

PATH = os.path.join(os.path.dirname(__file__), f'../{mol}/')

processing = PATH + "processing"

try:
    os.mkdir(processing)
except OSError:
    print ("Creation of the directory %s failed" % processing)
else:
    print ("Successfully created the directory %s " % processing)

with open(f"{data}" + f"{mol}.txt", "r") as f:
    smiles_list = tuple(f.read().split('\n')[:-1])


mol_list = []
smiles_list = random.sample(smiles_list, 30)

for smiles in smiles_list:
    print(smiles_list.index(smiles))
    m = Chem.MolFromSmiles(smiles)

    # num = m.GetNumHeavyAtoms()
    # if 400 <= num <= 500:
    #     mol_list.append(m)
    #     print(smiles)

    mol_list.append(m)

merger = PdfFileMerger()

i = 0
while i < len(mol_list):
    img=Draw.MolsToGridImage(
        # mol_list[i:i+25],molsPerRow=5,subImgSize=(500,500))
        mol_list[i:i+15],molsPerRow=3,subImgSize=(2000,2000))

    img.save(PATH + f'processing/{i}outputs.png')
    img = Image.open(PATH + rf'processing/{i}outputs.png')
    im = img.convert('RGB')
    im.save(PATH + rf'processing/{i}outputs.pdf')
    merger.append(PATH + f'processing/{i}outputs.pdf')
    # i += 25
    i += 15
merger.write(PATH + f"result.pdf")
merger.close()

try:
    shutil.rmtree(processing)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))
