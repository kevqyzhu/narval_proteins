from rdkit import RDLogger
from rdkit import Chem
import rdkit.Chem.Draw as Draw
from PIL import Image
from PyPDF2 import PdfFileMerger
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mol', default="largestfragments")
parser.add_argument('--model_dir', default="largestfragments_outputs_seq_2123_hidden_700_dp_0.3_lr_0.0005")
parser.add_argument('--epoch', type=int, default=89)
args = parser.parse_args()

mol = args.mol
model_dir = args.model_dir
epoch = args.epoch

def model_visualize(mol, model_dir, epoch):

    PATH = os.path.join(os.path.dirname(__file__), f"../{mol}/outputs/{model_dir}/")

    processing = PATH + "processing"

    try:
        os.mkdir(processing)
    except OSError:
        # print("Creation of the directory %s failed" % processing)
        pass
    else:
        # print("Successfully created the directory %s " % processing)
        pass
    
    with open(PATH + f'novel_outputs{epoch}.txt', "r") as f:
        smiles_list = tuple(f.read().split('\n')[:-1])

    mol_list = []
    for smiles in smiles_list[0:100]:
        # print(smiles_list.index(smiles))
        m = Chem.MolFromSmiles(smiles)
        mol_list.append(m)

    merger = PdfFileMerger()

    i = 0
    while i < len(mol_list):
        img=Draw.MolsToGridImage(
            mol_list[i:i+9],molsPerRow=3,subImgSize=(2000,2000))

        img.save(PATH + f'processing/{i}outputs{epoch}.png')
        img = Image.open(PATH + rf'processing/{i}outputs{epoch}.png')
        im = img.convert('RGB')
        im.save(PATH + rf'processing/{i}outputs{epoch}.pdf')
        merger.append(PATH + f'processing/{i}outputs{epoch}.pdf')
        i += 9
    merger.write(PATH + f"result{epoch}.pdf")
    merger.close()

    try:
        shutil.rmtree(processing)
    except OSError as e:
        # print ("Error: %s - %s." % (e.filename, e.strerror))
        pass

model_visualize(mol, model_dir, epoch)