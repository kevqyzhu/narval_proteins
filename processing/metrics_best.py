import os
import re
# import json
# import matplotlib.pyplot as plt
# from matplotlib import cm
from shutil import copyfile



def get_unique_valid(path):
    with open(path, "r") as f:
        results = f.read()
    results = re.findall('\d+[.,]*\d*[.,]*\d*', results)
    # print(results)
    # return float(results[2]) * float(results[4]) * float(results[6])
    # print(results)
    if len(results) >= 7:
        return [float(results[2]), float(results[4]), float(results[6])]
    else:
        return [0, 0, 0]
    # for bench in data['results']:
    #     print(bench['score'])

def get_num(path):
    with open(path, "r") as f:
        results = f.read()
    results = re.findall('\d+[.,]*\d*[.,]*\d*', results)
    return int(float(results[0])/1000)

def get_best_metrics(mol, model_type):
    metrics = os.path.join(os.path.dirname(__file__), f"../{model_type}/{mol}/outputs")

    regex = re.compile(r"_(\d+[.,]*\d*[.,]*\d*)")

    best_score = 0
    best_scores = None
    best_n_layers = 0
    best_n_hidden = 0
    best_drop_prob = 0
    best_lr = 0
    best_epoch = 0
    best_outputs = None
    best_num = 0

    x_plot = []
    y_plot = []
    score_plot = []

    for root, dirs, files in os.walk(metrics):
        # print(f"Root: {root}")
        for file in files:
            if "benchmark" in file:
                dir_name = os.path.basename(root)
                params_search = dir_name.strip(mol)
                params = [x for x in regex.findall(params_search)]
                params.extend(re.findall('\d+[.,]*\d*[.,]*\d*', file))

                metrics = os.path.join(root, file)
                # print(params)

                n_layers = int(params[1])
                n_hidden = int(params[2])
                drop_prob = float(params[3])
                lr = float(params[4])
                epoch = int(params[6])

                # ONLY FOR TO TOGGLE FOR MODEL SIZE
                if n_layers != 3 or n_hidden != 700:
                    continue

                scores = get_unique_valid(metrics)
                score = scores[0] * scores[1] * scores[2]

                x_param = n_hidden
                y_param = epoch

                x_plot.append(x_param)
                y_plot.append(y_param)
                score_plot.append(score)
                outputs = None
                num = get_num(metrics)
                if num == 0:
                    continue

                if best_score <= score:
                    best_scores = scores
                    best_score = score
                    best_n_layers = n_layers
                    best_n_hidden = n_hidden
                    best_drop_prob = drop_prob
                    best_lr = lr
                    best_epoch = epoch
                    for file in files:
                        if file == f"novel_outputs{epoch}.txt":
                            outputs = os.path.join(root, file)
                    if outputs is not None:
                        best_outputs = outputs
                    # print(best_outputs)
                    best_num = num
    # with open(os.path.join(os.path.dirname(__file__), f"../../../../best_models/metrics.txt"), "r") as f:
    #     metrics = f.read()

    best_models = os.path.join(os.path.dirname(__file__), f"../../best_models")
    with open(os.path.join(best_models, f"{model_type}-{mol}_large-{best_num}ksmiles_metrics.txt"), "w") as f:
                    f.write(f"NUM LAYERS: {best_n_layers} \n")  
                    f.write(f"NUM HIDDEN: {best_n_hidden} \n")
                    f.write(f"DROP PROB: {best_drop_prob} \n")
                    f.write(f"LR: {best_lr} \n")
                    f.write(f"EPOCH: {best_epoch} \n")
                    # f.write(f"UNIQUE AND VALID: {score} \n")
                    f.write(f"VALID: {best_scores[0]} \n")
                    f.write(f"UNIQUE: {best_scores[1]} \n")
                    f.write(f"NOVEL: {best_scores[2]} \n")
                    f.write(f"VALID, UNIQUE, AND NOVEL: {best_score} \n")
                    f.write("\n")

    src = best_outputs
    # print(src)
    dst_file = os.path.join(best_models, f"{model_type}-{mol}_large-{best_num}ksmiles.txt")
    copyfile(src, dst_file)

# datasets = ["cep", "gdb13", "largefragments", "largefragments_500-1000", "largefragments_1000-2000", "largefragments_2000-5000", "ligands", "logp", "moses", "oled", "polymers", "qm9", "sa", "unique94qed", "zinc"]
datasets = ["allab60"]

for mol in datasets:
    print(mol)
    get_best_metrics(mol, "smiles_RNN")



# os.rename(dst_file, )


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x_plot, y_plot, fd_plot, cmap=cm.get_cmap("jet"), linewidth=0)
# fig.tight_layout()

# plt.hist(score_plot)
#
# plt.show()

# hidden nodes: more the merrier (currently training 900)
# drop prob: doesn't seem to matter much
# lr: 0.0001 - 0.0005 seems to be an ideal range

# epochs: more the merrier



