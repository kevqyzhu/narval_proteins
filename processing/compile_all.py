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

def get_metrics(mol):
    best_models_outputs = os.path.join(os.path.dirname(__file__), f"../../../../all_compiled/selfies_rnn/{mol}/outputs")
    best_models_metrics = os.path.join(os.path.dirname(__file__), f"../../../../all_compiled/selfies_rnn/{mol}/metrics")

    os.makedirs(best_models_metrics, exist_ok=True)
    os.makedirs(best_models_outputs, exist_ok=True)

    metrics = os.path.join(os.path.dirname(__file__), f"../{mol}/outputs")

    regex = re.compile(r"_(\d+[.,]*\d*[.,]*\d*)")

    x_plot = []
    y_plot = []
    score_plot = []

    for root, dirs, files in os.walk(metrics):
        # print(f"Root: {root}")
        for file in files:
            if "benchmark" in file:
                # print(file)
                dir_name = os.path.basename(root)
                params_search = dir_name.strip(mol)
                params = [x for x in regex.findall(params_search)]
                params.extend(re.findall('\d+[.,]*\d*[.,]*\d*', file))
                # print(params)

                metrics = os.path.join(root, file)

                n_hidden = int(params[1])
                drop_prob = float(params[2])
                lr = float(params[3])
                if len(params) >= 6:
                    epoch = int(float(params[5]))
                else:
                    epoch = int(float(params[4]))

                scores = get_unique_valid(metrics)
                score = scores[0] * scores[1] * scores[2]
            
                x_param = n_hidden
                y_param = epoch

                x_plot.append(x_param)
                y_plot.append(y_param)
                score_plot.append(score)

                outputs = None
                if score >= 0.6:
                    for file in files:
                        if file == f"outputs{epoch}.txt":
                            outputs = os.path.join(root, file)
                    print(score)
                    with open(os.path.join(best_models_metrics, f"hidden_{n_hidden}_dp_{drop_prob}_lr_{lr}_epoch_{epoch}_metrics.txt"), "w") as f:
                                    f.write(f"NUM HIDDEN: {n_hidden} \n")
                                    f.write(f"DROP PROB: {drop_prob} \n")
                                    f.write(f"LR: {lr} \n")
                                    f.write(f"EPOCH: {epoch} \n")
                                    # f.write(f"UNIQUE AND VALID: {score} \n")
                                    f.write(f"VALID: {scores[0]} \n")
                                    f.write(f"UNIQUE: {scores[1]} \n")
                                    f.write(f"NOVEL: {scores[2]} \n")
                                    f.write(f"VALID, UNIQUE, AND NOVEL: {score} \n")
                                    f.write("\n")
                    src = outputs
                    dst_file = os.path.join(best_models_outputs, f"hidden_{n_hidden}_dp_{drop_prob}_lr_{lr}_epoch_{epoch}_outputs.txt")
                    copyfile(src, dst_file)


    
datasets = ["largesmiles", "hmd", "newlogp"]

for mol in datasets:
    print(mol)
    get_metrics(mol)