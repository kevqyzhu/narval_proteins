import os
import re
# import json
# import matplotlib.pyplot as plt
# from matplotlib import cm


def get_unique_valid(path):
    with open(path, "r") as f:
        results = f.read()
    results = re.findall('\d+[.,]*\d*[.,]*\d*', results)
    # return float(results[2]) * float(results[4]) * float(results[6])
    return [float(results[2]), float(results[4]), float(results[6])]
    # for bench in data['results']:
    #     print(bench['score'])


mol = "largefragments_1000-2000"
metrics = os.path.join(os.path.dirname(__file__), f"../{mol}/outputs")

with open(os.path.join(os.path.dirname(__file__),
                       f"../{mol}/selfies_metrics_compiled.txt"), "w") as f:
    pass

regex = re.compile(r"_(\d+[.,]*\d*[.,]*\d*)")

best_score = 0
best_n_hidden = 0
best_drop_prob = 0
best_lr = 0
best_epoch = 0

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

            n_hidden = int(params[1])
            drop_prob = float(params[2])
            lr = float(params[3])
            if len(params) >= 6:
                epoch = int(float(params[5]))
            else:
                epoch = int(float(params[4]))

            scores = get_unique_valid(metrics)
            score = scores[0] * scores[1] * scores[2]

            with open(os.path.join(os.path.dirname(__file__),
                                   f"../{mol}/selfies_metrics_compiled.txt"), "a") as f:
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

            x_param = n_hidden
            y_param = epoch

            x_plot.append(x_param)
            y_plot.append(y_param)
            score_plot.append(score)

            if best_score <= score:
                best_score = score
                best_n_hidden = n_hidden
                best_drop_prob = drop_prob
                best_lr = lr
                best_epoch = epoch

print(best_score)
print(best_n_hidden)
print(best_drop_prob)
print(best_lr)
print(best_epoch)


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



