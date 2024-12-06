import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--field", type=str, default="u")

args = parser.parse_args()

field_name = args.field + "_err"

methods = ["adam", "muon"]
init_lrs = {
    "adam": [8e-3, 4e-3, 2e-3, 1e-3, 5e-4],
    "muon": [8e-2, 4e-2, 2e-2, 1e-2, 5e-3],
}
total_lrs = 5

plt.figure(dpi=300)
for m in methods:
    for lr_idx in range(total_lrs):
        lr = init_lrs[m][lr_idx]
        tab = pd.read_csv(f"results/0_1000_{m}_{lr}_exp_1e-05.csv")
        u_err = tab[field_name].rolling(window=7).mean()
        epoch = tab["epoch"]
        plt.plot(epoch, u_err, "--" if m != "muon" else "-", label=f"{m} lr={lr}")

plt.xlabel("Epoch")
plt.ylabel("Value")

plt.grid()
plt.title(f"Relative $L^2$ error of ${args.field}$")
plt.legend()
plt.xscale("log")
plt.yscale("log")

plt.savefig(f"results/{args.field}_err.png")
