import math
from argparse import ArgumentParser

import torch
import numpy as np
from muon import Muon
from torch import optim
from torch.optim import lr_scheduler as sched
from torch.nn import functional as F
from tqdm import trange


import matplotlib.pyplot as plt
from diffop import vhes, vjac
from model import PINN
import pandas

from torch_optimizer import Shampoo

RE = 20
NU = 1 / RE
L = 1 / (2 * NU) - np.sqrt(1 / (4 * NU**2) + 4 * np.pi**2)


def rel_mse(x, y):
    return F.mse_loss(x, y) / torch.mean(torch.square(y))


def rel2(gt, pred):
    return (torch.linalg.norm(gt - pred) / torch.linalg.norm(gt)).item()


def u_func(x):
    return 1 - (torch.exp(L * x[:, 0]) * torch.cos(2 * np.pi * x[:, 1]))


def v_func(x):
    return L / (2 * np.pi) * torch.exp(L * x[:, 0]) * torch.sin(2 * np.pi * x[:, 1])


def p_func(x):
    return 0.5 * (1 - torch.exp(2 * L * x[:, 0]))


def create_scheduler(optimizer, args):
    name = str(args.sched).lower()
    epochs = args.epochs
    min_lr = args.min_lr
    if name == "step":
        return sched.StepLR(optimizer, step_size=epochs, gamma=0.5)
    elif name == "cosine":
        return sched.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif name == "exp":
        gamma = (min_lr / args.init_lr) ** (1 / epochs)
        return sched.ExponentialLR(optimizer, gamma)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def create_optimizer(pinn: PINN, args):
    name = str(args.optim).lower()
    adam_betas = (args.adam_beta1, args.adam_beta2)
    if name == "adam":
        return optim.Adam(pinn.parameters(), lr=args.init_lr, betas=adam_betas)
    elif name == "muon":
        return Muon(
            muon_params=pinn.muon_params(),
            lr=args.init_lr,
            momentum=args.momentum,
            nesterov=True,
            ns_steps=6,
            adamw_params=pinn.adamw_params(),
            adamw_lr=args.adamw_lr,
            adamw_betas=adam_betas,
            adamw_wd=0,
        )
    elif name == "shampoo":
        return Shampoo(
            params=pinn.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def pinn_loss(x_train, pinn, jacobian, hessian):
    uvp = pinn(x_train)
    uvp_xy = jacobian(x_train)
    uvp_xy_xy = hessian(x_train)

    u = uvp[:, 0]
    v = uvp[:, 1]

    u_x = uvp_xy[:, 0, 0]
    u_y = uvp_xy[:, 0, 1]
    v_x = uvp_xy[:, 1, 0]
    v_y = uvp_xy[:, 1, 1]
    p_x = uvp_xy[:, 2, 0]
    p_y = uvp_xy[:, 2, 1]

    u_xx = uvp_xy_xy[:, 0, 0, 0]
    u_yy = uvp_xy_xy[:, 0, 1, 1]
    v_xx = uvp_xy_xy[:, 1, 0, 0]
    v_yy = uvp_xy_xy[:, 1, 1, 1]

    convective_x = u * u_x + v * u_y
    convective_y = u * v_x + v * v_y

    # cavity flow equation:
    # 1. momentum x
    loss_mom_x = F.mse_loss((p_x + convective_x), (u_xx + u_yy) / RE)

    # 2. momentum y
    loss_mom_y = F.mse_loss((p_y + convective_y), (v_xx + v_yy) / RE)

    # 3. continuity
    loss_cont = F.mse_loss(u_x, -v_y)

    return loss_mom_x + loss_mom_y + loss_cont


def run(args):
    # Inputs are [x, y], Outputs are [u, v, p]
    print(args)
    torch.manual_seed(args.seed)
    pinn = PINN(in_channels=2, hid_channels=64, out_channels=3, num_hiddens=4)
    pinn.reset_parameters()
    pinn.compile()
    jacobian = vjac(pinn)
    hessian = vhes(pinn)

    optimizer = create_optimizer(pinn, args)
    scheduler = create_scheduler(optimizer, args) if args.sched else None

    ### Boundary data
    x = torch.linspace(0, 1, 128)
    y = torch.linspace(0, 1, 128)
    left = torch.stack([torch.zeros(128), y], dim=1)
    right = torch.stack([torch.ones(128), y], dim=1)
    up = torch.stack([x, torch.ones(128)], dim=1)
    lo = torch.stack([x, torch.zeros(128)], dim=1)
    bd_xy = torch.cat([left, right, up, lo], dim=0)
    bd_gt = torch.stack([u_func(bd_xy), v_func(bd_xy), p_func(bd_xy)], dim=1)

    ### Test data, the ground truth solution
    x_true, y_true = torch.meshgrid(x, y)
    test_xy = torch.stack([x_true.flatten(), y_true.flatten()], dim=1)
    test_gt = torch.stack([u_func(test_xy), v_func(test_xy), p_func(test_xy)], dim=1)

    plt.figure(dpi=200, figsize=(10, 4))
    losses = []
    u_errs = []
    v_errs = []
    p_errs = []
    current_lr = optimizer.param_groups[0]["lr"]

    def plot_once(i, current_lr):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title("Relative $L^2$ Error")
        plt.loglog(u_errs, label="U")
        plt.loglog(v_errs, label="V")
        plt.loglog(p_errs, label="P")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.title(f"Epoch {i}")
        plt.loglog(losses, label="Total Loss")
        plt.grid()
        plt.legend()

    rg = trange(args.epochs)
    ### Training Main Loop.
    for i in rg:
        optimizer.zero_grad()
        x_train = torch.rand(1024, 2)

        def closure():
            loss_equation = pinn_loss(x_train, pinn, jacobian, hessian)
            uvp_bd = pinn(bd_xy)
            loss_bd = (
                rel_mse(uvp_bd[:, 0], bd_gt[:, 0])
                + rel_mse(uvp_bd[:, 1], bd_gt[:, 1])
                + rel_mse(uvp_bd[:, 2], bd_gt[:, 2])
            )
            total = loss_equation + loss_bd
            total.backward()
            return total

        total = optimizer.step(closure)
        assert isinstance(total, torch.Tensor)
        if scheduler:
            scheduler.step()

        with torch.no_grad():
            uvp = pinn(test_xy)
            u_err = rel2(test_gt[:, 0], uvp[:, 0])
            v_err = rel2(test_gt[:, 1], uvp[:, 1])
            p_err = rel2(test_gt[:, 2], uvp[:, 2])
            u_errs.append(u_err)
            v_errs.append(v_err)
            p_errs.append(p_err)
            losses.append(total.item())
            current_lr = optimizer.param_groups[0]["lr"]
            rg.set_postfix(
                u_err=u_err,
                v_err=v_err,
                p_err=p_err,
                lr=current_lr,
            )

            if args.plot:
                plot_once(i, current_lr)
                plt.pause(0.01)

    rg.close()
    # Export data:
    plot_once(args.epochs, current_lr)
    fn = f"results/{args.seed}_{args.epochs}_{args.optim}_{args.init_lr}_{args.sched}_{args.min_lr}"
    plt.savefig(fn + ".png")
    table = pandas.DataFrame(
        {
            "u_err": u_errs,
            "v_err": v_errs,
            "p_err": p_errs,
            "loss": losses,
        },
        index=range(args.epochs),
    )
    table.index.name = "epoch"
    table.to_csv(fn + ".csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--init-lr", type=float, default=1.0e-3)
    parser.add_argument("--adamw-lr", type=float, default=3.0e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--sched", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")

    torch.set_float32_matmul_precision("high")
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True

    run(parser.parse_args())
