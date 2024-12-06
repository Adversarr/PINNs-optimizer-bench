import torch
from torch.func import grad, jacrev, hessian
from torch import vmap


def jacobian(y, x):
    """Assume the first dimension is the batch dimension, and perform vmap

    Args:
        y ():
        x ():
    """

    def get_vjp(v):  # [N] -> [B, N]
        return torch.autograd.grad(
            y, x, v.repeat(y.shape[0], 1), create_graph=True, retain_graph=True
        )[0]

    I_N = torch.eye(y.shape[1], device=y.device)
    return vmap(get_vjp)(I_N).transpose(0, 1)


def vjac(func):
    """Return a new function that computes the jacobian of the input function

    Args:
        func (): the model or function to compute the jacobian

    Returns:
        a vmapped jacobian function
    """
    return vmap(jacrev(func), in_dims=0, out_dims=0)


def vhes(func):
    """Return a new function that computes the hessian of the input function

    Args:
        func (): the model or function to compute the hessian

    Returns:
        a vmapped hessian function
    """
    return vmap(jacrev(jacrev(func)), in_dims=0, out_dims=0)


if __name__ == "__main__":
    x = torch.arange(0, 50, 1, dtype=torch.float32).reshape(10, 5).clone()
    x.requires_grad = True

    def func(x):
        sumed = x.sum(dim=-1, keepdim=True)
        return torch.concat([sumed, x])

    y = vmap(func)(x)
    J = jacobian(y, x)
    print(J.shape)  # should be [10, 5, 5]
    partial_y0_partial_x0 = J[:, 0, 0]
    jac_func = vjac(func)
    J2 = jac_func(x)
    print(J2.shape)  # 10, 5, 5
    print(f"close? {torch.allclose(J, J2)}")

    def sum_dim(x):
        return torch.sum(x)

    # y = vmap(sum_dim)(x).unsqueeze(-1)
    print(y.shape)  # should be [10, 1]

    hess_func = vhes(func)
    H2 = hess_func(x)
    print(H2.shape)  # 10, 5, 5

    hess_func = vhes(torch.sin)
    H2 = hess_func(x)
    print(H2.shape)  # 10, 5, 5
