# PINNs optimizer benchmark

Here we consider a standard Kovasznay flow problem with $\mathrm{Re}=20$ on unit square. The architecture of PINNs is FLS with `Stan` activation.
To guarantee stability, exponential decay learning rate schedule is applied, and the final-lr is set to `1e-5`

To run:

```sh
pip install -r requirements.txt
sh run_exp.sh
```

> The Muon optimizer in use is a (slightly) modified verision [here](https://github.com/Adversarr/Muon)

## Major Results

Compared with Adam optimizer, the proposed [Muon](https://github.com/KellerJordan/Muon) optimizer converges obviously faster & converge to a better solution, see below.

![u](./results/u_err.png)

![v](./results/v_err.png)

![p](./results/p_err.png)

## More Results

Adam 8e-3:

![adam, 2e-3](results/0_1000_adam_0.008_exp_1e-05.png)

Adam 4e-3:

![adam, 2e-3](results/0_1000_adam_0.004_exp_1e-05.png)

Adam 2e-3:

![adam, 2e-3](results/0_1000_adam_0.002_exp_1e-05.png)

Adam 1e-3:

![adam, 1e-3](results/0_1000_adam_0.001_exp_1e-05.png)

Adam 5e-4:

![adam, 1e-3](results/0_1000_adam_0.0005_exp_1e-05.png)


Muon 8e-2:

![muon 4e-2](results/0_1000_muon_0.08_exp_1e-05.png)

Muon 4e-2:

![muon 4e-2](results/0_1000_muon_0.04_exp_1e-05.png)

Muon 2e-2:

![muon 4e-2](results/0_1000_muon_0.02_exp_1e-05.png)

Muon 1e-2:

![muon 4e-2](results/0_1000_muon_0.01_exp_1e-05.png)


Muon 5e-3:

![muon 4e-2](results/0_1000_muon_0.005_exp_1e-05.png)