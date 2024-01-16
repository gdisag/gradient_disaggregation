# Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix

## Introduction 

We break the secure aggregation protocol of federated learning by showing that individual model updates may be recovered from sums given access to user summary metrics (specifically, participation frequency across training rounds). Our method, gradient disaggregation, observes multiple rounds of summed updates, then leverages summary metrics to recover individual updates. Read our paper here: https://arxiv.org/abs/2106.06089.

<p align="center">
<img src="https://raw.githubusercontent.com/gdisag/gradient_disaggregation/main/images/grad_disaggregated.png" width="300" height="200" >
</p>

## Requirements

```python
pip install scipy numpy
```

```python
python -m pip install -i https://pypi.gurobi.com gurobipy
```

## Quickstart - Reconstructing Participant Matrix P

Using the gradient_disaggregation code is fast and easy. Here we demonstrate disaggregating dummy gradients.

Source the setup script.
```python
source setup.sh
```

Import codebase, generate participant matrix P, synthetic gradients G, aggregated gradients P*G, and participation counts. Then call disaggregate to recover P.
```python
import gradient_disaggregation
import numpy as np

if __name__ == "__main__":

    num_users = 50
    num_rounds = 200
    gradient_size = 200

    # Trying to recover this! 
    G = np.random.uniform(-1, 1, size=(num_users, gradient_size)) 

    # Trying to recover this! Tells which users participated in which rounds.
    P = np.random.choice([0, 1], size=(num_rounds, num_users), p=[.5,.5]) 

    # Given. Observed aggregated updates
    G_agg = P.dot(G) 

    # Given. User summary metrics: for all users, we know how many times they participated across each 10 rounds.
    constraints = gradient_disaggregation.compute_P_constraints(P, 10) 

    # Disaggregate
    P_star = gradient_disaggregation.reconstruct_participant_matrix(G_agg, constraints, verbose=True, multiprocess=True)

    diff = np.sum(np.abs(P_star-P))
    if diff == 0:
       print("Exactly recovered P!")
    else:
       print("Failed to recover P!")
```

## Reconstructing Aggregated FedAvg Updates

Gradient disaggregation can disaggregate noisy aggregated model updates (e.g: FedAvg) -- see fedavg_test.py. 

## Cite

```
@inproceedings{lam2021gradient,
  title={Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix},
  author={Lam, Maximilian and Wei, Gu-Yeon and Brooks, David and Reddi, Vijay Janapa and Mitzenmacher, Michael},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```

