import numpy as np
from scipy.optimize import linprog


def expected_rev(S, w, r):
    """
    compute the expected revenue of set S
    """
    w, r = np.array(w), np.array(r)
    return np.sum(w[S]*r[S])/(1+np.sum(w[S]))

def primal_modified(weights, revenues, qualities, constrained_sets, delta):
    """
    Solve the primal problem where only the violated constraints are considered.

    weights: the weight of each item

    constrained_sets: the list of sets that we need to consider for the fairness constraint

    delta: fairness parameter
    """
    n = len(weights)

    constrained_sets = np.array(list(constrained_sets))

    num_sets = len(constrained_sets)

    b_ub = [delta for i in range(n**2)]
    b_ub.append(1)
    b_ub = np.array(b_ub)

    A_ub = None
    for i in range(n):
        for j in range(n):
            ### UPDATED 03/23/2023
            coeff_1 = np.array([1/qualities[i] if i in S else 0 for S in constrained_sets])
            coeff_2 = np.array([-1/qualities[j] if j in S else 0 for S in constrained_sets])
            if i == 0 and j == 0:
                A_ub = coeff_1 + coeff_2
            else:
                A_ub = np.vstack((A_ub, coeff_1 + coeff_2))

    A_ub = np.vstack((A_ub, np.ones(num_sets)))

    c = [-expected_rev(list(S), weights, revenues) for S in constrained_sets]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub)

    primal_sol = res.x
    primal_obj = -res.fun

    sub_idx = np.where(primal_sol > 0.001)[0]
    primal_sol_sub = primal_sol[sub_idx]
    sets_to_consider = constrained_sets[sub_idx]

    return primal_obj, primal_sol_sub, sets_to_consider
