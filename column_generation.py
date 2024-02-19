import numpy as np
import cvxpy as cp
from math import *

from approx_alg import half_approx, FPTAS
from primal import expected_rev
from staticMNL import staticMNL

time_alg = 0
call_alg = 0

def column_generation_optimize(weights, revenues, qualities, n, K, delta, current_sets=None, \
                               eps=0.1, num_iter=10**3, alg="half", verbose=False):
    """
    Optimize the problem using the column generation approach.
    ------
    Initialization: we start from the sets of size one.

    At each round,
        1) Solve the primal problem with current nonzero sets.
        2) we check whether the dual constraint is violated by the current dual variable.
        - If found set $S$ with the highest Rev-cost (i.e., the set the violates the constraint the most)
          We then add set $S$ to the primal problem. Go back to start of the loop.
        - If not found, the current primal solution is optimal.

    """
    global time_alg
    time_alg = 0

    global call_alg
    call_alg = 0

    maxRev, optSet, _ = staticMNL(n, K, revenues, np.insert(weights, 0, 1))
    print ("max revenue without fairness constraint = {}".format(maxRev))

    ### Initialization ###
    current_sets = [frozenset({i}) for i in range(n)]
    num_sets = len(current_sets)
    primal_obj = np.inf
    primal_sol = None
    dual_sol = None

    iter_used = 0

    # Initialize A, b and c
    A = np.ones(n)
    b = np.array([1] + [delta * qualities[i] * qualities[j] for i in range(n) for j in range(n)])

    for i in range(n):
        for j in range(n):
            coeff_1 = np.array([qualities[j] if i in S else 0 for S in current_sets])
            coeff_2 = np.array([-qualities[i] if j in S else 0 for S in current_sets])
            A = np.vstack((A, coeff_1 + coeff_2))

    c = np.array([expected_rev(list(S), weights, revenues) for S in current_sets])

    while iter_used <= num_iter:

        if iter_used % 10 == 0: print("Number of iteration = {}".format(iter_used))

        # Define and solve the CVXPY problem.
        x = cp.Variable(num_sets)
        prob = cp.Problem(cp.Maximize(c.T@x), [A @ x <= b, x >= 0])
        prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})

        ### If the amount of improvement is very small, return the current solution ###
        if iter_used != 0 and np.abs(prob.value - primal_obj)/primal_obj < 10**(-6):
            print("Primal obj = {}".format(primal_obj))
            print("The amount of improvement is very small.")
            return prob.value, x.value, current_sets, time_alg, call_alg

        # Update the variables
        primal_obj = prob.value
        primal_sol = x.value
        dual_sol = prob.constraints[0].dual_value
        print("Primal obj = {}".format(primal_obj))

        rho = dual_sol[0]
        z = dual_sol[1:].reshape((n,n))

        # Given the dual variables, let us find the set S that maximizes the Rev-Cost
        costs = np.array([np.sum([(z[i,j] - z[j,i])*qualities[j] for j in range(n) if j != i]) for i in range(n)])

        # Use the approx algorithm to find the near-optimal S
        if alg == "half":
            S_eps, max_objective, _, time_taken = half_approx(weights, revenues, costs, K)
        elif alg == "FPTAS":
            S_eps, max_objective, time_taken = FPTAS(weights, revenues, costs, K, eps=eps)
        else:
            raise ValueError("This is not a valid approx algorithm.")

        time_alg += time_taken
        call_alg += 1

        # If we find a set S that violates the dual constraint maximally, add it to current_sets
        # If all sets satisfy the violated constraints, then we have solved Problem FAIR (near-)optimally
        if max_objective > rho:
            # First, solve the primal problem with current nonzero sets
            current_sets.append(frozenset(S_eps))
            num_sets = len(current_sets)

            # Update A and c
            coeff_1 = np.array([qualities[j] if i in S_eps else 0 for i in range(n) for j in range(n)])
            coeff_2 = np.array([-qualities[i] if j in S_eps else 0 for i in range(n) for j in range(n)])

            A = np.c_[ A, np.insert(coeff_1 + coeff_2, 0, 1, axis=0) ]
            c = np.append(c, expected_rev(list(S_eps), weights, revenues))

        else:
            print ("We have found an optimal solution.")
            return primal_obj, primal_sol, current_sets, time_alg, call_alg

        iter_used += 1

    print ("Maximum number of iterations exceeded.")
    return primal_obj, primal_sol, current_sets[:-1], time_alg, call_alg
