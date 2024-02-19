import numpy as np
import time
import sys
import pickle
from math import *

from column_generation import column_generation_optimize

###################### Define Grids #######################

# alg = "half"
alg = "FPTAS"

range_random_seeds = range(50)
range_delta = np.arange(0, 1.2, 0.2)
param_grids = [(seed, delta) for seed in range_random_seeds for delta in range_delta]

# NOTE: Need to input the grid_idx from 1-300
grid_idx = int(sys.argv[1]) - 1
seed, delta = param_grids[grid_idx]

num_items = 10
K = 5
eps = 0.1

np.random.seed(seed)
print ("running experiment with seed {}".format(seed))

# generate an instance where w ~ unif(0.5, 1.5)
beta = -1
r = np.random.random(num_items)
intercept = np.random.random(num_items) * 0.5
w = np.exp(beta * r + intercept)
q = w

print ("delta = {}".format(delta))

time_start = time.time()

primal_obj, primal_sol, current_sets, time_alg, call_alg = \
                                        column_generation_optimize(w, r, q, num_items, K, delta, \
                                                                   current_sets=None, \
                                                                   eps=eps, \
                                                                   num_iter=10**2, \
                                                                   alg=alg, \
                                                                   verbose=False)

total_time = time.time()-time_start

print ("Total time spent = {}".format(total_time))
print ("Time spent in alg = {}".format(time_alg))
print ("Number of time we call alg = {}".format(call_alg))
print ("Average time spent in alg = {}".format(time_alg/call_alg))

sub_idx = np.where(primal_sol > 0.001)[0]
primal_sol_sub = primal_sol[sub_idx]
current_sets = np.array(current_sets)
sets_to_consider = current_sets[sub_idx]

print ("Final primal objective = {}".format(primal_obj))
print ("Number of sets to be considered = {}".format(len(sets_to_consider)))
