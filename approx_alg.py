import numpy as np
from scipy.optimize import linprog
import itertools
from itertools import chain, combinations

import time

def divide_intervals(weights, revenues, costs, K):
    """
    Divide the interval into sub-intervals such that each sub-interval is well-behaving.
    """
    n = len(weights)

    sigma_max = np.sum(sorted(weights)[-K::])
    dividing_pts = [0, sigma_max]

    # The signs of the utilities do not change
    for i in range(n):
        if costs[i] == 0:
            continue
        sigma = revenues[i]*weights[i]/costs[i] - 1
        if sigma > 0 and sigma < sigma_max:
            dividing_pts.append(sigma)

    # The ordering of utilities does not change
    for i in range(n):
        for j in range(i+1, n):
            # if the two items have the same cost, the order of their utilities do not change
            if costs[i] == costs[j]:
                continue

            sigma = (revenues[i]*weights[i] - revenues[j]*weights[j])/(costs[i]-costs[j]) - 1
            if sigma > 0 and sigma < sigma_max:
                dividing_pts.append(sigma)

    # The ordering of utility-to-weight ratio does not change
    for i in range(n):
        for j in range(i+1, n):
            # if the two items have the same cost, the order of their utilities do not change
            if costs[i]/weights[i] == costs[j]/weights[j]:
                continue

            sigma = (revenues[i] - revenues[j])/(costs[i]/weights[i]-costs[j]/weights[j]) - 1
            if sigma > 0 and sigma < sigma_max:
                dividing_pts.append(sigma)

    # The ordering of marginal increase in utility does not change
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                nominator = (revenues[j]*weights[j] * (weights[k]-weights[i]) \
                          +  revenues[k]*weights[k] * (weights[i]-weights[j]) \
                          +  revenues[i]*weights[i] * (weights[j]-weights[k]))
                denominator = (costs[j] * (weights[k]-weights[i]) \
                            +  costs[k] * (weights[i]-weights[j]) \
                            +  costs[i] * (weights[j]-weights[k]))
                if denominator == 0: continue
                sigma = nominator/denominator - 1
                if sigma > 0 and sigma < sigma_max:
                    dividing_pts.append(sigma)

    return np.unique(dividing_pts)

def find_max_objective(collection, weights, revenues, costs):
    """
    Find the set that yields the maximum rev(S)
    """
    best_obj = 0
    best_S = []

    for S in collection:
        objective = np.sum(revenues[S]*weights[S])/(1+np.sum(weights[S])) - np.sum(costs[S])
        if objective > best_obj:
            best_obj = objective
            best_S = S

    return best_S, best_obj

##################################################
############ Half Approx Algorithm ###############
##################################################

def half_approx_I(weights, revenues, costs, K, W_min, W_max, verbose=False):
    """
    Find an half-approx solution for the knapsack problem on a well-behaving interval I = [W_min, W_max]
    """

    weights = np.array(weights)
    revenues = np.array(revenues)
    costs = np.array(costs)

    n = len(weights)

    ### Step 0: Define a helper function that determines the half-approx solution for a sub-interval with two candidates ###
    def determine_half_approx(left_pt, right_pt, first_candidate, second_candidate):
        """
        Given interval [left_pt, right_pt), determine whether first_candidate or second_candidate is the 1/2-approx. solution
        """

        rw = revenues * weights

        utilities_left = revenues * weights/(1+left_pt) - costs
        utilities_right = revenues * weights/(1+right_pt) - costs

        U_left_first = np.sum(utilities_left[first_candidate])
        U_left_second = np.sum(utilities_left[second_candidate])
        U_right_first = np.sum(utilities_right[first_candidate])
        U_right_second = np.sum(utilities_right[second_candidate])

        # Compute the dividing pt
        break_pt = (np.sum(rw[first_candidate]) - np.sum(rw[second_candidate]))\
                    /(np.sum(costs[first_candidate]) - np.sum(costs[second_candidate])) - 1

        if U_left_first >= U_left_second:
            if U_right_first >= U_right_second:
                # the first_candidate is the 1/2-approx. solution for all W in [left_pt, right_pt)
                return[(left_pt, first_candidate)]
            else:
                # the first_candidate for [left_pt, break_pt], the second_candidate for [break_pt, right_pt]
                return [(left_pt, first_candidate), (break_pt, second_candidate)]
        else:
            if U_right_first >= U_right_second:
                return [(left_pt, second_candidate), (break_pt, first_candidate)]
            else:
                # the second_candidate is the 1/2-approx. solution for all W in [left_pt, right_pt)
                return [(left_pt, second_candidate)]


    ### Step 1: Initialization ###

    W_current = W_min
    dividing_pts = []

    collection = [[i] for i in range(n)]
    utilities_W_min = revenues * weights/(1+W_min) - costs

    # the items are ranked by their utility to weight ratio from highest to lowest
    utility_to_weight_ratio = utilities_W_min/weights
    rank = utility_to_weight_ratio.argsort()[::-1]

    ### Step 2: I_low ###
    # Add the items with the highest profit/weight ratio
    N_K = rank[:K]
    W_th = np.sum(weights[N_K])

    if verbose: print ("W_threshold = ", W_th)

    # Consider I_low if it is non-empty
    if W_th >= W_min:
        for i in range(1, K+1):
            W_temp = np.sum(weights[rank[:i]])

            if W_temp >= W_min and W_current < W_max:

                # Two points: W_current, min(W_temp, W_max)
                # Two candidates: rank[:i-1], rank[i-1]
                # Need to figure out: which one is the actual 1/2-approx. solution
                dividing_pts += determine_half_approx(W_current, min(W_temp, W_max), list(rank[:i-1]), [rank[i-1]])
                collection.append(list(rank[:i-1]))

                W_current = W_temp

    ### Step 3 ###
    # If there are more than K items with positive utilities
    if utilities_W_min[rank[K]] > 0 and W_th < W_max:

        # Start to evaluate the interval I_{high}
        if verbose:
            print ("Evaluate interval I_high")

        P_1, P_0, W_next = None, None, None

        # Initialize the profile
        if W_min <= W_th:
            P_1 = set(N_K)
            P_0 = set(range(n)).difference(P_1)
            W_next = W_th

            dividing_pts += [(W_th, list(N_K))]

        else:

            ### Solve the LP here ###
            A_ub = [list(weights), [1 for i in range(n)]]
            bounds = [(0, 1) for i in range(n)]

            # first solve the fractional knapsack problem
            b = [W_min, K]
            c = list(-utilities_W_min)
            x0 = [1 if i in set(rank[:K]) else 0 for i in range(n)]
            res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method='revised simplex', x0=x0)

            sol = np.round(res.x, 3)
            fractional_idx = np.where((sol != 0) & (sol != 1))[0]
            P_1 = set(np.where((sol == 1))[0])
            P_0 = set(np.where((sol == 0))[0])

            if len(fractional_idx) == 0:
                dividing_pts += [(W_min, list(P_1))]
                collection.append(list(P_1))
                W_next = np.sum(weights[list(P_1)])

            elif len(fractional_idx) == 2:

                collection += [list(P_1) + [fractional_idx[0]], list(P_1) + [fractional_idx[1]]]

                if weights[fractional_idx[0]] < weights[fractional_idx[1]]:

                    dividing_pts += determine_half_approx(W_min, min(np.sum(weights[list(P_1)]) + weights[fractional_idx[1]], W_max), \
                                                          list(P_1) + [fractional_idx[0]], [fractional_idx[1]])

                    P_1.add(fractional_idx[1])
                    P_0.add(fractional_idx[0])

                else:

                    dividing_pts += determine_half_approx(W_min, min(np.sum(weights[list(P_1)]) + weights[fractional_idx[0]], W_max), \
                                                          list(P_1) + [fractional_idx[1]], [fractional_idx[0]])

                    P_1.add(fractional_idx[0])
                    P_0.add(fractional_idx[1])

            elif len(fractional_idx) == 1:
                collection += [list(P_1), list(P_1) + [fractional_idx[0]]]
                dividing_pts += determine_half_approx(W_min, min(np.sum(weights[list(P_1)]) + weights[fractional_idx[0]], W_max), \
                                                          list(P_1), [fractional_idx[0]])

                P_1.add(fractional_idx[0])
            else:
                import IPython; IPython.embed()
                print (W_min, W_max)
                print (sol)
                raise Exception("Number of fractional variables is not 0 or 2.")

            W_next = np.sum(weights[list(P_1)])

        # Adaptively partition I_high
        while W_next < W_max:

            ### This part can be made faster ###
            utilities_W_next = revenues * weights/(1+W_next) - costs
            mat = np.array([[(utilities_W_next[j] - utilities_W_next[i])/(weights[j] - weights[i]) \
                             if (i in P_1) and (j in P_0) and (weights[j] > weights[i]) else -np.inf \
                             for j in range(n)] for i in range(n)])
            i_star, j_star = np.unravel_index(mat.argmax(), mat.shape)

            if mat[i_star, j_star] < 0:
                break

            dividing_pts += determine_half_approx(W_next, min(W_next - weights[i_star] + weights[j_star], W_max), list(P_1), [j_star])

            # update the profiles P_1 and P_0 and W_next
            P_1.remove(i_star); P_1.add(j_star)
            P_0.remove(j_star); P_0.add(i_star)
            W_next = W_next - weights[i_star] + weights[j_star]
            collection.append(list(P_1))

    elif utilities_W_min[rank[K]] < 0 and W_th < W_max:

        print ("There are less than K items with positive utilities")

        S = list(np.where(utilities_W_min > 0)[0])

        dividing_pts.append((W_min, S))

        collection.append(S)


    S_I, obj_I = find_max_objective(collection, weights, revenues, costs)
    return collection, dividing_pts, S_I, obj_I


def half_approx(weights, revenues, costs, K):
    """
    Find an half-approx solution for the infinite knapsack problems on [0, infinity]
    """

    start_time = time.time()

    intervals = divide_intervals(weights, revenues, costs, K)
    print ("number of intervals = {}".format(len(intervals)-1))

    best_S = None
    OPT = 0

    all_dividing_pts = []

    for i in range(len(intervals)-1):
        W_min = intervals[i]
        W_max = intervals[i+1]

        _, dividing_pts, S_I, obj_I = half_approx_I(weights, revenues, costs, K, W_min, W_max)

        if len(dividing_pts) == 0: print (W_min, W_max)

        all_dividing_pts += dividing_pts

        if obj_I > OPT:
            best_S = S_I
            OPT = obj_I

    print ("Total number of intervals after adaptive partitioning = {}".format(len(all_dividing_pts)))

    return best_S, OPT, all_dividing_pts, time.time() - start_time

##################################################
##################### FPTAS #######################
##################################################

def further_partition(sigma_l, sigma_u, scaled_revenues_l, scaled_revenues_u,
                      weighted_rev_half_approx, cost_half_approx, revenues, weights, costs, eps, n, K):

    # Check if there is a need for further partition
    if np.all(scaled_revenues_l == scaled_revenues_u):
        return [[sigma_l, scaled_revenues_l]]

    break_pts = []

    ### Compute the break_pts ###
    for i in range(n):
        if scaled_revenues_l[i] == scaled_revenues_u[i]:
            continue
        elif scaled_revenues_l[i] < scaled_revenues_u[i]:
            for u in range(scaled_revenues_l[i] + 1, scaled_revenues_u[i] + 1, 1):
                ### Compute the break pts
                W_break = (eps * u * weighted_rev_half_approx - K * revenues[i] * weights[i]) /\
                          (eps * u * cost_half_approx - K * costs[i]) - 1

                break_pts.append((W_break, i, u))

        else:
            for u in range(scaled_revenues_l[i], scaled_revenues_u[i], -1):
                ### Compute the break pts
                W_break = (eps * u * weighted_rev_half_approx - K * revenues[i] * weights[i]) /\
                          (eps * u * cost_half_approx - K * costs[i]) - 1

                break_pts.append((W_break, i, u-1))


    break_pts.sort(key = lambda x:x[0])

    result = [[sigma_l, scaled_revenues_l.copy()]]

    current_revenues = scaled_revenues_l.copy()

    for W_new, i_new, u_new in break_pts:

        result.append([W_new, current_revenues.copy()])
        result[-1][1][i_new] = u_new
        current_revenues = result[-1][1]

    return result


def get_assortment(best_weight_matrix, weight_profit, n, a_star, k_star):
    """
    Return the assortment based on the best_weight_matrix
    """
    a = a_star
    k = k_star
    result = []
    for i in range(n, 0, -1):
        weight, profit = weight_profit[i - 1]
        # we do not consider this item if its profit is negative
        if a < profit or profit < 0:
            continue
        if best_weight_matrix[i, a, k] == best_weight_matrix[i - 1, a - profit, k - 1] + weight:
            result.append(i-1)
            k -= 1
            a -= profit

    return result[::-1]

def FPTAS_I(weights, revenues, costs, K, sigma_l, sigma_u, half_approx_set, eps=0.1):
    """
    Fully polynomial-time approximation scheme method for solving knapsack problem on I = [sigma_l, sigma_u]
    """
    n = len(weights)
    assert n == len(costs)

    # if the half-approx set contains no items
    if len(half_approx_set) == 0:
        return [], 0, [sigma_l]

    # If none of the items have positive utility
    profits_sigma_l = revenues * weights/(1+sigma_l) - costs
    if np.all(profits_sigma_l <= 0):
        print("All profits are negative on [{}, {})".format(sigma_l, sigma_u))
        return [], 0, [sigma_l]

    ### STEP 3: FURTHER PARTITION ###
    result_further_partition = []

    adjusted_revenues_l = revenues * weights / (1+sigma_l) - costs
    adjusted_revenues_u = revenues * weights / (1+sigma_u) - costs

    adjusted_revenues_l_half_approx = np.sum(adjusted_revenues_l[half_approx_set])
    scaled_revenues_l = np.floor(adjusted_revenues_l/adjusted_revenues_l_half_approx * (K/eps)).astype(int)

    adjusted_revenues_u_half_approx = np.sum(adjusted_revenues_u[half_approx_set])
    scaled_revenues_u = np.floor(adjusted_revenues_u/adjusted_revenues_u_half_approx * (K/eps)).astype(int)

    weighted_rev_half_approx = np.sum((revenues * weights)[half_approx_set])
    cost_half_approx = np.sum(costs[half_approx_set])

    result_further_partition = further_partition(sigma_l, sigma_u, scaled_revenues_l, scaled_revenues_u,\
                                              weighted_rev_half_approx, cost_half_approx, \
                                              revenues, weights, costs, eps, n, K)

    further_patition_pts = [res[0] for res in result_further_partition]

    ### STEP 4: DYNAMIC PROGRAMMING ###
    all_collections = []

    # Get bound on max utilities
    bound = int(np.ceil(K**2/eps) + K)

    for idx in range(len(result_further_partition)):
        # Get the sub-interval, denoted by [lower_bound, upper_bound)
        # Get the rescaled utility on this sub-interval
        lower_bound = result_further_partition[idx][0]
        upper_bound = sigma_u if idx == len(result_further_partition) - 1 else result_further_partition[idx+1][0]

        rescaled_utility = result_further_partition[idx][1]

        ### PERFORM THE DP SCHEME ###

        weight_profit = [(weights[i], rescaled_utility[i]) for i in range(n)]

        best_weight_matrix = np.ones((n+1, bound+1, K+1)) * np.inf
        best_weight_matrix[0, 0, 0] = 0

        for i in range(1, n+1):
            # the weight and utility of the i-th item
            weight, utility = weight_profit[i - 1]

            for a in range(bound+1):
                for l in range(K+1):

                    # if the utility of item i is negative, do not consider it
                    if utility <= 0:
                        best_weight_matrix[i, a, l] = best_weight_matrix[i-1, a, l]
                    else:
                        if l > 0 and a >= utility:
                            best_weight_matrix[i, a, l] = \
                            min(best_weight_matrix[i-1, a, l], best_weight_matrix[i-1, a-utility, l-1] + weight)
                        else:
                            best_weight_matrix[i, a, l] = best_weight_matrix[i-1, a, l]

        collection = []
        feasible_set_idx = np.where((best_weight_matrix[n, :, :] <= upper_bound) & (best_weight_matrix[n, :, :] > 0))

        for i in range(len(feasible_set_idx[0])):
            a_i = feasible_set_idx[0][i]
            l_i = feasible_set_idx[1][i]

            try:
                assortment = get_assortment(best_weight_matrix, weight_profit, n, a_i, l_i)
            except:
                import IPython; IPython.embed()

            collection.append(assortment)

        all_collections = all_collections + collection

    best_assortment, OPT = find_max_objective(collection, weights, revenues, costs)

    return best_assortment, OPT, further_patition_pts


def FPTAS(weights, revenues, costs, K, eps=0.1):
    """
    FPTAS for the infinite knapsack problems on [0, infinity]
    """

    start_time = time.time()

    best_assortment = None
    OPT = 0

    ### Step 2: Apply 1/2-Approx. Algorithm
    _, _, all_dividing_pts, _ = half_approx(weights, revenues, costs, K)

    final_pts = []

    for idx in range(len(all_dividing_pts)):
        sigma_l = all_dividing_pts[idx][0]
        sigma_u = all_dividing_pts[idx+1][0] if idx < len(all_dividing_pts) - 1 else np.sum(sorted(weights)[-K::])

        half_approx_set = all_dividing_pts[idx][1]

        S_I, OPT_I, pts = FPTAS_I(weights, revenues, costs, K, sigma_l, sigma_u, half_approx_set, eps=eps)

        final_pts = final_pts + pts

        if OPT_I > OPT:
            best_assortment = S_I
            OPT = OPT_I

    print ("Total number of intervals after further partitioning in FPTAS :", len(final_pts))

    return best_assortment, OPT, time.time() - start_time
