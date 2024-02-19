from itertools import combinations
import numpy as np
import time, sys

### The following code is taken from the codebase for the following paper
###     Tulabandhula, T., Sinha, D., & Karra, S. (2022). Optimizing revenue while showing relevant assortments at scale.
###     European Journal of Operational Research, 300(2), 561-570.
### Accessed at https://github.com/thejat/scalable-data-driven-assortment-planning
### We used the staticMNL algorithm for benchmarking and evaluating the performance of our approximation algorithm

def calcRev(ast, p, v, prod):
    if len(p)==prod:
        p =  np.insert(p,0,0)   #making p a list of length n+1 by inserting a 0 in the beginning
    num = 0
    den = v[0]
    for s in range(len(ast)):
        num  = num + p[ast[s]]*v[ast[s]]
        den  = den + v[ast[s]]
    rev = num/den
    return rev

def staticMNL(prod, C, p, v):
    """
    # prod is the number of products
    # C is the capacity
    # p is a np array containing prices of products
    # v is the customer preference vector with the first entry denoting the parameter for the no-purchase option
    # The code below assumes v to be of length (number of products +1) and p to be of length number of products

    # Ensure that the entries of v are distinct otherwise there might be issues
    """
    st = time.time()
    n = prod + 1
    v =  v/v[0] # normalizing the no-purchase coefficient to 1

    if len(p)==prod:
        p =  np.insert(p,0,0) # appending 0 at the beginning to denote 0 price for the no-purchase option

    ispt = np.empty((n, n))*float("nan") # 2-D array for storing intersection points

    for j in range (1,n):
        ispt[0,j] = p[j]
        for i in range (1,j):
            ispt[i,j] = ((p[i]*v[i]) - (p[j]*v[j]))/(v[i] - v[j])

    ix = np.argsort(ispt.flatten())        # sorted indexing in the flattened array
    ret = int((n*n - n)/2)                 # number of relevant items, others are all inf

    pos = np.unravel_index(ix[:ret], np.shape(ispt))
    pos = np.column_stack((pos[0], pos[1]))


    numIspt = len(pos) # storing the number of intersection points

    sigma = np.ones((n-1, numIspt +1) ) * sys.maxsize # the number of possible permutations is 1+ the number of intersection points
    sigma[:,0] = 1+np.argsort(-1*v[1:]) # we want to sort v into descending order, so we are sorting -v in ascending order and storing the indexes

    A = np.ones((C, numIspt +1) ) * sys.maxsize
    G = np.ones((C, numIspt +1) ) * sys.maxsize
    B = np.ones((n-1, numIspt +1) ) * sys.maxsize

    Bcount = -1 # number of elements in current B vector

    A[:,0] = sigma[0:C,0]
    G[:,0] = sigma[0:C,0]

    for l in range(1,numIspt+1):
        sigma[:,l] = sigma[:,l-1]
        B[:, l] = B[:, l-1]

        # this is to ensure that the first coordinate is smaller -not sure if the below line is foolproof
        if(pos[l-1][0] > pos[l-1][1]):
            pos[l-1][0], pos[l-1][1] = pos[l-1][1], pos[l-1][0]

        if pos[l-1][0] != 0:
            idx1 = np.where(sigma[:,l-1] == pos[l-1][0])
            idx2 = np.where(sigma[:,l-1] == pos[l-1][1])

            sigma[idx1, l], sigma[idx2, l] =  sigma[idx2, l-1], sigma[idx1, l-1]

        else:
            B[Bcount + 1,l] = pos[l-1][1]
            Bcount = Bcount +1

        G[:,l] = sigma[0:C, l]
        temp = np.setdiff1d(G[:,l], B[:,l])
        A[0:len(temp), l ] = temp


    maxRev= 0 # denoting the maximum revenue encountered in the sets till now
    maxRevSet = -1 # denoting which set has the maximum revenue till now

    for l in range(numIspt+1):
        objs = A[np.where(A[:, l]< sys.maxsize), l].flatten()

        rev = calcRev(objs.astype('int'), p, v, prod)
        if rev > maxRev:
            maxRev = rev
            maxRevSet = l

    optSet = A[np.where(A[:, maxRevSet]< sys.maxsize), maxRevSet].flatten()
    optSet = optSet.astype(int)
    timeTaken = time.time() - st

    print (" ")
    print ("Products in the optimal assortment are", optSet)
    print ("Optimal revenue is", maxRev)
    print ("Time taken for StaticMNL is", timeTaken)

    return maxRev, optSet, timeTaken
