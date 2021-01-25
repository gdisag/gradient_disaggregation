# Please export MKL_NUM_THREADS=1 if multiprocess disaggregating

from numpy.linalg import svd
import copy
import numpy as np
import random
import scipy
import scipy.linalg
import sys
import time
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
import os

np.set_printoptions(threshold=sys.maxsize)

def relative_err(A, B):
    return np.mean(np.abs(A-B)/(np.abs(B)+1e-9))

def generate_A(r, n):
    return np.random.normal(0,1,size=(r,n))

def generate_T(m, r, sparsity=.2):
    return np.float32(np.rint(np.random.uniform(0, 1, (m, r)) <= sparsity))

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def compute_P_constraints(T, log_intervals):
    # From T, extract all log data that we need
    # - Number of participants per round
    # - Per user participation ums across rounds
    r = T.shape[1]
    m = T.shape[0]
    round_points = np.sum(T, axis=1)
    participant_points = []
    for user_id in range(r):
        user_participation_vector = T[:,user_id]
        interval = 0
        individual_participant_points = []
        while interval < m:
            n_participated = np.sum(user_participation_vector[interval:interval+log_intervals])
            individual_participant_points.append(((interval,interval+log_intervals), n_participated))
            interval += log_intervals
        participant_points.append(individual_participant_points)
    return participant_points

def disaggregate_grads(grads, participant_points, interval=None, gt=None, gt_P=None, verbose=False, noisy=True, override_dim_to_use=None):

    def intersects(x, y):
        return not (x[1] < y[0] or y[1] < x[0]) and (x[1] <= y[1] and x[0] >= y[0])

    # Params
    nrounds, ndim = grads.shape
    nusers = len(participant_points)

    # Try to determine interval from participant constraints 
    max_interval = max([x[0][1]-x[0][0] for y in participant_points for x in y])
    interval = max(nusers*5,max_interval*10) if interval is None else interval
    interval = min(interval, grads.shape[0])

    # Calculate dimension based on matrix rank
    if override_dim_to_use is not None:
        ndim_to_use = override_dim_to_use
    else:
        ndim_to_use = min(ndim, max(nrounds, nusers))    
        for ndim_to_use in range(max(nrounds,nusers), ndim, (ndim-max(nrounds,nusers))//10):
            if np.linalg.matrix_rank(grads[0:interval,0:ndim_to_use]) >= nusers:
                break
        ndim_to_use = ndim

    # Disaggregate gradients averaging over rounds
    disaggregated = 0
    c = 0
    all_P = []
    for total, row in list(enumerate(range(0, nrounds, interval))):
        if verbose:
            print("Set %d of %d" % (total, nrounds//interval))
        lower,upper = row,row+interval
        upper = min(upper, nrounds)
        cutoff_start = 0
        if upper-lower < interval:
            cutoff_start = np.abs((upper-interval)-lower)
            lower = upper-interval

        aggregated_chunk = grads[lower:upper,0:ndim_to_use]
        filtered_participant_points = [[y for y in x if intersects(y[0], (lower,upper))] for x in participant_points]
        filtered_participant_points = [[((y[0][0]-lower, y[0][1]-lower), y[1]) for y in x] for x in filtered_participant_points]
        Ps = reconstruct_participant_matrix(aggregated_chunk, filtered_participant_points, verbose=verbose, noisy=noisy)
        if len(Ps) == 1:
            P = Ps[0]
            all_P.append(P[cutoff_start:,:])
            disaggregated += np.linalg.lstsq(P, grads[lower:upper,:], rcond=None)[0]
            c += 1
            if gt_P is not None:
                ham_dist = np.sum(np.abs(gt_P[lower:upper,:]-P))
                if verbose:
                    print("Ham distance of P Matrix: %f" % ham_dist)
            if gt is not None:
                rel_err = relative_err(disaggregated/c, gt)
                if verbose:
                    print("Relative Error vs Ground Truth: %f" % rel_err)
            
    if c == 0:
        return float("inf")
    return disaggregated / c, np.concatenate(all_P, axis=0)

def reconstruct_participant_matrix(D, participant_points,
                                   top_k=None, verbose=False, noisy=False, column_timelimit=None,
                                   exit_after_tle=False, multiprocess=True):
    # D - T*Grads where T is the user participant matrix
    # participant_points - array of (round #, count), where count is 
    #                      the cumulative number of participations by
    #                      the specific user at round #
    # round_point - number of participants per round

    # Candidates for participant vectors

    if not multiprocess:
        participant_vectors = []
        n_possibilities_per_user = []
        statuses = []
        cached_space = None
        nrounds = D.shape[0]
        for i, individual_participant_point in enumerate(participant_points):
            if top_k is not None and len(participant_vectors) >= top_k:
                break
            if verbose:
                print("Reconstructing: %d of %d" % (i,len(participant_points)))
            p_vector, model, cached_space = compute_participant_candidate_vectors(D, individual_participant_point, nrounds, len(participant_points), noisy=noisy, timelimit=column_timelimit, cached_space=cached_space)
            if model.status == GRB.TIME_LIMIT and exit_after_tle:
                if verbose:
                    print("Time limit exceeded")
                return []        
            participant_vectors.append(p_vector)
            n_possibilities_per_user.append(len(p_vector))
            if verbose:
                print("Obtained candidate vector for user %d" % (i))
                sys.stdout.flush()
            if len(p_vector) <= 0:
                return []

        if verbose:
            print("# of candidate vectors per user: %s" % str(n_possibilities_per_user))

        # Naive method -- just stack and return
        naive = np.stack([x[0] for x in participant_vectors]).T
        return [np.rint(naive)]

    # Multiprocessing code
    nrounds = D.shape[0]
    nusers = len(participant_points)

    cached_space = compute_space(D, nusers, noisy=noisy)
    arguments = []
    for i, participant_point in enumerate(participant_points):
        arguments.append((i, None, participant_point, nrounds, nusers, noisy, column_timelimit, False,cached_space, verbose, exit_after_tle))
    if top_k is not None:
        arguments = arguments[:top_k]

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    try:
        indices_and_participant_vectors = pool.imap_unordered(compute_participant_candidate_vectors_multicore_wrapper, arguments)
        indices_and_participant_vectors = sorted(indices_and_participant_vectors, key=lambda x:x[0])
        participant_vectors = [x[1] for x in indices_and_participant_vectors]        
    except Exception:
        if verbose:
            print("Timelimit exceeded on one of the workers. Exiting")
        pool.close()
        pool.terminate()
        participant_vectors = [np.zeros((1,nrounds)) for i in range(nusers)]
    else:
        pool.close()
        pool.join()

    participant_vectors = np.concatenate([x for x in participant_vectors], axis=0).T
    return [np.rint(participant_vectors)]

def compute_space(D, nusers, noisy=True):
    
    # We use SVD to handle noisy gradients
    if noisy:
        u,s,vh = np.linalg.svd(D, full_matrices=False)
        cutoff = sorted(list(s.flatten()), reverse=True)[nusers]
        s[s<cutoff] = 0
        D = (u*s).dot(vh)

    # Constrain all binary variables to lie in the image of D
    bases = scipy.linalg.orth(D)
    inspace = nullspace(bases.T).T

    return inspace

def compute_participant_candidate_vectors_multicore_wrapper(args):
    user_index,D,p_points, nrounds,nusers,noisy,timelimit,find_all_solutions,cached_space,verbose,exit_after_tle = args
    solution, model, space = compute_participant_candidate_vectors(D, p_points, nrounds, nusers,
                                                                   noisy=noisy, timelimit=timelimit,
                                                                   find_all_solutions=find_all_solutions, cached_space=cached_space)
    if model.status == GRB.TIME_LIMIT:
        if exit_after_tle:
            raise Exception("TLE")
    if verbose:
        print(".",end="")
        sys.stdout.flush()
    return (user_index, solution)

def compute_participant_candidate_vectors(D, p_points, nrounds, nusers,
                                          noisy=False, timelimit=None, find_all_solutions=False, cached_space=None):

    # Create problem
    m = nrounds
    model = gp.Model()
    x = model.addMVar(shape=m, vtype=GRB.BINARY)

    if cached_space is None:
        inspace = compute_space(D, nusers, noisy=noisy)
    else:
        inspace = cached_space

    # Create constraints on participations
    for p_point in p_points:
        (low,high), s = p_point
        model.addConstr(x[low:high].sum() == s)

    # Solve
    model.setParam('OutputFlag', 0)
    model.setObjective(x@(inspace.T.dot(inspace)@x), GRB.MINIMIZE)

    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)

    model.setParam(GRB.Param.Threads, 1)
    model.optimize()
    
    return [model.X], model, inspace

def test_recover_p():

    # m - number of rounds
    # r - number of users
    # n - dimension of gradient
    # sparsity - percent participate per round
    # log_intervals - every log_intervals rounds will log how many times a user participated
    m = 200
    r = 20
    n = 200
    sparsity=.1
    #m = 100
    #r = 20
    #n = 100
    #sparsity = .1
    log_intervals = int(1/sparsity)

    # Create data
    A = np.rint(generate_A(r,n)) + 5
    T = generate_T(m,r,sparsity=sparsity)
    D = T.dot(A)

    print("Number of rounds: %d" % m)
    print("Number of users: %d" % r)
    print("Number of grad dims: %d" % n)
    print("Rank of participant matrix: %d" % np.linalg.matrix_rank(T))
    assert(np.linalg.matrix_rank(T) >= r)

    participant_points = compute_P_constraints(T, log_intervals)
    
    # Solve    
    T_reconstructed_candidates = reconstruct_participant_matrix(D, participant_points, verbose=True)
    T_reconstructed = T_reconstructed_candidates[0]
    err = np.linalg.norm(T_reconstructed-T)
    print("Error between reconstructed vs truth: %f" % 
          err)
    assert(err <= 1e-5)

def test_recover_grad_sanity():
    
    # m - number of rounds
    # r - number of users
    # n - dimension of gradient
    # sparsity - percent participate per round
    # log_intervals - every log_intervals rounds will log how many times a user participated
    m = 5000
    r = 80
    n = 400
    sparsity=.1
    log_intervals = 2*int(1/sparsity)

    # Create data
    user_grads = np.random.normal(0, 1, size=(r,n))

    # Construct rounds
    P = []
    aggregated = []
    total = 0
    for round in range(m):
        row = generate_T(1, r, sparsity=sparsity)        
        with_noise = user_grads + np.random.normal(0, .2, size=user_grads.shape)
        total += with_noise
        row_grads = row.dot(with_noise)
        P.append(row)
        aggregated.append(row_grads)
    P = np.concatenate(P, axis=0)
    aggregated = np.concatenate(aggregated, axis=0)

    print("Number of rounds: %d" % m)
    print("Number of users: %d" % r)
    print("Number of grad dims: %d" % n)
    print("Rank of participant matrix: %d" % np.linalg.matrix_rank(P))
    assert(np.linalg.matrix_rank(P) >= r)

    participant_points = compute_P_constraints(P, log_intervals)
    
    disaggregated, P = disaggregate_grads(aggregated, participant_points, verbose=True, gt=user_grads, gt_P=P)
    rel_err = relative_err(disaggregated, user_grads)
    print("Relative Error (percentage): %f" % rel_err)


def test_recover_grad():
    
    # m - number of rounds
    # r - number of users
    # n - dimension of gradient
    # sparsity - percent participate per round
    # log_intervals - every log_intervals rounds will log how many times a user participated
    m = 400
    r = 40
    n = 400
    sparsity=.5
    log_intervals = int(1/sparsity)*5
    noise = .2
    mat_intervals = 100

    # Create data
    A = np.random.normal(1, .1, size=(r,n))
    Ds = []
    Ts = []
    for i in range(0, m, mat_intervals):
        random_noise = np.random.normal(0, noise, size=(r,n))
        T = generate_T(mat_intervals,r,sparsity=sparsity)
        Ts.append(T)
        Ds.append(T.dot(A+random_noise))
    D = np.concatenate(Ds, axis=0)
    T = np.concatenate(Ts, axis=0)


    print("Number of rounds: %d" % m)
    print("Number of users: %d" % r)
    print("Number of grad dims: %d" % n)
    print("Rank of participant matrix: %d" % np.linalg.matrix_rank(T))
    assert(np.linalg.matrix_rank(T) >= r)

    participant_points = compute_P_constraints(T, log_intervals)
    
    disaggregated, P = disaggregate_grads(D, participant_points, verbose=True, gt=A, interval=mat_intervals)
    rel_err = relative_err(disaggregated, A)
    print("Relative Error (percentage): %f" % rel_err)

if __name__=="__main__":

    np.random.seed(0)
    #test_recover_p()
    #test_recover_grad()
    test_recover_grad_sanity()
