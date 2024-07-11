# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:13:50 2023 @author: EEM
MOdified on Sun JAN 20 05:10 2024 @author: SSJin
"""
# import numpy as np
# import scipy as sp
# from scipy import stats
# from scipy.stats import qmc
# import pandas as pd
# import networkx as nx
# import os
# import matplotlib.pyplot as plt

# import time

# import varstool.sampling.plhs as plhs


#%%
import ray
import time
import numpy as np
from typing import Tuple
from itertools import combinations


def _lhd_cost(arr: np.ndarray, axis: int=1) -> float:
    '''A simple cost function used in the PLHS Greedy algorithm
    Parameters
    ----------
    arr : array_like
        the input array (nxd dimensions)
    axis : int, optional
        the axis along which the cost is calculated
        defaults to ``1`` in PLHS
    Returns
    -------
    f : float
        the cost function
    '''

    # get the bins equal to the number of rows
    # in PLHS, each row is a sample series, and each column
    # corresponds to a parameter/factor/variable
    edges = np.linspace(start=0, stop=1, num=arr.shape[0] + 1)
    f = -np.sum(_bin_count(np.digitize(arr, edges), axis=axis))

    return f


def _bin_count(arr: np.ndarray, axis: int=0) -> np.ndarray:
    '''Calculates the number of unique values along the `axis` of the given
    `arr`. This function is used in PLHS algorithm to check LHS-conformity
    of the generated random samples.
    Parameters
    ----------
    arr : array_like
        the input array of interest
    axis : int, optional
        the axis along which the unique values are counted,
        ``0`` for `rows` and ``1`` for `columns`, defaults to ``0``
    Returns
    -------
    unique_count : array_like
        the number of unique values along each axis
    Source
    ------
    .. [1] https://stackoverflow.com/questions/
           48473056/number-of-unique-elements-per-row-in-a-numpy-array
           (the most efficient method)
    '''

    if axis:  # the method does operations row-wise...
        arr = arr.T

    n = arr.max() + 1
    a_off = arr + (np.arange(arr.shape[0])[:, None]) * n
    M = arr.shape[0] * n
    unique_count = (np.bincount(
        a_off.ravel(), minlength=M).reshape(-1, n) != 0).sum(1)
    return unique_count


def _sampler(sp: int, params: int, slices: int, seed: int=None) -> np.ndarray:
    '''A simple sampling algorithm to create lhs slices.
    Parameters
    ----------
    lb : int
        lower bound of the sequence
    ub : int
        upper bound of the sequence
    slices : int
        the number of slices
    seed : int
        seed number for randomization
    Returns
    -------
    sample_array : array_like
        the final sample array
    References
    ----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867.
    .. [2] Sheikholeslami, R., & Razavi, S. (2017). Progressive Latin Hypercube
           Sampling: An efficient approach for robust sampling-based analysis of
           environmental models. Environmental modelling & software, 93, 109-126
    '''

    # define the randomization seed number
    if seed:
        np.random.seed(seed)

    # check the dtype of input arguments
    msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int]:
        raise ValueError(msg.format('sp'))
    if type(params) not in [int]:
        raise ValueError(msg.format('params'))
    if type(slices) not in [int]:
        raise ValueError(msg.format('slices'))

    # check the number of slices and sample points
    if (sp % slices) != 0:
        raise ValueError("sample points must be a multiplier of slices.")

    # check the sign of the input arguments
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if params < 0:
        raise ValueError(sign_msg.format('params'))
    if slices < 0:
        raise ValueError(sign_msg.format('slices'))

    # calculate the number of slices
    slice_sp = sp // slices  # to get int

    # generate slices using sampling (int) without permutation
    def rand_perm(slice_sp, slices): return np.concatenate(
        [np.random.permutation(slice_sp) + 1 for _j in range(slices)])
    sample_array = np.stack([rand_perm(slice_sp, slices)
                             for _i in range(params)])

    # positional function definition
    def slice_spec(row, slice_sp): return np.stack(
        [(row == _j + 1) for _j in range(slice_sp)])

    # row-wise assessment
    for _row in range(0, sample_array.shape[0]):
        position_array = slice_spec(sample_array[_row, :], slice_sp)
        for kk in range(0, slice_sp):
            lb = (kk * slices) + 1
            ub = (kk + 1) * slices
            perm = _perm_intv(lb, ub, slices, seed)
            try:
                sample_array[_row, position_array[kk, :]] = perm
            except:  # sometimes a number might be missing due to randomness...
                raise RuntimeError(
                    "error! change the seed number and try again.")
    sample_array = np.random.uniform(sample_array - 1, sample_array)
    sample_array /= sp

    return sample_array.T

def _perm_intv(lb: int, ub: int, slices: int, seed: int=None) -> np.ndarray:
    '''A simple random sampling given the lower and upper bounds,
    without permutation, and amongst the integers in the interval.
    Parameters
    ----------
    lb : int
        lower bound of the sequence
    ub : int
        upper bound of the sequence
    slices : int or None
        the number of slices, defaults to ``None``
    Returns
    -------
    perm : array_like
        the sampled np.array
    '''

    # define the randomization seed number
    if seed:
        np.random.seed(seed)
    # # a simple sampling without permutation algorithm
    # length = np.abs(ub - lb) + 1
    # perm = np.arange(start=lb, stop=ub + 1, step=1)
    # for k in range(2, length + 1):
    #     index1 = int(np.ceil(np.random.rand() * k))
    #     index2 = perm[k - 1]
    #     perm[k - 1] = perm[index1 - 1]
    #     perm[index1 - 1] = index2
    # perm = perm[0:slices + 1]
    perm = np.random.randint(lb, ub+1, slices) # revised
    # DEBUG
    # print('perm is:')
    # print(perm)
    # END DEBUG

    return perm

def _knn(arr1: np.ndarray, arr2: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''A simple k-NN algorithm to find the minimum Euclidean distance
    Parameters
    ----------
    arr1 : array_like
        the first array of data having
        ``n`` rows and ``d`` columns
    arr2 : array_like
        the second array of data
        ``m`` rows and ``d`` columns
    k : int
        the number of neighbors
    Returns
    -------
    distances : array_like
        Euclidean distances between `arr1` and `arr2` points
    indices : array_like
        the indices of the distances between `arr1` and `arr2`
        points
    '''

    # calculating the distance between points
    distances = -2 * arr1 @ arr2.T + np.sum(arr2**2, axis=1) + \
        np.sum(arr1**2, axis=1)[:, np.newaxis]

    # taking into account the floating point discrepancies
    distances[distances < 0] = 0
    distances = distances**.5
    # print(distances)
    indices = np.argsort(distances, 0)
    # print('indices :', indices)
    distances = np.sort(distances, 0) # original
    # print(distances)
    # distances = distances[indices].reshape(arr1.shape[0], k) # ?ˆ˜? • by EEM 230611
    # distances = np.take_along_axis(distances, indices, axis=0) # ?ˆ˜? • by EEM 230611

    # reshaping the arrays
    indices = indices[0:k, :].T

    distances = distances[0:k, :].T.flatten().reshape(arr1.shape[0], k)

    return indices, distances


def _get_min_distance(arr: np.ndarray, k: int=3) -> float:
    '''The minimum Euclidean distance between sample points as a measure
    of sparsity of the sampling space
    Parameters
    ----------
    arr : array_like
        the input array of any size
    Returns
    -------
    min_distance : float
        the minimum distance calculated
    '''

    # idx, distance = _knn(arr, arr, k)  # idx index start from 0 ## original
    distance = _knn2(arr, arr, k)  # idx index start from 0 # revised by EEM 230611
    min_distance = np.min(distance[:, 1])

    return min_distance


def _get_corr(arr: np.ndarray) -> float:
    '''Calculates the correlation between the sample columns and
    reports the sum of squared correlation values.
    Parameters
    ----------
    arr : array_like
        the input array of any size
    Returns
    -------
    sq_corr : float
        sum of the squared correlation values
    '''

    return sum(sum(np.triu(np.corrcoef(arr, rowvar=False)**2, k=1)))


def _get_corr_sub(arr: np.ndarray) -> float:
    '''Calculates the correlation between the sample columns and
    reports the sum of squared correlation values.
    Parameters
    ----------
    arr : array_like
        the input array of any size
    Returns
    -------
    sq_corr : float
        sum of the squared correlation values
    '''

    return np.mean(np.array([_get_corr(x) for x in arr]))


def _get_min_distance_sub(arr: np.ndarray, k: int=3) -> float:
    '''Calculates the minimum Euclidean distance
    between sample points as a measure of sparsity of the
    sampling space in each slice. The returned value is averaged
    amongst the minimum value of the slices.
    Parameters
    ----------
    arr : array_like
        the input array of any size
    k : int
        the number of nearest neightbors, defaults to ``3``
    Returns
    -------
    min_distance : float
        the minimum distance calculated
    '''

    return np.mean(np.array([_get_min_distance(x, k) for x in arr]))


def _knn2(arr1: np.ndarray, arr2: np.ndarray, k: int) -> np.ndarray:
    '''A simple k-NN algorithm to find the minimum Euclidean distance
    Parameters
    ----------
    arr1 : array_like
        the first array of data having
        ``n`` rows and ``d`` columns
    arr2 : array_like
        the second array of data
        ``m`` rows and ``d`` columns
    k : int
        the number of neighbors
    Returns
    -------
    distances : array_like
        Euclidean distances between `arr1` and `arr2` points
    indices : array_like
        the indices of the distances between `arr1` and `arr2`
        points
    '''

    # calculating the distance between points
    distances = -2 * arr1 @ arr2.T + np.sum(arr2**2, axis=1) + \
        np.sum(arr1**2, axis=1)[:, np.newaxis]

    # taking into account the floating point discrepancies
    distances[distances < 0] = 0
    distances = distances**.5    
    
    distances_revised = np.partition(distances, k, axis=0, kind='introselect', order=None) # revised 230613
    distances_revised = np.sort(distances_revised.T[:,:k],axis=1) # revised 230613
    # print('revised')
    # print(distances_revised.shape)
    # print(distances_revised)
    return distances_revised
    
    # distances = np.sort(distances, 0) # original
    # # print('original')
    # # print(distances.shape)
    # # print(distances)
    # # reshaping the arrays
    # distances = distances[0:k, :].T.flatten().reshape(arr1.shape[0], k)
    # # print('reshape')
    # # print(distances.shape)
    # # print(distances)
    # return distances
"""
Contributors:
-------------
Sheikholeslami, Razi, (2017): algorithm, code in MATLAB (c)
Razavi, Saman, (2017): algorithm, code in MATLAB (c), supervision
Keshavarz, Kasra, (2021): code in Python 3
Matott, Shawn, (2019): code in C/++
"""
@ray.remote
def slhs(
    sp: int,
    params: int,
    slices: int,
    seed: int=None,
    iterations: int=20,
    criterion: str='maximin'
) -> Tuple[np.ndarray, np.ndarray]:
    '''This function created SLHS samples, based on [1] and [2]. In
    order to find optimal ordering of slices, the k-NN method is
    utilized.
    Parameters
    ----------
    sp : int
        number of sample points
    params : int
        number of parameters/variables/factors
    slices : int
        number of slices
    seed : int or None, optional
        seed number for randomization, defaults to ``None``
    iterations : int, optional
        maximum iteration number, defaults to ``20``
    criterion : str, optional
        the criterion for assessing the quality of sample points;
        the available options are: ``'maximin'`` and ``'correlation'``,
        defaults to ``'maximin'``
    Returns
    -------
    slhs_sample_x : array_like
        the final slhs sample array based on ``criterion``
    slhs_sample_x_slice : array_like
        the final slhs sample array slices based on ``criterion``
    References
    ----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867
    .. [2] Sheikholeslami, R., & Razavi, S. (2017). Progressive Latin Hypercube
           Sampling: An efficient approach for robust sampling-based analysis of
           environmental models. Environmental modelling & software, 93, 109-126
    '''

    # define the seed number
    if seed:
        np.random.seed(seed)

    # Check the inputs and raise appropriate exceptions
    msg_crt = ("'{}' is not defined; available options: 'maximin', 'correlation'")
    if type(criterion) is not str:
        raise TypeError(msg_crt.format(str(criterion)))
    if criterion not in ['maximin', 'correlation']:
        raise ValueError(msg_crt.format(criterion))

    # calculate the number of slices
    slice_sp = sp // slices  # to get int

    # Check the criterion
    if criterion == 'maximin':
        best_sample = _sampler(sp, params, slices)
        best_sub_sample = best_sample.reshape((slices, slice_sp, params))
        best_sample_cost = _get_min_distance(best_sample, k=3)
        best_sub_sample_cost = _get_min_distance_sub(best_sub_sample)
        cost_func = np.mean([best_sample_cost, best_sub_sample_cost])
        # print(best_sample_cost, best_sub_sample_cost)

        for it in range(iterations):
            new_sample = _sampler(sp, params, slices)
            new_sub_sample = new_sample.reshape((slices, slice_sp, params))
            new_sample_cost = _get_min_distance(new_sample)
            new_sub_sample_cost = _get_min_distance_sub(new_sub_sample)
            new_cost_func = np.mean([new_sample_cost, new_sub_sample_cost])
            # print(new_sample_cost, new_sub_sample_cost)
            # check the cost function value
            if new_cost_func > cost_func:
                best_sample = new_sample
                cost_func = new_cost_func

        slhs_sample_maximin = best_sample
        # slhs_sample_maximin_slice = slhs_sample_maximin.reshape(
        #     (slices, slice_sp, params))

        return slhs_sample_maximin#, slhs_sample_maximin_slice

    elif criterion == 'correlation':
        best_sample = _sampler(sp, params, slices)
        best_sub_sample = best_sample.reshape((slices, slice_sp, params))
        best_sample_cost = _get_corr(best_sample)
        best_sub_sample_cost = _get_corr_sub(best_sub_sample)
        cost_func = np.mean([best_sample_cost, best_sub_sample_cost])

        for it in range(iterations):
            new_sample = _sampler(sp, params, slices)
            new_sub_sample = new_sample.reshape((slices, slice_sp, params))
            new_sample_cost = _get_corr(new_sample)
            new_sub_sample_cost = _get_corr_sub(new_sub_sample)
            new_cost_func = np.mean([new_sample_cost, new_sub_sample_cost])

            # check the cost function value
            if new_cost_func < cost_func:
                best_sample = new_sample
                cost_func = new_cost_func

        slhs_sample_correl = best_sample
        # slhs_sample_correl_slice = slhs_sample_correl.reshape(
        #     (slices, slice_sp, params))

        return slhs_sample_correl#, slhs_sample_correl_slice

@ray.remote
def _greedy_plhs(
    sp: int,
    slices: int,
    sample: np.array
) -> Tuple[np.array, np.array, float, float]:
    '''Generate progressive lating hypercube samples (plhs)
    This function is a Progressive Latin Hypercube Sampling (PLHS)
    using an optimal Sliced Lating Hypercube Sampling design (SLHS)
    in the frame of a greedy algorithm; based on [1] and [2]
    Parameters
    ----------
    sp : int
        number of sample points
    slices : int
        number of slices
    sample : array_like
        the sampled matrix/array
    Returns
    -------
    plhs : array_like
        plhs sample array
    plhs_slices : array_like
        plhs sample slices (sub-samples)
    f_priori : float
        objective function value before optimization
    f_posteriori : float
        objective function value after optimization
    References
    ----------
    .. [1] Ba, S., Myers, W.R., Brenneman, W.A., 2015. Optimal sliced Latin
           hypercube designs. Technometrics 57 (4), 479e487.
           http://dx.doi.org/10.1080/00401706.2014.957867
    .. [2] Sheikholeslami, R., & Razavi, S. (2017). Progressive Latin Hypercube
           Sampling: An efficient approach for robust sampling-based analysis of
           environmental models. Environmental modelling & software, 93, 109-126
    '''

    # check the dtype of input arguments
    msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int]:
        raise ValueError(msg.format('sp'))
    if type(slices) not in [int]:
        raise ValueError(msg.format('slices'))

    # check the number of slices and sample points
    if (sp % slices) != 0:
        raise ValueError("sample points must be a multiplier of slices.")

    # check the sign of the input arguments
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if slices < 0:
        raise ValueError(sign_msg.format('slices'))

    # slice_sp = sp // slices
    # row-wise slicing - PLHS standard
    # sample = np.array(sample,dtype=object)
    # print(sample)
    sub_sample = np.array(np.split(sample, slices, axis=0))
    # priori cost function value
    f_priori = np.mean([_lhd_cost(sl_agg) for sl_agg in
                        [np.concatenate(sub_sample[0:t + 1, ...]) for t in range(slices)]])

    # let's find out the first two slices that results in the lowest
    # cost function and make the original code more efficient...
    # pay attention to axis=0, PLHS standard is row-wise...
    indices = list(range(sub_sample.shape[0]))
    def least_cost(idx): return _lhd_cost(
        np.concatenate(np.take(sub_sample, idx, axis=0)))
    greedy_indices = list(
        min(combinations(indices, 2), key=least_cost))  # 2: pair

    # find the next slices in a loop and add its indice to the
    # `greedy_indices` list
    indices = list(set(indices) - set(greedy_indices))
    for _ in range(len(indices)):
        greedy_indices = list(min([greedy_indices + [idx]
                                   for idx in indices], key=least_cost))
        indices = list(set(indices) - set(greedy_indices))  # same as above...

    # check the `posteriori` cost function value
    # pay attention to axis=0, PLHS standard is row-wise...
    plhs_slices = np.take(sub_sample, greedy_indices, axis=0)
    plhs = np.concatenate(plhs_slices)
    f_posteriori = np.mean([_lhd_cost(sl_agg) for sl_agg in
                            [np.concatenate(plhs_slices[0:t + 1, ...]) for t in range(slices)]])

    return (plhs, plhs_slices, f_priori, f_posteriori)

def plhs(sp: int, params: int, slices: int, seed: int=None, iterations: int=10, criterion: str='maximin') -> Tuple[np.ndarray, np.ndarray]:
    # slhs ?•¨?ˆ˜ë¥? ë³‘ë ¬ë¡? ?‹¤?–‰?•˜?Š” ?°ì½”ë ˆ?´?„°
    @ray.remote
    def slhs_parallel(sp: int, params: int, slices: int, seed: int=None, iterations: int=20, criterion: str='maximin') -> Tuple[np.ndarray]:
        return slhs.remote(sp, params, slices, seed, iterations, criterion)
    @ray.remote
    def greedy_plhs_parallel(sp: int, slices: int, sample: np.array) -> Tuple[np.array, np.array, float, float]:
        return _greedy_plhs.remote(sp, slices, sample)

    # ?‹œ?‘? ?„ ?„¤? •?•˜?—¬ ìµœì  ë¹„ìš© ?•¨?ˆ˜ ?„ ?ƒ
    # f_best = 0

    # ë³‘ë ¬ ì²˜ë¦¬ë¥? ?œ„?•´ ray ê°ì²´ ?‚¬?š©
    # ray_samples = []
    # ray_slices = []
    ray_slhs_dataset = []
    for _iter in range(iterations):
        if seed:
            seed += 50
        ray_slhs_data = slhs_parallel.remote(sp, params, slices, seed, iterations, criterion)
        
        # slhs ?•¨?ˆ˜ë¥? ë³‘ë ¬ë¡? ?‹¤?–‰?•˜ê³? ê²°ê³¼ë¥? ray ê°ì²´?— ????¥
        ray_slhs_dataset.append(ray_slhs_data)
    
    slhs_sample = np.array(ray.get(ray.get(ray_slhs_dataset)))
    
    ray_greedy_dataset = []
    # ray ê°ì²´ ê²°ê³¼ë¥? ê°?? ¸????„œ ìµœì  ë¹„ìš© ?•¨?ˆ˜?— ????•œ ìµœìƒ?˜ ?ƒ˜?”Œ ?„ ?ƒ
    for i in range(iterations):
        # plhs_candidate, plhs_candidate_slices, _, f_candidate = greedy_plhs_parallel.remote(sp, slices, slhs_sample[i,:,:])
        ray_greedy_data = greedy_plhs_parallel.remote(sp, slices, slhs_sample[i,:,:])
        ray_greedy_dataset.append(ray_greedy_data)
        
    ray_greedy_dataset = np.array(ray.get(ray.get(ray_greedy_dataset)), dtype=object)
    best = np.argmin(ray_greedy_dataset[:,3])    
    plhs_sample = ray_greedy_dataset[best,0].copy()
    plhs_sample_slices = ray_greedy_dataset[best,1].copy()
    return plhs_sample, plhs_sample_slices

#%%

# ray.shutdown()
# ray.init()

import pickle
import json

with open('config.json') as f:
    config = json.load(f)
    
n_rep = config['n_rep']
n_samples = config['n_samples']
n_slices = config['n_slices']
rand_seed0 = config['rand_seed0']
n_iter = config['n_iter']
n_params = config['n_params']

# n_rep, n_samples = 20, 10000
# n_slices = [25] # 25, 50, 100

for i in range(n_rep):
    rand_seed1 = rand_seed0 * (i+1)

    for n_slice in n_slices:
        X = dict()

        s2_time = time.time()
        # Crieterion for optimizer:  maximin // correlation
        X_temp = plhs(sp = n_samples, params=n_params, seed=rand_seed1, slices = n_slice, iterations = n_iter, criterion='maximin')
        e2_time = time.time()
        print('time :',e2_time - s2_time)


        # ray.shutdown()
        #%%
        # print(X_temp[0] == Y_temp[0])
        #%%

        outfile = 'PLHS_data_' + str(rand_seed1) +'_' + str(n_slice) +'_' + str(n_samples) +'_' + str(n_params) +'.pkl'
        print(outfile)
        #%%

        ## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ##
        ## Modified
        # os.chdir(r'.\testPLHS')
        ##
        ## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ##

        with open(outfile,'wb') as file:
            pickle.dump(X_temp,file)



#%%
# from cProfile import Profile
# # profiler = Profile()
# # profiler.run('plhs(sp = n_samples, params=n_params, seed=rand_seed0, slices = n_slices)')

# import cProfile
# cProfile.run('plhs(sp = n_samples, params=n_params, seed=rand_seed0, slices = n_slices, iterations = 10)')
#%%
# x = np.array([3, 1, 2])
# print(np.argsort(x,0))
# print(x[np.argsort(x)])

# x_before = [[0.        , 0.51824943, 0.63817193, 0.88165522],
#  [0.51824943, 0.        , 0.79407987, 0.71123707],
#  [0.63817193, 0.79407987, 0.        , 0.48312259],
#  [0.88165522, 0.71123707, 0.48312259, 0.        ]]


# x = [[0.    ,     0.    ,     0.    ,     0.        ],
#  [0.51824943, 0.51824943, 0.48312259, 0.48312259],
#  [0.63817193, 0.71123707, 0.63817193, 0.71123707],
#  [0.88165522, 0.79407987, 0.79407987, 0.88165522]]

# x_ind = [[0, 1, 2, 3],
#  [1, 0, 3, 2],
#  [2, 3, 0, 1],
#  [3, 2, 1, 0]]


# print(np.argsort(x_before))

# x = np.array(
# [[0.        , 0.        , 0.        , 0.        ],
#  [0.41026109, 0.83494465, 0.29984271, 0.29984271],
#  [0.67862496, 0.94845008, 0.67862496, 0.41026109],
#  [1.00842187, 1.00842187, 0.83494465, 0.94845008]])

# print(np.sort(x,0).T)

#%%
# import numpy as np

# arr = np.array([5, 3, 8, 2, 6, 1])
# print(arr)
# # ë°°ì—´?„ ? •? ¬?•˜?—¬ ?‘??? ?ˆ˜ë¶??„° ?‹œ?‘?•˜?„ë¡? ?ŒŒ?‹°?…˜
# arr_partitioned = np.partition(arr, 2)
# print(arr_partitioned)

# # ?ŒŒ?‹°?…˜?œ ë°°ì—´?—?„œ 3ë²ˆì§¸ë¡? ?‘??? ê°’ì„ ì°¾ìŒ
# third_smallest_value = np.min(arr_partitioned[:3])

# print(third_smallest_value)  # ì¶œë ¥: 3

