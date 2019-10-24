import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count


def shuffle_mat_nozerorow(mat_a):
    """
    Shuffle each matrix column independently, making sure there's no row with only zeros.
    This is done inplace on the matrix passed, pass with .copy (should increase time).
    """
    count = 1
    while np.count_nonzero(mat_a.sum(axis=1)) != mat_a.shape[0]:
        np.apply_along_axis(np.random.shuffle, 0, mat_a)
        count += 1
        if count > 1e+04:
            raise Exception('cannnot find a matrix with no row with only zeros')
            break
    return mat_a


def gen_rand_onezeromat(shape: tuple):
    """
    Generates a random one-zero matrix, with no zero-only rows.
    """
    mat_a = np.concatenate((np.ones(shape[1]).reshape(1, shape[1]), np.zeros([shape[0] - 1, shape[1]])), axis=0)
    return shuffle_mat_nozerorow(mat_a)


def gen_pop(_popsize, shape: tuple, seed):
    """
    Returns a list of random one-zero matrices, with no zero-only rows.
    """
    np.random.seed(seed)
    pop = []
    for i in range(_popsize):
        pop.append(gen_rand_onezeromat(shape))
    return pop


def gen_pop_mp(_popsize, shape: tuple):
    """
    Returns a list of random one-zero matrices, with no zero-only rows.
    Multiprocessing implementation
    """
    num_processors = cpu_count()
    p = Pool(processes=num_processors)
    print(f'using {num_processors} processors')

    pop = []
    for i in range(_popsize):
        pop.append(p.apply(gen_rand_onezeromat, args=(shape,)))
    return pop




if __name__ == "__main__":
    shape = (89, 16290)
    ini_pop = gen_pop(10, shape, 42)
    elite_size = 2
    mut_rate = 0.1
    balance_penalty = 1
    generations = 10

    pop_ranks = genetic_algorithm(ini_pop, elite_size, mut_rate, balance_penalty,
                                  generations, nl_en_job_domain, nl_en_editor_domain_quality)

    allpop_ranks = pd.concat(pop_ranks)
    allpop_ranks.groupby('gen')['fitness'].max().plot()
    num_processors = cpu_count() - 1
    p = Pool(processes=num_processors)
    print(f'using {num_processors} processors')

    shape = (3, 5)
    res = []
    for i in range(10):
        res.append(p.apply(gen_rand_onezeromat, args=(shape,)))

    print(res)

    # pop = gen_pop_mp(4,(3,5),42)
    # [print(i) for i in pop]

    # pool.apply(f, args)
    # pool.map(f, iterable)

# def foo(x):
#     return x*2
#
# if __name__ == '__main__':
#     num_processors = cpu_count() - 1
#     p = Pool(processes=num_processors)
#     print(f'using {num_processors} processors')
#
#     res = []
#     for i in range(10):
#         res.append(p.apply(foo, args=(4,)))
#     print(res)
