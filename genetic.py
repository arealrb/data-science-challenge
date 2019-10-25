import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
import numpy as np
import zipfile
import re
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count

###############
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


#
# a = gen_pop(4, (6, 8), 42)
# [print(x.sum(axis=1), x.sum(axis=0)) for x in a]

def iec(mat_a, mat_b, bp):
    """
    Inversion exchange crossover.
    Returns a tuple of mat_a and mat_b with column range exchanged and inverted at the breakpoints bp.
    """
    return np.concatenate([mat_a[:, :bp[0]], np.flip(mat_b[:, bp[0]:bp[1]], axis=1), mat_a[:, bp[1]:]], axis=1), \
           np.concatenate([mat_b[:, :bp[0]], np.flip(mat_a[:, bp[0]:bp[1]], axis=1), mat_b[:, bp[1]:]], axis=1)


def rand_iec(mat_a, mat_b):
    """
    Random Inversion exchange crossover.
    iec with random breakpoints.
    """
    bp = np.sort(np.random.choice(np.arange(mat_a.shape[1]), 2, replace=False))
    return iec(mat_a, mat_b, bp)


def mutate(mat_a):
    """
    Swap adjacent columns on a random mutation point mt.
    This is done in place, pass with .copy to avoid.
    """
    mt = np.random.choice(np.arange(mat_a.shape[1]))
    mat_a[:, mt - 1], mat_a[:, mt] = mat_a[:, mt], mat_a[:, mt - 1].copy()
    return mat_a


def mutate_pop(_pop: list, _mut_rate: float):
    """
    Mutate population at given mutation rate.
    This is done in place, pass with .copy to avoid.
    Example:
    pop = gen_pop(10,(5,10),42)
    pop2 = mutate_pop([x.copy() for x in pop],0.2)
    [(a==b).mean() for a,b in zip(pop,pop2)]
    """
    for ind in _pop:
        if np.random.rand() <= _mut_rate:
            mutate(ind)
    return _pop


def quality_per_job(_agent_job, _job_domain, _agent_domain_quality):
    """
    Returns the quality per task in a matrix of agent_jobs assignment
    """
    return (np.matmul(_job_domain, _agent_domain_quality.T) * _agent_job.T).sum(axis=1)


def price_per_job(_agent_job, _job_domain, _agent_domain_skill, _job_words, _price_per_word):
    """
    Returns the price per task in a matrix of agent_jobs assignment
    """
    return (np.matmul(_job_domain, _agent_domain_skill.T) * _agent_job.T).sum(axis=1) * _job_words * _price_per_word


def ind_fit_func(_agent_job, _job_domain, _agent_domain_quality, _balance_penalty):
    """
    Consider quality and job assignment balance for now.
    """
    total_quality = quality_per_job(_agent_job, _job_domain, _agent_domain_quality).sum()
    jobs_per_agent = _agent_job.sum(axis=1)

    # the bigger the range between min and max jobs per agent, the lower this factor
    range_factor = 1 / jobs_per_agent.ptp()

    dist_skew = skew(jobs_per_agent)
    # the bigger the skewness in absolute value of the distribution of jobs per agents, the lower this factor
    skew_factor = 1 - np.abs(dist_skew)

    dist_kurt = kurtosis(jobs_per_agent)
    # the bigger the kurtosis of the distribution of jobs per agents, the lower this factor
    kurt_factor = 0.5 * (2 - dist_kurt)

    balance_factor = range_factor * skew_factor * kurt_factor
    # if _balance_penalty=1, _balance_penalty**(balance_factor)=1;
    # increase _balance_penalty to increase the importance of this factor
    fitness = total_quality * (_balance_penalty ** balance_factor)

    return [fitness, total_quality, balance_factor, range_factor, dist_skew, dist_kurt]


def rank_pop(_pop, _job_domain, _agent_domain_quality, gen, _balance_penalty):
    """
    Ranks a population using hte fitness function ind_fit_func. Returns a dataframe sorted down by fitness.
    """
    colnames = ['index'] + 'fitness, total_quality, balance_factor, range_factor, dist_skew, dist_kurt'.split(', ')
    pop_rank = []
    for i in range(len(_pop)):
        pop_rank.append([i] + ind_fit_func(_pop[i], _job_domain, _agent_domain_quality, _balance_penalty))
    pop_rank_df = pd.DataFrame(pop_rank, columns=colnames).sort_values('fitness', ascending=False)
    pop_rank_df['gen'] = gen
    return pop_rank_df


def rank_pop_mp(_pop, _job_domain, _agent_domain_quality, gen, _balance_penalty, pool):
    """
    Ranks a population using hte fitness function ind_fit_func. Returns a dataframe sorted down by fitness.
    """
    colnames = ['index'] + 'fitness, total_quality, balance_factor, range_factor, dist_skew, dist_kurt'.split(', ')
    pop_rank = []
    for i in range(len(_pop)):
        pop_rank.append([i] + pool.apply(ind_fit_func, args=(_pop[i], _job_domain,
                                                             _agent_domain_quality, _balance_penalty,)))
    pop_rank_df = pd.DataFrame(pop_rank, columns=colnames).sort_values('fitness', ascending=False)
    pop_rank_df['gen'] = gen
    return pop_rank_df


def next_gen(pop, pop_rank_df, elite_size, mut_rate):
    if elite_size % 2 != 0:
        raise Exception('elite_size should be even or pop will decrease')
    next_pop = []
    # pass elite to next generation
    elite_index = pop_rank_df['index'][:elite_size].values
    for i in elite_index:
        next_pop.append(pop[i])

    matingpool_size = pop_rank_df.shape[0] - elite_size
    # select mating pool based on fitness proportion
    fit_arr = pop_rank_df['fitness'].values
    fit_arr = np.array([i if i > 0 else 1 for i in fit_arr])
    fit_prop = fit_arr / fit_arr.sum()
    selection = np.random.choice(pop_rank_df['index'], size=matingpool_size, replace=False, p=fit_prop)
    # crossovers
    for i in range(0, selection.size, 2):
        offsp1, offsp2 = rand_iec(pop[selection[i]], pop[selection[i + 1]])
        next_pop.append(offsp1)
        next_pop.append(offsp2)
    # mutations
    next_pop = mutate_pop(next_pop, mut_rate)
    return next_pop


def genetic_algorithm(_pop, elite_size, mut_rate, balance_penalty,
                      generations, _job_domain, _agent_domain_quality):
    pop_ranks = []
    # pop zero
    pop_rank_df = rank_pop(_pop, _job_domain, _agent_domain_quality, 0, balance_penalty)
    pop_ranks.append(pop_rank_df)

    for gen in range(1, generations):
        _pop = next_gen(_pop, pop_rank_df, elite_size, mut_rate)
        pop_rank_df = rank_pop(_pop, _job_domain, _agent_domain_quality, gen, balance_penalty)
        print(pop_rank_df.head(1))
        pop_ranks.append(pop_rank_df)

    best_ind = _pop[pop_rank_df['index'].values[0]]
    return pop_ranks, best_ind


def genetic_algorithm_mp(_pop, elite_size, mut_rate, balance_penalty,
                         generations, _job_domain, _agent_domain_quality, pool):
    """
    Multiprocessing implementation.
    """
    pop_ranks = []
    # pop zero
    pop_rank_df = rank_pop_mp(_pop, _job_domain, _agent_domain_quality, 0, balance_penalty, pool)
    pop_ranks.append(pop_rank_df)

    for gen in range(1, generations):
        _pop = next_gen(_pop, pop_rank_df, elite_size, mut_rate)
        pop_rank_df = rank_pop_mp(_pop, _job_domain, _agent_domain_quality, gen, balance_penalty, pool)
        print(pop_rank_df.head(1))
        pop_ranks.append(pop_rank_df)

    best_ind = _pop[pop_rank_df['index'].values[0]]
    return pop_ranks, best_ind


def gen_pop_mp(_popsize, shape: tuple, pool):
    """
    Returns a list of random one-zero matrices, with no zero-only rows.
    Multiprocessing implementation
    """
    pop = []
    for i in range(_popsize):
        pop.append(pool.apply(gen_rand_onezeromat, args=(shape,)))
    return pop


if __name__ == "__main__":
    # load data
    archive = zipfile.ZipFile('dataset.zip', 'r')
    clients = pd.read_csv(BytesIO(archive.read('clients.csv')))
    editors = pd.read_csv(BytesIO(archive.read('editors.csv')))
    tasks = pd.read_csv(BytesIO(archive.read('tasks.csv')))
    tickets = pd.read_csv(BytesIO(archive.read('tickets.csv')))
    # clients = pd.read_csv('clients.csv')
    # editors = pd.read_csv('editors.csv')
    # tasks = pd.read_csv('tasks.csv')
    # tickets = pd.read_csv('tickets.csv')
    tickets_clients = pd.merge(left=tickets, right=clients, how='left', left_on='client_id.1', right_on='id')
    tasks_tickets_clients = pd.merge(left=tasks, right=tickets_clients, how='left', left_on='ticket_id',
                                     right_on='id_x')
    editors.drop(editors.columns[0], axis=1, inplace=True)
    editors.reset_index(inplace=True)
    editors_r = editors.sample(frac=1, random_state=42)
    editors_r.reset_index(drop=True, inplace=True)
    editors_r['language_pair'] = tasks_tickets_clients.language_pair.sample(n=418, random_state=42).values
    domains = 'travel,fintech,ecommerce,sports,gamming,health_care'.split(',')
    # nl_en
    nl_en_tasks = tasks_tickets_clients.loc[tasks_tickets_clients['language_pair'] == 'nl_en']
    enc = OneHotEncoder()
    nl_en_jde = enc.fit_transform(nl_en_tasks['domain'][:, None]).toarray()
    # fix col order to ['travel', 'fintech', 'ecommerce', 'sports', 'gamming', 'health_care']
    nl_en_jdedf = pd.DataFrame(nl_en_jde, columns=enc.categories_)
    nl_en_job_domain = nl_en_jdedf[domains].values
    nl_en_job_words = nl_en_tasks['number_words_x'].values
    nl_en_editors = editors_r[editors_r['language_pair'] == 'nl_en']
    nl_en_editor_domain_skill = nl_en_editors[domains].values
    nl_en_editor_domain_quality = pd.DataFrame(nl_en_editor_domain_skill).apply(lambda x: np.mean(x / 5) * (x / 5),
                                                                                axis=1).values
    # data loaded

    # parameters
    shape = (89, 16290)
    pop_size = 50
    elite_size = 8
    mut_rate = 0.01
    balance_penalty = 1
    generations = 500

    # # with MP
    # # open mp pool
    # num_processors = cpu_count()
    # pool = Pool(processes=num_processors)
    # print(f'using {num_processors} processors')
    # print('generating pop')
    # ini_pop = gen_pop_mp(pop_size, shape, pool)
    # print('running ga')
    # pop_ranks, best_ind = genetic_algorithm_mp(ini_pop, elite_size, mut_rate, balance_penalty,
    #                                            generations, nl_en_job_domain, nl_en_editor_domain_quality, pool)
    # pool.close()
    # pool.join()

    # no MP
    ini_pop = gen_pop(pop_size, shape, 42)
    pop_ranks, best_ind = genetic_algorithm(ini_pop, elite_size, mut_rate, balance_penalty,
                                            generations, nl_en_job_domain, nl_en_editor_domain_quality)

    allpop_ranks = pd.concat(pop_ranks)
    # allpop_ranks.groupby('gen')['fitness'].max().plot()
    # plt.show()
    fname = f'results/nl_en_pop{pop_size}_elite{elite_size}' \
            f'_mutrate{mut_rate}_balancep{balance_penalty}_gens{generations}'
    allpop_ranks.groupby('gen').head(1).to_csv(fname+'.csv', index=False)
    pd.DataFrame(best_ind).to_hdf(fname+'.h5', key='data', index=False, header=None)
