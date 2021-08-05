# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:53:41 2021

@author: Stephane Liem Nguyen

We added env compared to the version in 10-armed testbed
"""

from exploration_exploitation_functions import *
import multiprocessing
from functools import partial

def single_run_wrapper(_, env, horizon, **kwargs):
    # first argument will be used for the range(n_runs)..
    return single_run(env, horizon, **kwargs)
    
def init_pool():
    # We use parallelization because all runs are independent between them.
    # https://stackoverflow.com/questions/5236364/how-to-parallelize-list-comprehension-calculations-in-python
    try:
        cpus = multiprocessing.cpu_count() # max(multiprocessing.cpu_count()//2, 2)
    except NotImplementedError:
        cpus = 2   # arbitrary default
    
    return multiprocessing.Pool(processes=cpus)

def multiple_runs(pool, n_runs, env, horizon, **kwargs):
    # https://docs.python.org/3/library/functools.html#functools.partial
    partial_function = partial(single_run_wrapper, env=env, horizon=horizon, **kwargs)
    return zip(*pool.map(partial_function, range(n_runs)))