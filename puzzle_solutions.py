import datetime
import itertools
import numpy as np
import pandas as pd

def transitive_closure(X):
    """
    (x, y), (m, n); For y = m add (x, n) below
    """
    R = set(X)
    while True:
        Ri = R.union(set((x,n) for x,y in R for m,n in R if m == y))
        if Ri == R:
            return R
        R = Ri

        
def traverse(N):
    jumps = range(1,6)
    jump_options = [list(itertools.permutations(jumps, x)) for x in range(N)]
    jump_options = [item for sublist in jump_options for item in sublist]
    return [x for x in jump_options if sum(x)==N]



def merge_intervals(intervals):
    """
    [[1,4], [3,7], [11,17]] -> [[1,7], [11,17]]
    """
    intervals.sort(key=lambda x:x[0])
    result = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)
    return result


def remove_non_primes(l):
    for x in l:
        for y in l:
            if y % x == 0 and x != y:
                l.remove(y) # Mutable lists lovely :)
    return l


def find_opposites(N):
    N = pd.Series(sorted(set(N)))
    vc = N.abs().value_counts() > 1
    return vc.loc[vc].index.to_list()


def sum_expected_vals(n):
    return sum([1 / (1 - x/n) for x in range(n)])


def f4():
    return np.random.randint(1,5)

def f7a():
    x = f4() + f4()
    while x > 7 or x == 2:
        x = f4() + f4()
    return x

def f7b():
    map7 = dict(zip([tuple(z) for z in [[tuple(y) for y in x] for x in np.split(np.array(list(itertools.product(range(1,5), range(1,5)))[:-2]), 7)]], range(1,8)))
    f4_tuple = (f4(), f4())
    for x in map7:
        if f4_tuple in x:
            return map7[x]
    else:
        return f7b()