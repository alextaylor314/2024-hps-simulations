import itertools
import numpy as np
from scipy.optimize import minimize

# get all the factors of a number; used to work out block sizes
def get_factors(x):
    factors = []
    for i in range(1, int(np.sqrt(x)) + 1):
        if x % i == 0:
            factors.append(i)
            if i != x // i:
                factors.append(x // i)
    factors.sort()
    return factors

# estimate the error of correlated variables using the blocking algorithm
def blocking(array):
    # get different block sizes to split the array into
    dimension = len(array)
    n_blocks = get_factors(dimension)[1:] # exclude unblocked case
    block_sizes = [dimension // n for n in n_blocks]
    # create a list of errors and errors-of-errors for different block sizes
    array_mean = array.mean()
    error_list = []
    error_of_error_list = []
    for n, bs in zip(n_blocks, block_sizes):
        averages = np.array([array[bs*i : bs*(i+1)].mean() for i in range(n)])
        error = np.sqrt(np.square(averages - array_mean).mean() / (n-1))
        error_list.append(error)
        error_of_error = error / np.sqrt(2*(n-1))
        error_of_error_list.append(error_of_error)
    '''
    Maximising this function (achieved below by minimising its negative)
    gives a metric that defines how well-converged the error estimates are
    at a particular block size, calculated as the number of larger blocks
    whose error ranges overlap with a fitted value constrained to lie within
    the error range of the current block.
    '''
    def block_convergence(x, larger_blocks):
        n_overlap = 0
        for lb in larger_blocks:
            if (x >= lb[1]-lb[2]) and (x <= lb[1]+lb[2]):
                n_overlap += 1
        return n_overlap
    '''
    For each block size, get the degree of convergence by fitting the
    block_convergence function, ie. fitting a value constrained to lie within
    the block's own error bounds so that it also lies within the error bounds
    of as many of the larger blocks as possible.
    '''
    convergence_list = [] # list of convergence values by block size
    to_iterate = np.flip(np.c_[block_sizes, error_list, error_of_error_list],
                         axis=0) # will iterate from smaller to larger
    for block_info in to_iterate:
        size = block_info[0]
        error = block_info[1]
        error_of_error = [2]
        # get list of blocks larger than this one
        args = np.array([i for i in to_iterate if i[0]>size])
        # calculate bounds
        lower_bound = error - error_of_error
        upper_bound = error + error_of_error
        bounds = [(lower_bound, upper_bound)]
        # maximum possible overlap between this block's error range and others
        n_overlap = -minimize(fun=lambda x,lb: -block_convergence(x, lb),
                              x0=error, args=args, bounds=bounds).fun
        convergence_list.append(n_overlap)
    # return the block error that maximises the convergence criterion
    block_size = to_iterate[np.argmax(convergence_list), 0]
    est_error = to_iterate[np.argmax(convergence_list), 1]
    return block_size, est_error

# just get the block size
def block_size(array):
    block_size, error = blocking(array)
    return block_size

# just get the blocking error
def blocking_error(array):
    block_size, error = blocking(array)
    return error
