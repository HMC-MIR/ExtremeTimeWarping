import numpy as np
cimport numpy as np
cimport cython
import sys
import time

cdef double MAX_FLOAT = float('inf')

@cython.cdivision(True) # prevent checks on ZeroDivisionError
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # prevents extra checks that are required for calling a list relative to the end like mylist[-5])
@cython.nonecheck(False) # prevents checks on isNone
def get_accum_cost_and_steps(Cin, params, WarpMax=None):

    cdef np.ndarray[double, ndim=2] C = np.array(Cin, dtype=np.float64)
    cdef np.ndarray[unsigned int, ndim=1] x_steps = np.array(params['x_steps'], dtype=np.uint32)
    cdef np.ndarray[unsigned int, ndim=1] y_steps = np.array(params['y_steps'], dtype=np.uint32)
    cdef np.ndarray[double, ndim=1] weights = np.array(params['weights'], dtype=np.float64)

    cdef unsigned int num_rows = C.shape[0]
    cdef unsigned int num_cols = C.shape[1]
    cdef unsigned int num_steps = np.size(weights)
    cdef unsigned int max_row_step = max(x_steps)
    cdef unsigned int max_col_step = max(y_steps)

    cdef np.ndarray[unsigned int, ndim=2] steps = np.zeros((num_rows, num_cols), dtype=np.uint32)
    cdef np.ndarray[double, ndim=2] accum_cost = np.ones((max_row_step + num_rows, max_col_step + num_cols), dtype=np.float64) * MAX_FLOAT

    cdef double best_cost, cost
    cdef unsigned int best_index, row, col, i

    if params['subsequence']:
        for col in range(num_cols):
            accum_cost[max_row_step, col + max_col_step] = C[0, col]
    else:
        accum_cost[max_row_step, max_col_step] = C[0,0]

    cdef bint remove_x = False
    cdef bint remove_y = False

    # filling the accumulated cost matrix
    for row in range(max_row_step, num_rows + max_row_step, 1):

        if WarpMax and row % WarpMax == 0 and row > 1:
            remove_x = True

        for col in range(max_col_step, num_cols + max_col_step, 1):

            if WarpMax and col % WarpMax == 0:
                remove_y = True

            best_cost = accum_cost[<unsigned int>row, <unsigned int>col]
            best_index = 0

            # go through each step, find the best one
            for i in range(num_steps):

                if remove_x and x_steps[(i)]==1 and y_steps[(i)]==0:
                    continue
                if remove_y and y_steps[(i)]==1 and x_steps[(i)]==0:
                    continue

                cost = accum_cost[<unsigned int>((row - x_steps[(i)])), <unsigned int>((col - y_steps[(i)]))] + weights[i] * C[<unsigned int>(row - max_row_step), <unsigned int>(col - max_col_step)]

                if cost < best_cost:
                    best_cost = cost
                    best_index = i

            # save the best cost and best cost index
            accum_cost[row, col] = best_cost
            steps[<unsigned int>(row - max_row_step), <unsigned int>(col - max_col_step)] = best_index
            remove_y = False

        remove_x = False

    # return the accumulated cost matrix and steps taken
    return accum_cost[max_row_step:, max_col_step:], steps


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_path(np.ndarray[double, ndim=2] accum_cost, np.ndarray[unsigned int, ndim=2] steps_for_cost, params):

    cdef np.ndarray[unsigned int, ndim=1] x_steps = params['x_steps']
    cdef np.ndarray[unsigned int, ndim=1] y_steps = params['y_steps']
    cdef bint subseq = params['subsequence']

    cdef unsigned int num_rows = accum_cost.shape[0]
    cdef unsigned int num_cols = accum_cost.shape[1]
    cdef unsigned int row = num_rows - 1
    cdef unsigned int col = np.argmin(accum_cost[num_rows - 1, :]) if subseq else num_cols - 1
    cdef unsigned int end_col = col
    cdef double end_cost = accum_cost[row, col]

    cdef unsigned int row_step, col_step, i
    cdef unsigned int num_steps = 1

    # make as large as could need, then chop at the end
    cdef np.ndarray[unsigned int, ndim=2] path = np.zeros((2, num_rows + num_cols), dtype=np.uint32)

    path[0, 0] = row
    path[1, 0] = col

    cdef bint done = (subseq and row == 0) or (row == 0 and col == 0)
    cdef np.ndarray[unsigned int, ndim=1] track_steps = np.zeros(len(x_steps), dtype=np.uint32)

    while not done:

        if accum_cost[row, col] == MAX_FLOAT:
            break

        i = steps_for_cost[row, col]
        track_steps[i] += 1

        row_step = x_steps[i]
        col_step = y_steps[i]

        # backtrack by 1 step
        row = row - row_step
        col = col - col_step

        # add your new location onto the path
        path[0, num_steps] = row
        path[1, num_steps] = col
        num_steps = num_steps + 1

        # check to see if you're done
        done = (subseq and row == 0) or (row == 0 and col == 0)

    return np.fliplr(path[:, 0:num_steps]), end_col, end_cost, track_steps

