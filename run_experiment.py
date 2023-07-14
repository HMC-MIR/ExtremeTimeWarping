from dtw_algorithm import get_accum_cost_and_steps, get_path
import numpy as np
import librosa as lb
import os.path
import os
from pathlib import Path
import pickle
import multiprocessing as mp
import time
import argparse
import tqdm
from datetime import timedelta

N_CORES = 32

def alignDTW(chroma1, chroma2, algorithm, steps, weights, warp_max, subsequence, outfile):

    times = []
    times.append(time.time())

    F1 = np.load(chroma1) # 12 x N
    F2 = np.load(chroma2) # 12 x M
    N, M = F1.shape[1], F2.shape[1] # we assume that N >= M
    times.append(time.time())

    # apply downsampling or adaptive weights
    if algorithm == "downsampleQuantized" or algorithm == "DTW2_downsampleQuantized":
        # we wish to only select M columns of of F1 to get (12 x M)
        index = [int(round(x)) for x in np.linspace(0, N-1, M)]
        F1 = F1[:, index]
    elif algorithm == "downsampleInterpolate" or algorithm == "DTW2_downsampleInterpolate":
        # we want to multiply matrix (12 x N) by (N x M) to get (12 x M)
        transform = np.zeros((N, M))
        index = np.linspace(0, N-1, M) # M indices evenly spaced between [0, N-1]
        for col, index in enumerate(index):
            # at column m, insert weight RIGHT at position ROW and LEFT at position ROW+1
            row = int(index)
            right = index - int(index)
            left = 1 - right
            # if we are at the last row, insert weight 1
            if row + 1 == N:
                transform[row, col] = 1
                continue
            transform[row, col] = left
            transform[row+1, col] = right
        F1 = F1 @ transform
    elif algorithm == "upsampleQuantized" or algorithm == "DTW2_upsampleQuantized":
        index = [int(x) for x in np.linspace(0, M-1, N)]
        F2 = F2[:, index]
    elif algorithm == "upsampleInterpolate" or "DTW2_upsampleInterpolate":
        # we want to multiply matrix (12 x M) by (M x N) to get (12 x N)
        transform = np.zeros((M, N))
        index = np.linspace(0, M-1, N) # N indices evenly spaced between [0, M-1]
        for col, index in enumerate(index):
            # at column N, insert weight RIGHT at position ROW and LEFT at position ROW+1
            row = int(index)
            right = index - int(index)
            left = 1 - right
            # if we are at the last row, insert weight 1
            if row + 1 == M:
                transform[row, col] = 1
                continue
            transform[row, col] = left
            transform[row+1, col] = right
        F2 = F2 @ transform
    elif algorithm == "adaptiveWeight1":
        weights = np.array([N/M, 1])
    elif algorithm == "adaptiveWeight2":
        weights = np.array([N/M, 1, 1 + N/M])
    times.append(time.time())

    # compute cost matrix
    C = 1 - F1.T @ F2
    times.append(time.time())

    # run DTW algorithm
    x_steps = steps[:,0].astype(np.uint32) # horizontal steps
    y_steps = steps[:,1].astype(np.uint32) # veritcal steps
    params = {'x_steps': x_steps, 'y_steps': y_steps, 'weights': weights, 'subsequence': subsequence}
    D, s = get_accum_cost_and_steps(C, params, warp_max)
    times.append(time.time())

    # retrieve paths and steps taken
    path, _, _, track_steps = get_path(D, s, params)
    times.append(time.time())

    if algorithm == "downsampleQuantized" or algorithm == "downsampleInterpolate" or algorithm == "DTW2_downsampleQuantized" or algorithm == "DTW2_downsampleInterpolate":
        path[0] = path[0] * N / M
    elif algorithm == "upsampleQuantized" or algorithm == "upsampleInterpolate" or algorithm == "DTW2_upsampleQuantized" or algorithm == "DTW2_upsampleInterpolate":
        path[1] = path[1] * M / N

    times.append(time.time())
    if outfile:
        with open(outfile, 'wb') as f:
            pickle.dump(path, f)
        with open(str(outfile).replace(".pkl","_steps.pkl"), 'wb') as f:
            pickle.dump(track_steps, f)

    times.append(time.time())
    return algorithm, np.diff(times)


def get_settings(algorithm):

    steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
    weights = np.array([2,3,3])
    warp_max, subsequence = None, False

    if algorithm == 'DTW2' or algorithm in ['DTW2_downsampleQuantized','DTW2_downsampleInterpolate','DTW2_upsampleQuantized','DTW2_upsampleInterpolate']:
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([1,2,2])
    elif algorithm == 'DTW3':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
    elif algorithm == 'DTW4':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,1])
    elif algorithm == 'DTW5':
        steps = np.array([0,1,1,0]).reshape((-1,2))
        weights = np.array([1,1])
    elif algorithm == 'DTW1_add3':
        steps = np.array([1,1,1,2,2,1,1,3,3,1]).reshape((-1,2))
        weights = np.array([2,3,3,4,4])
    elif algorithm == 'DTW1_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([2,3,3,4,4,5,5])

    elif algorithm == 'adaptiveWeight1':
        steps = np.array([0,1,1,0]).reshape((-1,2))
    elif algorithm == 'adaptiveWeight2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))

    elif algorithm == 'selectiveTransitions2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, warp_max = np.array([1,1,2]), 2
    elif algorithm == 'selectiveTransitions3':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, warp_max = np.array([1,1,2]), 3
    elif algorithm == 'selectiveTransitions4':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, warp_max = np.array([1,1,2]), 4
    elif algorithm == 'selectiveTransitions5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, warp_max = np.array([1,1,2]), 5

    elif algorithm == 'SubDTW1':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,2]), True
    elif algorithm == 'SubDTW2':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights, subsequence = np.array([2,3,3]), True
    elif algorithm == 'SubDTW3':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights, subsequence = np.array([1,2,2]), True
    elif algorithm == 'SubDTW4':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,1]), True
    elif algorithm == 'SubDTW5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, subsequence = np.array([0,1,1]), True
    elif algorithm == 'SubDTW6':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,1]), True
    elif algorithm == 'SubDTW7':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,2]), True

    elif algorithm == 'SubDTW3_add3':
        steps = np.array([1,1,1,2,2,1,1,3,3,1]).reshape((-1,2))
        weights, subsequence = np.array([1,2,2,1,3]), True
    elif algorithm == 'SubDTW6_add3':
        steps = np.array([0,1,1,0,1,1,1,3,3,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,1,1,3]), True
    elif algorithm == 'SubDTW3_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights, subsequence = np.array([1,2,2,1,3,1,4]), True
    elif algorithm == 'SubDTW6_add4':
        steps = np.array([0,1,1,0,1,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights, subsequence = np.array([1,1,1,1,3,1,4]), True

    elif algorithm == 'SubDTW_selectiveTransitions2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warp_max, subsequence = 2, True
    elif algorithm == 'SubDTW_selectiveTransitions3':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warp_max, subsequence = 3, True
    elif algorithm == 'SubDTW_selectiveTransitions4':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warp_max, subsequence = 4, True
    elif algorithm == 'SubDTW_selectiveTransitions5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warp_max, subsequence = 5, True

    elif algorithm == 'DTW2_add3':
        steps = np.array([1,1,1,2,2,1,1,3,3,1]).reshape((-1,2))
        weights = np.array([1,2,2,4,4])
    elif algorithm == 'DTW2_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([1,2,2,4,4,5,5])

    return steps, weights, warp_max, subsequence


def get_jobs_for_benchmark(warp1, warp2, algorithm, args):

    # get directory with chroma features
    featdir1 = Path(f"Mazurkas_median_{warp1}/features/clean")
    featdir2 = Path(f"Mazurkas_median_{warp2}/features/clean")

    # desginate output directory
    outdir = Path(f'{args.output_dir}/{algorithm}_{warp1}_{warp2}')
    outdir.mkdir(parents=True, exist_ok=True)

    # get algorithm settings
    steps, weights, warp_max, subsequence = get_settings(algorithm)

    # loop through entire batch to get list of jobs
    jobs = []
    with open(f'cfg_files/{args.batch}.txt', 'r') as f:
        for line in f:
            x, y = line.strip().split(' ')
            # find location of pre-computed chroma features
            chroma1 = (featdir1 / x).with_suffix('.npy')
            chroma2 = (featdir2 / y).with_suffix('.npy')
            # create unique outfile based on the two performances
            pair = os.path.basename(x) + '__' + os.path.basename(y)
            outfile = (outdir / pair).with_suffix('.pkl')
            if not os.path.exists(outfile):
                jobs.append((chroma1, chroma2, algorithm, steps, weights, warp_max, subsequence, outfile))
    return jobs


def get_jobs_for_all_benchmark(benchmarks, args):
    jobs = []
    for warp1, warp2, algorithm in benchmarks:
        jobs.extend(get_jobs_for_benchmark(warp1, warp2, algorithm, args))
    return jobs

def get_benchmarks(algorithms, subseq=False):
    subseq = '_subseq20' if subseq else ''
    return [[f'{warp1}{subseq}', warp2, algorithm] for algorithm in algorithms
            for warp1, warp2 in [('x1.000', 'x1.000'), ('x1.260', 'x1.000'), ('x1.260', 'x0.794'),
                ('x1.588', 'x0.794'), ('x1.588', 'x0.630'), ('x2.000', 'x0.630'), ('x2.000', 'x0.500')]]
            # for warp1, warp2 in [('x0.500', 'x0.500'), ('x0.630', 'x0.630'), ('x0.794', 'x0.794'),
            #     ('x1.000', 'x1.000'), ('x1.260', 'x1.260'), ('x1.588', 'x1.588'), ('x2.000', 'x2.000')]]


if __name__ == "__main__":

    # python3 run_experiment.py --batch train_toy --output experiments
    # python3 run_experiment.py --batch train_small --output experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default='train_toy', type=str, help="data to run experiments on")
    parser.add_argument("--output_dir", default='experiments', type=str, help="where to save experiments")
    parser.add_argument('--track_steps', action='store_true', help='whether to save steps')
    parser.set_defaults(track_steps=False)
    args = parser.parse_args()

    #benchmarks = get_benchmarks(['DTW1', 'DTW2', 'DTW3', 'DTW4', 'DTW5', 'DTW1_add3', 'DTW1_add4', 'downsampleQuantized', 'downsampleInterpolate', 'adaptiveWeight1', 'adaptiveWeight2', 'selectiveTransitions2','selectiveTransitions3','selectiveTransitions4','selectiveTransitions5'])
    benchmarks = get_benchmarks(['DTW2_add3', 'DTW2_add4', 'DTW2_downsampleQuantized', 'DTW2_downsampleInterpolate', 'DTW2_upsampleQuantized', 'DTW2_upsampleInterpolate'])
    with open(f"cfg_files/{args.batch}.txt", 'r') as f:
        args.num_pairs = sum(1 for _ in f)
    print(f"Running {args.num_pairs * len(benchmarks)} experiments for {args.batch} 🤯")

    jobs = get_jobs_for_all_benchmark(benchmarks, args)
    print(f"Collected {len(jobs)} pairs to align 🏗️")

    start = time.time()
    with mp.Pool(processes = N_CORES) as pool:
        results = pool.starmap(alignDTW, tqdm.tqdm(jobs, total=len(jobs)))

    Path('times').mkdir(parents=True, exist_ok=True)

    with open(f"times/{args.batch}_times.csv", 'w') as o:
        o.write("algorithm,load_chroma,downsample_adaptive,compute_cost,accum_steps,get_path,upsample,output\n")
        for algorithm, times in results:
            load_chroma, downsample_adaptive, compute_cost, accum_steps, get_path, upsample, output = times
            o.write(f"{algorithm},{load_chroma},{downsample_adaptive},{compute_cost},{accum_steps},{get_path},{upsample},{output}\n")

    end = time.time() - start
    print(f"Entire python script took {str(timedelta(seconds = int(end)))} 💨")



