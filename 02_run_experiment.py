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

N_CORES = mp.cpu_count()

def alignDTW(chroma1, chroma2, algorithm, steps, weights, warp_max, subsequence, outfile):

    if 'adaptiveHop' in algorithm:
        base, version = os.path.split(chroma1)
        version = version.replace(".npy", "")
        base, mazurka = os.path.split(base)
        base = os.path.split(os.path.split(base)[0])[0]
        wav_file = Path(f"{base}/wav_22050_mono/{mazurka}/{version}").with_suffix(".wav")
        y, sr = lb.core.load(wav_file, sr=22050)
        length1 = str(chroma1)[:str(chroma1).index(os.sep)].split("_x")[-1]
        length2 = str(chroma2)[:str(chroma2).index(os.sep)].split("_x")[-1]
        factor = float(length1) / float(length2)
        hop_length = round(512 * factor / 64.0) * 64
        F1 = lb.feature.chroma_cqt(y, sr=sr, hop_length=hop_length, norm=2)
        F2 = np.load(chroma2) # 12 x M
        N, M = F1.shape[1], F2.shape[1]
    else:
        F1 = np.load(chroma1) # 12 x N
        F2 = np.load(chroma2) # 12 x M
        N, M = F1.shape[1], F2.shape[1] # we assume that N >= M

    # apply downsampling or adaptive weights
    if "downsampleQuantized" in algorithm or "adaptiveHopDownsample" in algorithm:
        # we wish to only select M columns of of F1 to get (12 x M)
        index = [int(round(x)) for x in np.linspace(0, N-1, M)]
        F1 = F1[:, index]
    elif "downsampleInterpolate" in algorithm:
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
    elif "upsampleQuantized" in algorithm:
        index = [int(x) for x in np.linspace(0, M-1, N)]
        F2 = F2[:, index]
    elif "upsampleInterpolate" in algorithm:
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

    # compute cost matrix
    C = 1 - F1.T @ F2

    # run DTW algorithm
    x_steps = steps[:,0].astype(np.uint32) # horizontal steps
    y_steps = steps[:,1].astype(np.uint32) # veritcal steps
    params = {'x_steps': x_steps, 'y_steps': y_steps, 'weights': weights, 'subsequence': subsequence}
    D, s = get_accum_cost_and_steps(C, params, warp_max)

    # retrieve paths and steps taken
    path, _, _, track_steps = get_path(D, s, params)

    if "adaptiveHop" in algorithm:
        path[0] = path[0] * factor
    elif "downsample" in algorithm:
        path[0] = path[0] * N / M
    elif "upsample" in algorithm:
        path[1] = path[1] * M / N

    if outfile:
        with open(outfile, 'wb') as f:
            pickle.dump(path, f)
        with open(str(outfile).replace(".pkl","_steps.pkl"), 'wb') as f:
            pickle.dump(track_steps, f)


def get_settings(algorithm):

    steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
    weights = np.array([2,3,3])
    warp_max, subsequence = None, False

    if 'DTW2' in algorithm:
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
    elif algorithm == 'DTW2w2_add3':
        steps = np.array([1,1,1,2,2,1,1,3,3,1]).reshape((-1,2))
        weights = np.array([1,2,2,3,3])
    elif algorithm == 'DTW2_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([1,2,2,4,4,5,5])
    elif algorithm == 'DTW2w2_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([1,2,2,3,3,4,4])

    return steps, weights, warp_max, subsequence


def get_jobs_for_benchmark(warp1, warp2, algorithm, args):

    # get directory with chroma features
    featdir1 = Path(f"median_{warp1}/features/clean")
    featdir2 = Path(f"median_{warp2}/features/clean")

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
    """
    Return a list of tuples containing two different time warp durations and an algorithm.
    """
    subseq = '_subseq20' if subseq else ''
    return [(f'{warp1}{subseq}', warp2, algorithm) for algorithm in algorithms
            for warp1, warp2 in [('x1.000', 'x1.000'), ('x1.260', 'x1.000'), ('x1.260', 'x0.794'),
                ('x1.588', 'x0.794'), ('x1.588', 'x0.630'), ('x2.000', 'x0.630'), ('x2.000', 'x0.500')]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str, help="data to run experiments on")
    parser.add_argument("--output_dir", required=True, type=str, help="where to save experiments")
    parser.add_argument("--dtw", action="store_true", help="run all normal DTW variants")
    parser.add_argument("--down", action="store_true", help="run all downsample variants")
    parser.add_argument("--up", action="store_true", help="run all upsample variants")
    parser.add_argument("--adapt", action="store_true", help="run all adaptiveWeight variants")
    parser.add_argument("--select", action="store_true", help="run all selectiveTransition variants")
    parser.add_argument("--hop", action="store_true", help="run all adaptiveHop variants")
    args = parser.parse_args()

    algos = []
    if args.dtw:
        algos += ['DTW1', 'DTW2', 'DTW3', 'DTW4', 'DTW5', 'DTW1_add3', 'DTW1_add4']
    if args.down:
        algos += ['DTW2_downsampleQuantized', 'DTW2_downsampleInterpolate']
    if args.up:
        algos += ['DTW2_upsampleQuantized', 'DTW2_upsampleInterpolate']
    if args.adapt:
        algos += ['adaptiveWeight1', 'adaptiveWeight2']
    if args.select:
        algos += ['selectiveTransitions2','selectiveTransitions3','selectiveTransitions4','selectiveTransitions5']
    if args.hop:
        algos += ['DTW2_adaptiveHop','DTW2_adaptiveHopDownsample']

    benchmarks = get_benchmarks(algos)

    with open(f"cfg_files/{args.batch}.txt", 'r') as f:
        args.num_pairs = sum(1 for _ in f)
    print(f"Running {args.num_pairs * len(benchmarks)} experiments for {args.batch} 🤯")

    jobs = get_jobs_for_all_benchmark(benchmarks, args)
    print(f"Collected {len(jobs)} pairs to align 🏗️")

    start = time.time()
    with mp.Pool(processes = N_CORES) as pool:
        results = pool.starmap(alignDTW, tqdm.tqdm(jobs, total=len(jobs)))

    duration = time.time() - start
    print(f"Entire python script took {str(timedelta(seconds = int(duration)))} 💨")

