from dtw_algorithm import DTW_Cost_To_AccumCostAndSteps, DTW_GetPath
import numpy as np
import librosa as lb
import os.path
import os
from pathlib import Path
import pickle
import multiprocessing
import time
from datetime import timedelta
import argparse

N_CORES = 40
FILE_LIMIT = 767789

def alignDTW(featfile1, featfile2, steps, weights, method, WarpMax, outfile, subsequence):

    if 'adaptiveHop' in method:
        base, version = os.path.split(featfile1)
        version = version.replace(".npy", "")
        base, mazurka = os.path.split(base)
        base = os.path.split(os.path.split(base)[0])[0]
        wav_file = Path(f"{base}/wav_22050_mono/{mazurka}/{version}").with_suffix(".wav")
        y, sr = lb.core.load(wav_file, sr=22050)
        length1 = str(featfile1)[:str(featfile1).index(os.sep)].split("_x")[-1]
        length2 = str(featfile2)[:str(featfile2).index(os.sep)].split("_x")[-1]
        factor = float(length1) / float(length2)
        hop_length = round(512 * factor) # / 64.0) * 64
        F1 = lb.feature.chroma_stft(y, sr=sr, hop_length=hop_length, norm=2)
        F2 = np.load(featfile2) # 12 x M
        N, M = F1.shape[1], F2.shape[1]
    else:
        F1 = np.load(featfile1) # 12 x N
        F2 = np.load(featfile2) # 12 x M
        # we assume that N >= M
        N, M = F1.shape[1], F2.shape[1]
    times = []
    times.append(time.time())
    if method == "downsampleQuantized" or "adaptiveHopDownsample" in method:
        # we wish to only select M columns of of F1 to get (12 x M)
        index = [int(round(x)) for x in np.linspace(0, N-1, M)]
        F1 = F1[:, index]
    elif method == "downsampleInterpolate":
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
    elif method == "upsampleQuantized":
        index = [int(x) for x in np.linspace(0, M-1, N)]
        F2 = F2[:, index]
    elif method == "upsampleInterpolate":
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
    elif method == "adaptiveWeight1":
        weights = np.array([N/M, 1])
    elif method == "adaptiveWeight2":
        weights = np.array([N/M, 1, 1 + N/M])

    C = 1 - F1.T @ F2 # cost distance metric
    times.append(time.time())

    dn = steps[:,0].astype(np.uint32)
    dm = steps[:,1].astype(np.uint32)
    parameters = {'dn': dn, 'dm': dm, 'dw': weights, 'SubSequence': subsequence}

    [D, s] = DTW_Cost_To_AccumCostAndSteps(C, parameters, WarpMax)
    times.append(time.time())

    [wp, endCol, endCost], track_steps_dict = DTW_GetPath(D, s, parameters)
    times.append(time.time())

    if method == "downsampleQuantized" or method == "downsampleInterpolate":
        wp[0] = wp[0] * N / M
    elif method == "upsampleQuantized" or method == "upsampleInterpolate":
        wp[1] = wp[1] * M / N
    elif 'adaptiveHop' in method:
        wp[0] = wp[0] * factor

    if outfile:
        with open(outfile, 'wb') as file:
            pickle.dump(wp, file)

    if args.track_steps:
        track_steps_dict['steps'] = steps
        with open(str(outfile).replace(".pkl","_steps.pkl"), 'wb') as fp:
            pickle.dump(track_steps_dict, fp)

    # profile will mark times [before_accum_cost, after_accum_cost, after_path]
    return wp, np.diff(times) if args.profile else wp


def align_batch(querylist, featdir1, featdir2, outdir, steps, weights, method, warpMax, subsequence, args):

    inputs = []
    with open(querylist, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert len(parts) == 2
            featfile1 = (featdir1 / parts[0]).with_suffix('.npy')
            featfile2 = (featdir2 / parts[1]).with_suffix('.npy')
            queryid = os.path.basename(parts[0]) + '__' + os.path.basename(parts[1])
            outfile = (outdir / queryid).with_suffix('.pkl')
            if os.path.exists(outfile):
                args.skip += 1
            else:
                inputs.append((featfile1, featfile2, steps, weights, method, warpMax, outfile, subsequence))
    with multiprocessing.Pool(processes = N_CORES) as pool:
        pool.starmap(alignDTW, inputs)
        # with implicitly calls pool.close() after pool.starmap()


def get_settings(algorithm):

    steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
    weights = np.array([2,3,3])
    method = "DTW"
    warpMax = None
    subsequence = False

    if algorithm == 'DTW2':
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

    elif algorithm == 'DTW1_downsampleQuantized':
        method = "downsampleQuantized"
    elif algorithm == 'DTW1_downsampleInterpolate':
        method = "downsampleInterpolate"
    elif algorithm == 'DTW1_upsampleQuantized':
        method = "upsampleQuantized"
    elif algorithm == 'DTW1_upsampleInterpolate':
        method = "upsampleInterpolate"
    elif algorithm == 'DTW_adaptiveWeight1':
        steps = np.array([0,1,1,0]).reshape((-1,2))
        method = "adaptiveWeight1"
    elif algorithm == 'DTW_adaptiveWeight2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        method = "adaptiveWeight2"

    elif algorithm == 'DTW_selectiveTransitions2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 2
    elif algorithm == 'DTW_selectiveTransitions3':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 3
    elif algorithm == 'DTW_selectiveTransitions4':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 4
    elif algorithm == 'DTW_selectiveTransitions5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 5

    elif algorithm == 'SubDTW1':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        subsequence = True
    elif algorithm == 'SubDTW2':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([2,3,3])
        subsequence = True
    elif algorithm == 'SubDTW3':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([1,2,2])
        subsequence = True
    elif algorithm == 'SubDTW4':
        steps = np.array([1,1,1,2,2,1]).reshape((-1,2))
        weights = np.array([1,1,1])
        subsequence = True
    elif algorithm == 'SubDTW5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([0,1,1])
        subsequence = True
    elif algorithm == 'SubDTW6':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,1])
        subsequence = True
    elif algorithm == 'SubDTW7':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        subsequence = True

    elif algorithm == 'SubDTW3_add3':
        steps = np.array([1,1,1,2,2,1,1,3,3,1]).reshape((-1,2))
        weights = np.array([1,2,2,1,3])
        subsequence = True
    elif algorithm == 'SubDTW6_add3':
        steps = np.array([0,1,1,0,1,1,1,3,3,1]).reshape((-1,2))
        weights = np.array([1,1,1,1,3])
        subsequence = True
    elif algorithm == 'SubDTW3_add4':
        steps = np.array([1,1,1,2,2,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([1,2,2,1,3,1,4])
        subsequence = True
    elif algorithm == 'SubDTW6_add4':
        steps = np.array([0,1,1,0,1,1,1,3,3,1,1,4,4,1]).reshape((-1,2))
        weights = np.array([1,1,1,1,3,1,4])
        subsequence = True

    elif algorithm == 'SubDTW_selectiveTransitions2':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 2
        subsequence = True
    elif algorithm == 'SubDTW_selectiveTransitions3':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 3
        subsequence = True
    elif algorithm == 'SubDTW_selectiveTransitions4':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 4
        subsequence = True
    elif algorithm == 'SubDTW_selectiveTransitions5':
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
        warpMax = 5
        subsequence = True

    elif algorithm == 'DTW1_adaptiveHop':
        method = 'adaptiveHop'
    elif algorithm == 'DTW1_adaptiveHopDownsample':
        method = 'adaptiveHopDownsample'
    elif algorithm == 'DTW3_adaptiveHop':
        method = 'adaptiveHop'
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])
    elif algorithm == 'DTW3_adaptiveHopDownsample':
        method = 'adaptiveHopDownsample'
        steps = np.array([0,1,1,0,1,1]).reshape((-1,2))
        weights = np.array([1,1,2])

    return steps, weights, method, warpMax, subsequence


def run_single_benchmark(warp1, warp2, algorithm, args):
    """
    Inputs: 'train_toy', 'Mazurkas_median_x1.000', 'Mazurkas_median_x1.000', 'DTW1'
    Output: Run specified DTW algorithm for a single system description
    """
    featdir1 = Path(f"Mazurkas_median_{warp1}/features/clean")
    featdir2 = Path(f"Mazurkas_median_{warp2}/features/clean")
    outdir = Path(f'{args.output}/{algorithm}_{warp1}_{warp2}')
    outdir.mkdir(parents=True, exist_ok=True)
    steps, weights, method, warpMax, subsequence = get_settings(algorithm)

    align_batch(f'cfg_files/{args.filelist}.txt', featdir1, featdir2, outdir, steps, weights, method, warpMax, subsequence, args)
    args.completed += args.num_pairs
    args.mod = (args.mod + 1) % 5


def run_multiple_benchmark(benchmarks, args):
    time_lines, args.completed, args.mod, args.skip, start_run = [], 0, 0, 0, time.time()
    for benchmark in benchmarks:
        warp1, warp2, algorithm = benchmark
        start_time = time.time()
        run_single_benchmark(warp1, warp2, algorithm, args)
        total_time = time.time() - start_time
        avg_time = total_time / (args.num_pairs - args.skip) if args.num_pairs != args.skip else None
        print(f"{algorithm} finished for {args.filelist} with {warp1} {warp2} in {total_time:.3f} sec ({avg_time:.3f} avg)")
        print(f"Done with {args.completed} out of {args.total} ({args.completed*100/args.total:.2f}%) skip {args.skip} experiments", end=" ")
        print("🏃", "💨"*args.mod, "\n", sep="")
        time_lines.append([algorithm, warp1, warp2, total_time, avg_time])
        args.skip = 0
    end_run = time.time() - start_run
    with open(args.save_times, 'w') as f:
        for algorithm, warp1, warp2, total_time, avg_time in time_lines:
            f.write(f"{algorithm},{warp1},{warp2},{total_time},{avg_time}\n")
        f.write(f'Total run time for {args.filelist} was {end_run}')
    print(f'Total run time for {args.filelist} was {end_run}')
    print('='*40)

def get_benchmarks(algorithms, subseq=False):
    subseq = '_subseq20' if subseq else ''
    return [[f'{warp1}{subseq}', warp2, algorithm] for algorithm in algorithms
            for warp1, warp2 in [('x1.000', 'x1.000'), ('x1.260', 'x1.000'), ('x1.260', 'x0.794'),
                ('x1.588', 'x0.794'), ('x1.588', 'x0.630'), ('x2.000', 'x0.630'), ('x2.000', 'x0.500')]]


if __name__ == "__main__":

    print(f"Changing open file limit for current user session to {FILE_LIMIT} 😈")
    os.system(f'ulimit -n {FILE_LIMIT}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", default='train_toy', type=str, help="data to run experiments on")
    parser.add_argument("--save_times", default='train_toy_times.txt', type=str, help="file to save exerpiments times")
    parser.add_argument("--output", default='experiments_test', type=str, help="where to save experiments")
    parser.add_argument("--profile", default=None, help="how to save profiling")
    parser.add_argument('--track_steps', action='store_true', help='whether to save steps')
    parser.set_defaults(track_steps=False)
    args = parser.parse_args()

    benchmarks = get_benchmarks(['DTW1', 'DTW2', 'DTW3']) # , 'DTW4', 'DTW5', 'DTW1_add3', 'DTW1_add4', 'DTW1_downsampleQuantized', 'DTW1_downsampleInterpolate', 'DTW1_upsampleQuantized', 'DTW1_upsampleInterpolate', 'DTW_adaptiveWeight1', 'DTW_adaptiveWeight2', 'DTW_selectiveTransitions2','DTW_selectiveTransitions3','DTW_selectiveTransitions4','DTW_selectiveTransitions5'])

    with open(f"cfg_files/{args.filelist}.txt", 'r') as f:
        args.num_pairs = sum(1 for _ in f)

    args.total = args.num_pairs * len(benchmarks)
    print('='*40, f"\n🎉 Running {args.total} experiments for {args.filelist} 🎉\n", sep="")
    run_multiple_benchmark(benchmarks, args)

