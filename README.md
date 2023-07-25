# ExtremeTimeWarping ⏭︎

Dynamic time warping (DTW) is a dynamic programming algorithm used determine the optimal alignment between two sequences. This paper studies the effect of time warping severity on the performance of DTW in a systematic manner and experimentally explores several ways to improve the robustness of DTW to varying levels of time warping.

## Environment setup 🏗️

Create a new enviornment from this repo's yml file using any package manager.

```console
micromamba create -f env.yml
```

## Running experiments 🧪

Run experiments by running the following bash command. Batch options include: `train_toy`, `train_small`, `train_medium`, `train_full` and `test_full`. Experiments results will be saved in the directory specified by `output`.

The following algorithms are run via each flag:
- dtw: DTW1, DTW2, DTW3, DTW4, DTW5, DTW1_add3, DTW1_add4
- down: DTW2_downsampleQuantized, DTW2_downsampleInterpolate
- up: DTW2_upsampleQuantized, DTW2_upsampleInterpolate
- adapt: adaptiveWeight1-2
- select: selectiveTransitions2-5


```console
python3 02_run_experiment.py --batch train_toy \
    --output experiments_toy \
    --dtw --down --up --adapt --select
```

## Results 💽

Alignment results are saved in `results/`.
