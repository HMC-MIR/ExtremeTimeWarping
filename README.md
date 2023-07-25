# ExtremeTimeWarping ⏭︎

Dynamic time warping (DTW) is a dynamic programming algorithm used determine the optimal alignment between two sequences. Though designed to handle an unknown amount of time warping, we have found both in practice and anecdotally in conversation with other researchers that it often performs poorly when the two aligned sequences differ greatly in duration length. This paper studies the effect of time warping severity on the performance of DTW in a systematic manner and experimentally explores several ways to improve the robustness of DTW to varying levels of time warping. To make our study concrete, we will focus on an audio-audio alignment scenario in which the goal is to accurately estimate the temporal alignment between two different audio recordings of the same piece of music (e.g. two different piano performances of a composition)

## Environment setup 🏗️

Create a new enviornment from this repo's yml file using any package manager.

```console
$ micromamba create -f env.yml
```

## Running experiments 🧪

```console
python3 02_run_experiment.py --batch train_toy \
    --output experiments_toy \
    --dtw --down --up --adapt --select
```
