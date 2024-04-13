# ExtremeTimeWarping ⏭︎

Dynamic time warping (DTW) is a dynamic programming algorithm used determine the optimal alignment between two sequences. This paper studies the effect of time warping severity on the performance of DTW in a systematic manner and experimentally explores several ways to improve the robustness of DTW to varying levels of time warping.

## Environment setup 🏗️

Create a new enviornment from this repo's yaml file using any package manager.

```console
micromamba create -f environment.yaml # create env
micromamba activate DTW # activate env
```

## Running experiments 🧪

After constructing the data by running notebook `01_prepare_data.ipynb`. Run experiments by running the following bash command. Batch options include: `train_toy`, `train_small`, `train_medium`, `train_full` and `test_full`. Experiments results will be saved in the directory specified by `output`.

The following algorithms are run via each flag:
- `dtw`: DTW1, DTW2, DTW3, DTW4, DTW5, DTW1_add3, DTW1_add4
- `down`: DTW2_downsampleQuantized, DTW2_downsampleInterpolate
- `up`: DTW2_upsampleQuantized, DTW2_upsampleInterpolate
- `adapt`: adaptiveWeight1-2
- `select`: selectiveTransitions2-5
- `hop`: DTW2_adaptiveHop, DTW2_adaptiveHopDownsample


```console
python3 02_run_experiment.py --batch train_toy \
    --output_dir experiments_toy \
    --dtw --down --up --adapt --select --hop
```

## Results 💽

Alignment results are saved in `results/`.

## Altering DTW implementation

To alter the original Cython implementation you will need to alter `dtw_algorithm.pyx`. To compile Cython, you will need to run the following command.

```console
python3 setup.py build_ext --inplace
```

If there are errors with importing numpy, manually the path to the package in your environment to the `CFLAG` environment variable.

```console
export CFLAGS="-I /home/apham/ttmp/micromamba/envs/DropDTW/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
```


## Citation

Jittisa Kraprayoon, Austin Pham, and TJ Tsai.  “Improving the Robustness of DTW to Global Time Warping Conditions in Audio Synchronization.”  Applied Sciences, 14(4): 1459, 2024.



### Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. 2144050.  Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
