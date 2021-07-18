# MADE: Exploration via Maximizing Deviation from Explored Regions

Code for [MADE: Exploration via Maximizing Deviation from Explored Regions](https://arxiv.org/abs/2106.10268)

In this repository, we provide code for MADE algorithm in the paper mentioned above. The corresponding code is in sub-directory `dreamer_made` and `rad_made`. We also provide several example scripts in the directories.

If you find this repository useful for your research, please cite:
```
@article{zhang2021made,
  title={MADE: Exploration via Maximizing Deviation from Explored Regions},
  author={Zhang, Tianjun and Rashidinejad, Paria and Jiao, Jiantao and Tian, Yuandong and Gonzalez, Joseph and Russell, Stuart},
  journal={arXiv preprint arXiv:2106.10268},
  year={2021}
}
```


You can also install custom version of `dm_control` to run experiments on `Walker_Run_Sparse` and `Cheetah_Run_Sparse`. You could do this by following command:

```
cd ../envs/dm_control
pip install .
```

## Installation 
```
conda env install -f conda_env.yml
```

## Instructions
### MADE (RAD)
```
python train.py env=hopper_hop batch_size=512 action_repeat=2 logdir=runs_rad_made beta_init=0.5
```


# MADE (Dreamer)
Our code is built on top of the [Dreamer](https://github.com/danijar/dreamer) repository.

## Installation

You could install all dependencies by following command:

```
pip install --user tensorflow-gpu==2.2.0
pip install --user tensorflow_probability
pip install --user git+git://github.com/deepmind/dm_control.git
pip install --user pandas
pip install --user matplotlib

# Install custom dm_control environments for walker_run_sparse and cheetah_run_sparse
cd ../envs/dm_control
pip install .
```

## Instructions
### Dreamer + MADE
```
python dreamer.py --logdir ./logdir/dmc_pendulum_swingup/dreamer_made/0 --task dmc_pendulum_swingup --seed 0 --beta 0.1
```

## Acknowledgement
Our code is built on top of the [DrQ](https://github.com/denisyarats/drq), [Dreamer](https://github.com/danijar/dreamer) and [RE3](https://github.com/younggyoseo/RE3)repository.
