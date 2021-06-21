# MADE: Exploration via Maximizing Deviation from Explored Regions

Code for [MADE: Exploration via Maximizing Deviation from Explored Regions](https://arxiv.org/abs/2106.10268)

In this repository, we provide code for MADE algorithm in the paper mentioned above. The corresponding code is in sub-directory `dreamer_made` and `rad_made`. We also provide several example scripts in the directories.

If you find this repository useful for your research, please cite:
```
```


You can also install custom version of `dm_control` to run experiments on `Walker_Run_Sparse` and `Cheetah_Run_Sparse`. You could do this by following command:

```
cd ../envs/dm_control
pip install .
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
pip3 install --user tensorflow-gpu==2.2.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib

# Install custom dm_control environments for walker_run_sparse / cheetah_run_sparse
cd ../envs/dm_control
pip3 install .
```

## Instructions
### Dreamer + MADE
```
python dreamer.py --logdir ./logdir/dmc_pendulum_swingup/dreamer_made/0 --task dmc_pendulum_swingup --seed 0 --beta 0.1
```

## Installation 

All of the dependencies are in the `requirements.txt`. They can be installed manually or with the following command:

```
pip install -r requirements.txt
```

## Acknowledgement
Our code is built on top of the [DrQ](https://github.com/denisyarats/drq) and [Dreamer](https://github.com/danijar/dreamer) repository.
