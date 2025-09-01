#!/bin/bash


python3 main_eval.py --group --record --exp_name react_sfm_eth --rl_model_weight n_samples_0100000 --output-dir exps/results_MPC_RL/react_sfm_eth --dset-file datasets_eth.yaml --follow-weight 1 --collision_radius 0.5 > logs/react_sfm_eth.txt 2>&1
