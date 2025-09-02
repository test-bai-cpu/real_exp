import os, sys
import logging
import yaml
from time import time

import numpy as np
import random

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.group_linear_mpc import GroupLinearMPC
from sim.mpc import mpc_utils

from obs_data_parser import ObsDataParser

#### RL model
import torch
from rl.rl_agent import SAC
from rl.utils import load_config
#### -----------------------------------


#### RL model
def preprocess_rl_obs(obs, current_state, robot_vx, robot_vy, goal_pos):
    """ img_obs: A Numpy array with (max_human, 4) in float32.
        Process it into torch tensor with (bs, max_humna*4) in float32.
    """
    obs = obs.copy()
    current_state = current_state.copy()
    current_pos = current_state[:2].reshape(1, -1)
    obs[:, :2] = obs[:, :2] - current_pos
    obs[obs > 1e4] = 0

    obs[:, 2] = obs[:, 2] - robot_vx
    obs[:, 3] = obs[:, 3] - robot_vy

    goal_pos = np.array(goal_pos).reshape(1, -1)
    goal_pos = goal_pos - current_pos
    goal_vx_vy = np.array([-robot_vx, -robot_vy]).reshape(1, -1)
    obs = obs.reshape(1, -1)
    obs = np.concatenate([goal_pos, goal_vx_vy, obs], axis=1)
    return obs


def set_random_seed(seed):
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


if __name__ == "__main__":
    # configue and logs
    args = get_args()
    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    log_fname = os.path.join(args.output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_fname, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('byteflow').setLevel(logging.WARNING)

    check_args(args, logger)

    # which datasets to preload
    yaml_stream = open(args.dset_file, "r")
    yaml_dict = yaml.safe_load(yaml_stream)
    dsets = yaml_dict["datasets"]
    flags = yaml_dict["flags"]
    if not len(dsets) == len(flags):  
        logger.error("datasets file - number of datasets and flags are not equal!")
        raise Exception("datasets file - number of datasets and flags are not equal!")

    envs_arg = []
    for i in range(len(dsets)):
        dset = dsets[i]
        flag = flags[i]
        envs_arg.append((dset, flag))
    args.envs = envs_arg

    if args.dset_file == "datasets_eth.yaml":
        data_file = "eth_ucy_test"
    elif args.dset_file == "datasets_syn.yaml": # synthetic datasets
        data_file = "synthetic_test"
        
    sim = Simulator(args, f"data/{data_file}.json", logger)
    
    ######################### RL model #####################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_config = load_config("rl_config.yaml")
    result_dir = os.path.join(rl_config["result_dir"], args.exp_name)
    rl_agent = SAC(rl_config["state_shape"], rl_config["action_shape"],
                   rl_config["latent_dim"], device)
    with_exploration = False
    if len(args.rl_model_weight) > 0:
        path_to_checkpoint = os.path.join(result_dir, args.rl_model_weight)
        logger.info(f"Load the pretrained model from: {path_to_checkpoint}")
        rl_agent.load_pretrained_agent(path_to_checkpoint)
        with_exploration = False

    train_info = {}
    ########################################################################
    
    sim.case_id_list.sort()

    mpc_config = mpc_utils.parse_config_file("crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)

    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))
    
    for case_id_index in range(500):
        case_id = random.choice(sim.case_id_list)
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        
        ### MPC initialization
        mpc = GroupLinearMPC(mpc_config, args, logger)

        while not done:
            current_state, target, robot_speed, robot_motion_angle = obs_data_parser.get_robot_state(obs)
            robot_vx = robot_speed * np.cos(robot_motion_angle)
            robot_vy = robot_speed * np.sin(robot_motion_angle)
            nearby_human_state = obs_data_parser.get_human_state(obs) ## padding to max_humans, padding with 1e6 (for pos and vel). Human_state is (n, 4): pos_x, pos_y, vel_x, vel_y
            
            # start_time = time()
            
            ############ RL model output the follow_pos ############
            rl_obs = preprocess_rl_obs(nearby_human_state, current_state, robot_vx, robot_vy, sim.goal_pos) ## TODO: can move it outside the loop?
            if with_exploration:
                rl_actions, _, entropies = rl_agent.get_action(torch.FloatTensor(rl_obs).to(device), with_exploration=with_exploration)
            else:
                rl_actions = rl_agent.get_action(torch.FloatTensor(rl_obs).to(device), with_exploration=with_exploration)
            rl_actions = rl_actions.cpu().detach().numpy()

            # Rescale actions. rl_actions is (1, 4): pos_x, pos_y, vel_x, vel_y, and they are all relative values to the robot, both pos and vel
            follow_pos = rl_actions[0, :2].copy()

            follow_pos = follow_pos * max_follow_pos_delta     # Since max_follow_pos_delta > 0
            # revert the relative pos to global pos
            follow_pos = follow_pos + current_state[:2]

            follow_state = np.array([follow_pos[0], follow_pos[1], 0.0, 0.0])
            follow_state = follow_state.reshape(1, -1)
            ########################################################

            # end_time = time()
            # print("RL spend time: ", end_time - start_time)
            
            ############ use fixed way to generate a follow state ##
            # follow_state = obs_data_parser.get_follow_state(obs, robot_motion_angle, target) ## follow_state is (4,): pos_x, pos_y, speed, motion_angle
            ########################################################

            for mpc_steps_in_one_follow_state in range(10):
                
                # start_time = time()
                
                ###### MPC generate action ######
                action_mpc = mpc.get_action(obs, target, follow_state)
                ################################
                
                # end_time = time()
                
                # print("MPC spend time: ", end_time - start_time)

                start_time = time()
                obs, reward, done, info, time_step, info_dict = sim.step(action_mpc, follow_state)
                end_time = time()
                
                # print("Sim spend time: ", end_time - start_time)
                
                if done == True:
                    break