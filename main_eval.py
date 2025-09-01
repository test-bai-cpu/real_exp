import os, sys
import csv
import logging
import yaml
import numpy as np
import time
import random
import pandas as pd
import pickle

from config import get_args, check_args
from sim.simulator import Simulator
from sim.mpc.ped_nopred_mpc import PedNoPredMPC
from controller.group_linear_mpc import GroupLinearMPC
from controller.crowd_aware_MPC import CrowdAwareMPC
from controller import mpc_utils
from obs_data_parser import ObsDataParser

#### RL model
import torch
from torch.utils.tensorboard import SummaryWriter
from rl.rl_agent import SAC
from rl.trainer import ContinuousSACTrainer
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

#### -----------------------------------

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    ########## Initialize the evaluation results csv file ###########
    if args.dset_file == "datasets.yaml":
        data_file = "eth_ucy_test"
    elif args.dset_file == "datasets_syn.yaml": # synthetic datasets
        data_file = "synthetic_test"
        
    # data_file = "synthetic_test"
    # data_file = "eth_ucy_test"
    # data_file = "eth_ucy_train"
    print("<<<< The args.react are: ", args.react, args)
    sim = Simulator(args, f"data/{data_file}.json", logger)
    os.makedirs(os.path.join(sim.output_dir, "evas"), exist_ok=True)
    eva_res_dir = os.path.join(sim.output_dir, "evas", f"{data_file}_{args.exp_name}.csv")
    headers = [
        "case_id", "start_frame", "success", "fail_reason", "navigation_time", "path_length",
        "path_smoothness", "motion_smoothness", "min_ped_dist", "avg_ped_dist",
        "min_laser_dist", "avg_laser_dist"
    ]
    with open(eva_res_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row
    #################################################################

    ######################### RL model #####################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_config = load_config("rl_config.yaml")
    # result_dir = rl_config["result_dir"]
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
    # np.random.shuffle(sim.case_id_list)

    mpc_config = mpc_utils.parse_config_file("controller/crowd_mpc.config")
    obs_data_parser = ObsDataParser(mpc_config, args)

    # max_follow_pos_delta = rl_config["max_follow_pos_delta"]
    max_follow_pos_delta = (mpc_config.getint('mpc_env', 'mpc_horizon') *
                            mpc_config.getfloat('mpc_env', 'max_speed'))
    
    ######################### Get the test cases want to check ######################
    # fail_case_file = "exps/failed_cases_noreward.csv"
    # fail_case_df = pd.read_csv(fail_case_file)
    # collision_fail_case_ids = fail_case_df[fail_case_df['fail_reason'] == 'Collision']['case_id'].tolist()
    # time_fail_case_ids = fail_case_df[fail_case_df['fail_reason'] == 'Time']['case_id'].tolist()
    #################################################################################
    
    # random_case_id = random.choice(sim.case_id_list)
    # random_case_id = sim.case_id_list[0:20]
    # for case_id in sim.case_id_list[0:20]:
    # for case_id in [2546, 4511, 2874]:
    # for case_id in random_case_id:
    # for case_id in collision_fail_case_ids:
    
    # for case_id in [2105]:
    # for case_id in sim.case_id_list:
    for case_id_index in range(500):
        case_id = random.choice(sim.case_id_list)
        sim.logger.info(f"Now in the case id: {case_id}")
        obs = sim.reset(case_id)
        done = False
        
        ###### MPC initialization ######
        # mpc = CrowdAwareMPC(mpc_config, args.use_a_omega, args.differential)
        mpc = GroupLinearMPC(mpc_config, args, logger)
        ################################

        time_step = 0
        while not done:
            current_state, target, robot_speed, robot_motion_angle = obs_data_parser.get_robot_state(obs)
            robot_vx = robot_speed * np.cos(robot_motion_angle)
            robot_vy = robot_speed * np.sin(robot_motion_angle)
            nearby_human_state = obs_data_parser.get_human_state(obs) ## padding to max_humans, padding with 1e6 (for pos and vel). Human_state is (n, 4): pos_x, pos_y, vel_x, vel_y
            
            # start_time = time.time()
            
            ############ RL model output the follow_pos ############
            rl_obs = preprocess_rl_obs(nearby_human_state, current_state, robot_vx, robot_vy, sim.goal_pos) ## TODO: can move it outside the loop?
            if with_exploration:
                rl_actions, _, entropies = rl_agent.get_action(torch.FloatTensor(rl_obs).to(device), with_exploration=with_exploration)
            else:
                rl_actions = rl_agent.get_action(torch.FloatTensor(rl_obs).to(device), with_exploration=with_exploration)
            rl_actions = rl_actions.cpu().detach().numpy()

            # Rescale actions. rl_actions is (1, 4): pos_x, pos_y, vel_x, vel_y, and they are all relative values to the robot, both pos and vel
            follow_pos = rl_actions[0, :2].copy()
            # follow_vel = rl_actions[0, 2:].copy()
            # print("rl_actions: ", rl_actions)

            ## Now rerange the follow_pos and follow_vel (-1, 1) -> (-3,3). 
            # follow_pos = (follow_pos + 1) * (max_follow_pos_delta + max_follow_pos_delta) - max_follow_pos_delta     # Since max_follow_pos_delta > 0
            follow_pos = follow_pos * max_follow_pos_delta     # Since max_follow_pos_delta > 0
            # revert the relative pos to global pos
            follow_pos = follow_pos + current_state[:2]

            ## Velocity: (-1, 1) -> (-max_rev_speed, max_speed)
            # follow_vel = (follow_vel + 1) * (mpc.max_speed + mpc.max_rev_speed) / 2 - mpc.max_rev_speed     # Since max_rev_speed > 0
            # follow_vel = follow_vel + np.array([robot_vx, robot_vy])

            # follow_speed = np.linalg.norm(follow_vel)
            # follow_motion_angle = np.mod(np.arctan2(follow_vel[1], follow_vel[0]), 2 * np.pi)
            # print("follow pos: ", follow_pos)
            follow_state = np.array([follow_pos[0], follow_pos[1], 0.0, 0.0])
            follow_state = follow_state.reshape(1, -1)
            ########################################################

            # end_time = time.time()
            # print("RL spend time: ", end_time - start_time)
            
            ############ use fixed way to generate a follow state ############
            # follow_state = obs_data_parser.get_follow_state(obs, robot_motion_angle, target) ## follow_state is (4,): pos_x, pos_y, speed, motion_angle
            ########################################################

            for mpc_steps_in_one_follow_state in range(10):
                
                # start_time = time.time()
                
                ###### MPC generate action ######
                # action_mpc, _ = mpc.get_action(obs, current_state, target, nearby_human_state, follow_state)
                action_mpc = mpc.get_action(obs, target, follow_state)
                # print(">>> in Training, action_mpc =", action_mpc)
                ################################
                
                # end_time = time.time()
                
                # print("MPC spend time: ", end_time - start_time)

                start_time = time.time()
                obs, reward, done, info, time_step, info_dict = sim.step(action_mpc, follow_state)
                end_time = time.time()
                
                # print("Sim spend time: ", end_time - start_time)
                
                if done == True:
                    break

        ################# save the robot path and human path #############################
        save_filename = f"{data_file}_{args.exp_name}.pkl"
        save_filepath = os.path.join(sim.output_dir, "evas", save_filename)
        
        existing_data = {}
        if os.path.exists(save_filepath):
            try:
                with open(save_filepath, "rb") as f:
                    existing_data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                existing_data = {}
        
        existing_data[case_id] = sim.save_all_traj.copy()
        
        with open(save_filepath, "wb") as f:
            pickle.dump(existing_data, f)
            logger.info(f"Case {case_id} trajectory appended to {save_filepath}")
        #################################################################################

        ############## save the evaluation results to the csv file ##############
        result_dict = sim.evaluate(output=True)
        with open(eva_res_dir, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                case_id,
                sim.start_frame,
                result_dict["success"],
                result_dict["fail_reason"],
                result_dict["navigation_time"],
                result_dict["path_length"],
                result_dict["path_smoothness"],
                result_dict["motion_smoothness"],
                result_dict["min_ped_dist"],
                result_dict["avg_ped_dist"],
                ])  # Write the header row
        #########################################################################