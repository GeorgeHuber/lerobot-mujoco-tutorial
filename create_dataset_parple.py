import sys
import signal
import time
import cv2
from argparse import RawTextHelpFormatter
from threading import Thread
import warnings

import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.papras7dof_env import PaprasEnv

import sys
sys.path.append('/home/student/Desktop/')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pdb

# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
SEED = 0 
# SEED = None <- Uncomment this line to randomize the object positions

REPO_NAME = 'papras7dof_pnp'
NUM_DEMO = 1 # Number of demonstrations to collect
ROOT = "./papras7dof_demo" # The root directory to save the demonstrations

TASK_NAME = 'Put mug cup on the plate' 
xml_path = './asset/papras_scene.xml'
# pdb.set_trace()
# Define the environment
PnPEnv = PaprasEnv(xml_path, seed = SEED, state_type = 'joint_angle')

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    # ans = input("Do you want to delete it? (y/n) ")
    # if ans == 'y':
    if True:
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                REPO_NAME,
                root=ROOT,
                robot_type="papras7dof",
                fps=20, # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["action"], # 7 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False # Start recording when the robot starts moving

from PARPLE.configs import BaseConfig
import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
parser.add_argument('--save_dir', type=str, default='demo_data', help='Directory to save the collected data')

follower_config, leader_config, env_config = BaseConfig().parse(parser)
args, _ = parser.parse_known_args()
SAVE_DIR_BASE = args.save_dir

from PARPLE.paprle.teleoperator import Teleoperator
from PARPLE.paprle.follower import Robot
from PARPLE.paprle.leaders import LEADERS_DICT
from PARPLE.paprle.envs import ENV_DICT
from PARPLE.paprle.feedback import Feedback
from PARPLE.paprle.utils.misc import make_episode
from threading import Thread


# create our configurations for collision checking, teleop and env
robot_config, leader_config, env_config = robot_config, leader_config, env_config

TELEOP_DT = robot_config.robot_cfg.teleop_dt = leader_config.teleop_dt
robot = Robot(robot_config)
leader = LEADERS_DICT[leader_config.type](robot, leader_config, env_config, render_mode=env_config.render_leader) # Get signals from teleop devices, outputs joint positions or eef poses as teleop commands.
teleop = Teleoperator(robot, leader_config, env_config, render_mode=env_config.render_teleop) # Solving IK for joint positions if not already given, check collision, and output proper joint positions.
# self.env = ENV_DICT[env_config.name](self.robot, leader_config, env_config, render_mode=env_config.render_env, leader=self.leader) # Actually send joint positions to the robot.
# self.env.vis_info = self.leader.update_vis_info(self.env.vis_info)

# if not env_config.off_feedback:
#     self.feedback = Feedback(self.robot, self.leader, self.teleop, self.env)
#     self.feedback_thread = Thread(target=self.feedback.send_feedback)
#     self.feedback_thread.start()

shutdown = False
def shutdown_handler(sig, frame):
    #@TODO: save dataset and exit gracefully
    print("Shutting down the system..")
    # self.env.close()
    print("🚫🌏 Env closed")
    teleop.close()
    print("🚫🤖 Teleop closed")
    leader.close()
    print("🚫🎮 Leader closed")
    sys.exit()
    
signal.signal(signal.SIGINT, shutdown_handler)

reset = False
teleop.reset(init_env_qpos)
shutdown = leader.launch_init(init_env_qpos)
if shutdown: return
while not leader.is_ready:
    if shutdown: exit(0)
    time.sleep(0.01)
    
initial_command = leader.get_status()
initial_qpos = teleop.step(initial_command, initial=True) # process initial command
self.env.initialize(initial_qpos) # slowly move to initial qpos, using moveit
if TIME_DEBUG: self.log_time('Initialization Time')
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:

    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done: 
            # Save the episode data and reset the environment
            
            self.reset = False
            init_env_qpos = self.env.reset()
            save_dir = make_episode(self.robot_config, self.leader_config, self.env_config, folder_name=SAVE_DIR_BASE)
            self.teleop.reset(init_env_qpos)
            shutdown = self.leader.launch_init(init_env_qpos)  # Wait in the initialize function until the leader is ready (for visionpro and gello)
            if shutdown: return
            while not self.leader.is_ready:
                if self.shutdown: return
                time.sleep(0.01)
            self.leader.close_init()
            command = self.leader.get_status()
            initial_qpos = self.teleop.step(command, initial=True)
            self.env.initialize(initial_qpos)
            if TIME_DEBUG: self.log_time('Reset Time')
            self.leader.require_end = False
\
            dataset.save_episode()
            PnPEnv.reset(seed = SEED)
            episode_id += 1
            
        # Teleoperate the robot and get delta end-effector pose with 
        
        step_dict = {}
        step_dict['obs'] = self.env.get_observation()

        # 1. Get command from leader
        command = self.leader.get_status()
        step_dict['command'] = command
        
        #@TODO: harmonize resets
        action, reset  = PnPEnv.teleop_robot() #pos, rot, gripper_bool
        #@TODO: continuous gripper or binary?
        qposes = self.teleop.step(command)
        
        step_dict['target_qpos'] = qposes
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose, 
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                }
            )
        PnPEnv.render(teleop=True)
        

PnPEnv.env.close_viewer()