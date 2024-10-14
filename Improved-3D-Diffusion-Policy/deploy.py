import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import MultiRealSense

zenoh_path="/home/gr1p24ap0049/projects/gr1-dex-real/teleop-zenoh"
sys.path.append(zenoh_path)
from communication import *
from retarget import ArmRetarget


import numpy as np
import torch
from termcolor import cprint

class GR1DexEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=224,
                 num_points=4096,
                 use_waist=False):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        
        self.use_waist = use_waist
        
        # camera
        self.camera = MultiRealSense(use_front_cam=True, # by default we use single cam. but we also support multi-cam
                            front_num_points=num_points,
                            img_size=img_size)

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # robot comm
        self.upbody_comm = UpperBodyCommunication()
        self.hand_comm = HandCommunication()
        self.arm_solver = ArmRetarget("AVP")
    
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            act = action_util.joint25_to_joint32(act)
            
            filtered_act = act.copy()
            filtered_pos = filtered_act[:-12]
            filtered_handpos = filtered_act[-12:]
            if not self.use_waist:
                filtered_pos[0:6] = 0.
            
            self.upbody_comm.set_pos(filtered_pos)
            self.hand_comm.send_hand_cmd(filtered_handpos[6:], filtered_handpos[:6])
            
            
            cam_dict = self.camera()
            self.cloud_array.append(cam_dict['point_cloud'])
            self.color_array.append(cam_dict['color'])
            self.depth_array.append(cam_dict['depth'])
            
            try:
                hand_qpos = self.hand_comm.get_qpos()
            except:
                cprint("fail to fetch hand qpos. use default.", "red")
                hand_qpos = np.ones(12)
            env_qpos = np.concatenate([self.upbody_comm.get_pos(), hand_qpos])
            self.env_qpos_array.append(env_qpos)
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
    
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict
    
    def reset(self, first_init=True):
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
    
    
        # pos init
        qpos_init1 = np.array([-np.pi / 12, 0, 0, -1.6, 0, 0, 0, 
            -np.pi / 12, 0, 0, -1.6, 0, 0, 0])
        qpos_init2 = np.array([-np.pi / 12, 0, 1.5, -1.6, 0, 0, 0, 
                -np.pi / 12, 0, -1.5, -1.6, 0, 0, 0])
        hand_init = np.ones(12)
        # hand_init = np.ones(12) * 0

        if first_init:
            # ======== INIT ==========
            upbody_initpos = np.concatenate([qpos_init2])
            self.upbody_comm.init_set_pos(upbody_initpos)
            self.hand_comm.send_hand_cmd(hand_init[6:], hand_init[:6])

        upbody_initpos = np.concatenate([qpos_init1])
        self.upbody_comm.init_set_pos(upbody_initpos)
        q_14d = upbody_initpos.copy()
            
        body_action = np.zeros(6)
        
        # this is a must for eef pos alignment
        arm_pos, arm_rot_quat = action_util.init_arm_pos, action_util.init_arm_quat
        q_14d = self.arm_solver.ik(q_14d, arm_pos, arm_rot_quat)
        self.upbody_comm.init_set_pos(q_14d)
        time.sleep(2)
        
        print("Robot ready!")
        
        # ======== INIT ==========
        # camera.start()
        cam_dict = self.camera()
        self.color_array.append(cam_dict['color'])
        self.depth_array.append(cam_dict['depth'])
        self.cloud_array.append(cam_dict['point_cloud'])

        try:
            hand_qpos = self.hand_comm.get_qpos()
        except:
            cprint("fail to fetch hand qpos. use default.", "red")
            hand_qpos = np.ones(12)

        env_qpos = np.concatenate([self.upbody_comm.get_pos(), hand_qpos])
        self.env_qpos_array.append(env_qpos)
                        
        self.q_14d = q_14d
        self.body_action = body_action
    

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        obs_cloud = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
            
        return obs_dict


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 224
    num_points = 4096
    use_waist = True
    first_init = True
    record_data = True

    env = GR1DexEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points,
                             use_waist=use_waist)

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    if record_data:
        import h5py
        root_dir = "/home/gr1p24ap0049/projects/gr1-learning-real/"
        save_dir = root_dir + "deploy_dir"
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo.h5"
        color_array = np.array(env.color_array)
        depth_array = np.array(env.depth_array)
        cloud_array = np.array(env.cloud_array)
        qpos_array = np.array(env.qpos_array)
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("color", data=np.array(color_array))
            f.create_dataset("depth", data=np.array(depth_array))
            f.create_dataset("cloud", data=np.array(cloud_array))
            f.create_dataset("qpos", data=np.array(qpos_array))
        
        choice = input("whether to rename: y/n")
        if choice == "y":
            renamed = input("file rename:")
            os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
            new_name = record_file_name.replace("demo.h5", renamed+'.h5')
            cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
        else:
            cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
