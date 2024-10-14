import numpy as np
import diffusion_policy_3d.common.rotation_util as rotation_util
import torch

init_arm_pos = np.array([[ 0.27605826,  0.25689336, -0.12734995],
 [ 0.29785562, -0.27283502, -0.10588663]])
init_arm_quat = np.array([[-0.50385825,  0.04293381,  0.86230873,  0.02659343],
 [-0.51277079,  0.03748328,  0.85770428,  0.00211863]])
init_q14d = np.array([-0.09729076,  0.04804406,  0.03029635, -1.7516746,   0.06275351,  0.,
  0.,         -0.16716751, -0.06796489, -0.12827234, -1.75219428, -0.01397494,
  0.,          0.,        ])

def joint32_to_joint25(joint):
    #  q_upper_body = [0.0, waist_pitch, 0.0, head_pitch, 0.0, head_yaw]
    # used joint: waist 1 + head 2 + arm 5*2 + hand 6*2 = 25
    # full joint: waist 3 + head 3 + arm 7*2 + hand 6*2 = 32
    new_joint = np.zeros(1+2+5*2+6*2)
    # waist
    new_joint[0] = joint[1]
    # head
    new_joint[1] = joint[3]
    new_joint[2] = joint[5]
    # arm
    new_joint[3:3+5] = joint[6:6+5]
    new_joint[3+5:3+5+5] = joint[6+5+2:6+5+2+5]
    # hand
    new_joint[3+5+5:3+5+5+12] = joint[6+5+2+5+2:6+5+2+5+2+12]
    return new_joint

def joint25_to_joint32(new_joint):
    joint = np.zeros(32)
    # waist
    joint[1] = new_joint[0] 
    # head
    joint[3] = new_joint[1]
    joint[5] = new_joint[2]
    # arm
    joint[6:6+5] = new_joint[3:3+5]
    joint[6+5+2:6+5+2+5] = new_joint[3+5:3+5+5]
    # hand
    joint[6+5+2+5+2:6+5+2+5+2+12] = new_joint[3+5+5:3+5+5+12]  

    return joint  

def extract_eef_action(eef_action):
    body_action = [0, eef_action[0], 0, eef_action[1], 0, eef_action[2]]
    arm_pos = eef_action[3:9].reshape(2, 3)
    arm_rot_6d = eef_action[9:21].reshape(2, 6)
    # arm_rot_quat = rotation_util.rotation_6d_to_quaternion(torch.from_numpy(arm_rot_6d))
    # arm_rot_quat = arm_rot_quat.numpy()
    hand_action = eef_action[21:21+12]
    return body_action, arm_pos, arm_rot_6d, hand_action

def extract_abs_eef(delta_pos, delta_rot_6d, abs_pos, abs_quat):
    new_pos = delta_pos + abs_pos
    abs_rot_6d = rotation_util.quaternion_to_rotation_6d(torch.from_numpy(abs_quat)).numpy()
    new_rot_6d =  abs_rot_6d + delta_rot_6d
    new_quat = rotation_util.rotation_6d_to_quaternion(torch.from_numpy(new_rot_6d)).numpy()
    return new_pos, new_quat, new_rot_6d
    
    
