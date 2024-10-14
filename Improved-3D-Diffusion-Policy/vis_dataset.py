import zarr
import cv2
from termcolor import cprint
import time
from tqdm import tqdm
import visualizer
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="data/box_zarr")


parser.add_argument("--use_img", type=int, default=0)
parser.add_argument("--vis_cloud", type=int, default=0)
parser.add_argument("--use_pc_color", type=int, default=0)
parser.add_argument("--downsample", type=int, default=0)

args = parser.parse_args()
use_img = args.use_img
dataset_path = args.dataset_path
vis_cloud = args.vis_cloud
use_pc_color = args.use_pc_color
downsample = args.downsample

with zarr.open(dataset_path) as zf:
    print(zf.tree())

# get data
if use_img:
    all_img = zf['data/img']
all_point_cloud = zf['data/point_cloud']
all_episode_ends = zf['meta/episode_ends']


    
# devide episodes by episode_ends
for episode_idx, episode_end in enumerate(all_episode_ends):
    if episode_idx == 0:
        if use_img:
            img_episode = all_img[:episode_end]
        
        point_cloud_episode = all_point_cloud[:episode_end]
    else:
        if use_img:
            img_episode = all_img[all_episode_ends[episode_idx-1]:episode_end]
        point_cloud_episode = all_point_cloud[all_episode_ends[episode_idx-1]:episode_end]

    # print(img_episode.shape)
    # print(point_cloud_episode.shape)
    
    save_dir = f"visualizations/{dataset_path}/{episode_idx}"
    if vis_cloud:
        os.makedirs(save_dir, exist_ok=True)
    cprint(f"replay episode {episode_idx}", "green")
    # replay image
    for i in range(point_cloud_episode.shape[0]):
        
        pc = point_cloud_episode[i]

        # downsample
        if downsample:
            num_points = 4096
            idx = np.random.choice(pc.shape[0], num_points, replace=False)
            pc = pc[idx]

        if use_img:
            img = img_episode[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            time.sleep(0.05)
            
        # if vis_cloud and i >= 50:
        if vis_cloud:
            if not use_pc_color:
                pc = pc[:, :3]
            visualizer.visualize_pointcloud(pc, img_path=f"{save_dir}/{i}.png")
            print(f"vis cloud saved to {save_dir}/{i}.png")

        print(f"frame {i}/{point_cloud_episode.shape[0]}")

        # if i == 200:
        #     break
    
    if vis_cloud:
        # to video
        os.system(f"ffmpeg -r 10 -i {save_dir}/%d.png -vcodec mpeg4 -y visualizations/{dataset_path}/{episode_idx}.mp4")

        
        
    




