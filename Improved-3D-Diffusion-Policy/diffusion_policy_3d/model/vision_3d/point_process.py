# provide some torch/numpy implementations for point cloud processing
# @Yanjie Ze

import torch
import numpy as np

__all__ = ["shuffle_point_torch", "pad_point_torch", "uniform_sampling_torch"]

def shuffle_point_numpy(point_cloud):
    B, N, C = point_cloud.shape
    indices = np.random.permutation(N)
    return point_cloud[:, indices]

def pad_point_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    if num_points > N:
        num_pad = num_points - N
        pad_points = np.zeros((B, num_pad, C))
        point_cloud = np.concatenate([point_cloud, pad_points], axis=1)
        point_cloud = shuffle_point_numpy(point_cloud)
    return point_cloud

def uniform_sampling_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    # padd if num_points > N
    if num_points > N:
        return pad_point_numpy(point_cloud, num_points)
    
    # random sampling
    indices = np.random.permutation(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

def shuffle_point_torch(point_cloud):
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]

def pad_point_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros(B, num_pad, C).to(device)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud

def uniform_sampling_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    # padd if num_points > N
    if num_points == N:
        return point_cloud
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)
    
    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

