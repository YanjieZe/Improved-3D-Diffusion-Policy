from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
from scipy.ndimage import zoom

class GR1DexDatasetImage(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=True,
            use_depth=False,
            ):
        super().__init__()
        cprint(f'Loading GR1DexDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth


        buffer_keys = [
            'state', 
            'action',]
        
        if self.use_img:
            buffer_keys.append('img')
        if self.use_depth:
            buffer_keys.append('depth')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
       
        if self.use_img:
            image = sample['img'][:,].astype(np.float32)
        if self.use_depth:
            depth = sample['depth'][:,].astype(np.float32)
            
        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'action': sample['action'].astype(np.float32)}
        if self.use_img:
            data['obs']['image'] = image
        if self.use_depth:
            data['obs']['depth'] = depth
            
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data

