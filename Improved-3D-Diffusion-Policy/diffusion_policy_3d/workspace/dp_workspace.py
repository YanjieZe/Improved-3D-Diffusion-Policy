if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import time
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.policy.diffusion_image_policy import DiffusionImagePolicy
from diffusion_policy_3d.dataset.base_dataset import BaseImageDataset
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.json_logger import JsonLogger
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DPWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        # dataset element: {'obs', 'action'}
        # obs: {'image': (16,3,96,96) with range [0,1],  'agent_pos': (16,2)}
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

  
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            verbose = True
        else:
            verbose = False
        
        
        RUN_VALIDATION = False # reduce time cost
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in tqdm.tqdm(range(cfg.training.num_epochs)):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for batch_idx, batch in enumerate(train_dataloader):
                    # device transfer
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    
                    t2 = time.time()
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

              
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                    
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
              
                step_log['test_mean_score'] = - train_loss
                    
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # stop wandb run
        wandb_run.finish()
    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path()
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        # runner_log = env_runner.run(policy)
        # cprint(f"---------------- Eval Results --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')
        
    def to_jit(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        tag = "latest"
        # tag = "best"
        lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
        
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        lastest_ckpt_path = str(lastest_ckpt_path)
        jit_policy_path = lastest_ckpt_path.replace('.ckpt', '_jit.pt').replace("/checkpoints/", "/jit/")
        jit_policy_dir = os.path.dirname(jit_policy_path)
        os.makedirs(jit_policy_dir, exist_ok=True)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        device = torch.device('cpu')
        
        # configure dataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        state_shape = cfg.task.shape_meta.obs.agent_pos.shape[0]
        img_shape = cfg.task.shape_meta.obs.image.shape
        
        with torch.no_grad():
            obs_dict = {
            'agent_pos': torch.ones((1, policy.n_obs_steps, state_shape), device=device),
            'image': torch.ones((1, policy.n_obs_steps, *img_shape), device=device),
            # 'image': torch.ones((1, policy.n_obs_steps, *img_shape), device=device),
            }
            result = policy(obs_dict)
            traced_policy = torch.jit.trace(policy, obs_dict)
            traced_policy.save(jit_policy_path)
        cprint(f"JIT policy saved to {jit_policy_path}", 'green')
        cprint("input shape for the policy:", 'yellow')
        cprint(f"agent_pos: {obs_dict['agent_pos'].shape}", 'yellow')
        cprint(f"image: {obs_dict['image'].shape}", 'yellow')

        cprint("-----------------------------", "yellow")

        # test the inference speed
        infer_times = 10
        t0 = time.time()
        for i in range(infer_times):
            with torch.no_grad():
                result = traced_policy(obs_dict)
        cprint(f"FPS for JIT policy: {infer_times/(time.time()-t0):.3f}", 'green')

        t0 = time.time()
        for i in range(infer_times):
            with torch.no_grad():
                result = policy(obs_dict)
        cprint(f"FPS for original policy: {infer_times/(time.time()-t0):.3f}", 'green')

        
    def get_model(self, ckpt_path=None):
        cfg = copy.deepcopy(self.cfg)
        
        if ckpt_path is None:
            tag = "latest"
            # tag = "best"
            lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
            
            if lastest_ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
                self.load_checkpoint(path=lastest_ckpt_path)
        else:
            if ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {ckpt_path}", 'magenta')
                self.load_checkpoint(path=ckpt_path)
            else:
                raise ValueError(f"Checkpoint file not found: {ckpt_path}")

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model    
        policy.eval()

        return policy
    
@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):
    workspace = DPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
