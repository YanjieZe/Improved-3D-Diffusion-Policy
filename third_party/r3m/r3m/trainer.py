# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T

epsilon = 1e-8
def do_nothing(x): return x

class Trainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_im, b_lang = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = b_im.shape[0]
        b_im_r = b_im.reshape(bs*5, 3, 224, 224)
        alles = model(b_im_r)
        alle = alles.reshape(bs, 5, -1)
        e0 = alle[:, 0]
        eg = alle[:, 1]
        es0 = alle[:, 2]
        es1 = alle[:, 3]
        es2 = alle[:, 4]

        full_loss = 0

        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        l0loss = torch.linalg.norm(alles, ord=0, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        metrics['l0loss'] = l0loss.item()
        full_loss += model.module.l2weight * l2loss
        full_loss += model.module.l1weight * l1loss
 

        t3 = time.time()
        ## Language Predictive Loss
        if model.module.langweight > 0:
            ## Number of negative examples to use for language
            num_neg = model.module.num_negatives

            ## Trains to have G(e_0, e_t, l) be higher than G(e_0, e_<t, l) and G(e*_0, e*_<t, l)
            ## where e* is a different video. For e_t uses e_g, e_1, and e_2

            ## Setting the positive examples
            sim_pos1, _ = model.module.get_reward(e0, eg, b_lang)
            sim_pos2, _ = model.module.get_reward(e0, es1, b_lang)
            sim_pos3, _ = model.module.get_reward(e0, es2, b_lang)

            ## Adding e_<t as the first negative example
            sim_negs1 = []
            sim_negs2 = []
            sim_negs3 = []
            sim_negs1.append(model.module.get_reward(e0, e0, b_lang)[0])
            sim_negs2.append(model.module.get_reward(e0, es0, b_lang)[0])
            sim_negs3.append(model.module.get_reward(e0, es1, b_lang)[0])

            ## For the specified number of negative examples from other videos
            ## add e* as a negative
            for _ in range(num_neg):
                negvidid = torch.randperm(e0.size()[0])
                sim_negs1.append(model.module.get_reward(e0[negvidid], eg[negvidid], b_lang)[0])
                negvidid = torch.randperm(e0.size()[0])
                sim_negs2.append(model.module.get_reward(e0[negvidid], es1[negvidid], b_lang)[0])
                negvidid = torch.randperm(e0.size()[0])
                sim_negs3.append(model.module.get_reward(e0[negvidid], es2[negvidid], b_lang)[0])
            sim_negs1 = torch.stack(sim_negs1, -1)
            sim_negs_exp1 = torch.exp(sim_negs1)
            sim_negs2 = torch.stack(sim_negs2, -1)
            sim_negs_exp2 = torch.exp(sim_negs2)
            sim_negs3 = torch.stack(sim_negs3, -1)
            sim_negs_exp3 = torch.exp(sim_negs3)

            ## Compute InfoNCE loss
            rewloss1 = -torch.log(epsilon + (torch.exp(sim_pos1) / (epsilon + torch.exp(sim_pos1) + sim_negs_exp1.sum(-1))))
            rewloss2 = -torch.log(epsilon + (torch.exp(sim_pos2) / (epsilon + torch.exp(sim_pos2) + sim_negs_exp2.sum(-1))))
            rewloss3 = -torch.log(epsilon + (torch.exp(sim_pos3) / (epsilon + torch.exp(sim_pos3) + sim_negs_exp3.sum(-1))))
            rewloss = (rewloss1 + rewloss2 + rewloss3) / 3

            ### Mask out videos without language
            with torch.no_grad():
                mask = torch.FloatTensor([1.0 * (b != "") for b in b_lang]).cuda()
            rewloss = rewloss * mask
            rewloss = rewloss.mean()
            lacc1 = (1.0 * (sim_negs1.max(-1)[0] < sim_pos1)).mean()
            lacc2 = (1.0 * (sim_negs2.max(-1)[0] < sim_pos2)).mean()
            lacc3 = (1.0 * (sim_negs3.max(-1)[0] < sim_pos3)).mean()
            metrics['rewloss'] = rewloss.item()
            metrics['rewacc1'] = lacc1.item()
            metrics['rewacc2'] = lacc2.item()
            metrics['rewacc3'] = lacc3.item()
            full_loss += model.module.langweight * rewloss

        t5 = time.time()
        ## Within Video TCN Loss
        if model.module.tcnweight > 0:
            ## Number of negative video examples to use
            num_neg_v = model.module.num_negatives

            ## Computing distance from t0-t2, t1-t2, t1-t0
            sim_0_2 = model.module.sim(es2, es0) 
            sim_1_2 = model.module.sim(es2, es1)
            sim_0_1 = model.module.sim(es1, es0)

            ## For the specified number of negatives from other videos
            ## Add it as a negative
            neg2 = []
            neg0 = []
            for _ in range(num_neg_v):
                es0_shuf = es0[torch.randperm(es0.size()[0])]
                es2_shuf = es2[torch.randperm(es2.size()[0])]
                neg0.append(model.module.sim(es0, es0_shuf))
                neg2.append(model.module.sim(es2, es2_shuf))
            neg0 = torch.stack(neg0, -1)
            neg2 = torch.stack(neg2, -1)

            ## TCN Loss
            smoothloss1 = -torch.log(epsilon + (torch.exp(sim_1_2) / (epsilon + torch.exp(sim_0_2) + torch.exp(sim_1_2) + torch.exp(neg2).sum(-1))))
            smoothloss2 = -torch.log(epsilon + (torch.exp(sim_0_1) / (epsilon + torch.exp(sim_0_1) + torch.exp(sim_0_2) + torch.exp(neg0).sum(-1))))
            smoothloss = ((smoothloss1 + smoothloss2) / 2.0).mean()
            a_state = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean()
            metrics['tcnloss'] = smoothloss.item()
            metrics['aligned'] = a_state.item()
            full_loss += model.module.tcnweight * smoothloss

        metrics['full_loss'] = full_loss.item()
        
        t6 = time.time()
        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()

        t7 = time.time()
        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP tine {t3-t2}, Lang time {t5-t3}, TCN time {t6-t5}, Backprop time {t7-t6}"
        return metrics, st