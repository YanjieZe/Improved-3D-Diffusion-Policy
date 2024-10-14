"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import os
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
from termcolor import cprint

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

# allow for detecting segmentation fault
# import faulthandler
# faulthandler.enable()
# cprint("[fault handler enabled]", "cyan")

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
