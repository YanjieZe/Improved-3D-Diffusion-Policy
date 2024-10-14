# R3M: A Universal Visual Representation for Robot Manipulation

This project studies how to learn generalizable visual representation for robotics from videos of humans and natural language. It contains pre-trained representation on the Ego4D dataset trained in the [R3M paper](https://arxiv.org/abs/2203.12601)

![](https://cs.stanford.edu/~surajn/images/r3m_robot_teaser_2.gif)

## Installation

To install R3M from an existing conda environment, simply run `pip install -e .` from this directory. 

You can alternatively build a fresh conda env from the r3m_base.yaml file [here](https://github.com/facebookresearch/r3m/blob/main/r3m/r3m_base.yaml) and then install from this directory with `pip install -e .`

You can test if it has installed correctly by running `import r3m` from a python shell.

## Using the representation

To use the model in your code simply run:
```
from r3m import load_r3m
r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
```

Further example code to use a pre-trained representation is located in the example [here](https://github.com/facebookresearch/r3m/blob/main/r3m/example.py).

If you have any issue accessing or downloading R3M please contact Suraj Nair: surajn (at) stanford (dot) edu

## Training the representation

To train the representation run:

`python train_representation.py hydra/launcher=local hydra/output=local agent.langweight=1.0 agent.size=50 experiment=r3m_test dataset=ego4d doaug=rctraj agent.l1weight=0.00001 batch_size=16 datapath=<PATH TO PARSED Ego4D DATA> wandbuser=<WEIGHTS AND BIASES USER> wandbproject=<WEIGHTS AND BIASES PROJECT>`
 
Note: For fast training, the Ego4D data loading code assumes that the dataset has been parsed into frames, with a folder for each video clip and frames of the videoclip (resized to [224 x 224]) numbered within the directory (for example `000123.jpg`). It also assumes a file called `manifest.csv` which has a row for each clip, with the path to the clip folder, the clip length, and the natural language pairing for the clip. 
 
## Evaluating the representation with behavior cloning

See the `eval` branch [here](https://github.com/facebookresearch/r3m/tree/eval/evaluation).

## License

R3M is licensed under the MIT license.

## Ackowledgements

Parts of this code are adapted from the DrQV2 [codebase](https://github.com/facebookresearch/drqv2)
