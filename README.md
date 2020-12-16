# Learning "near-contact" grasping strategy with Deep Reinforcement Learning

This is an implementation of Deep Deterministic Policy Gradient from Demonstration (DDPGfD) to train a policy to perform "near-contact" grasping tasks, where object's starting position is random within graspable region. We took one "near-contact" strategy from [this paper](https://ieeexplore.ieee.org/document/8968468) as expert demonstration and train a RL controller to handle a variety of objects with random starting position. 

This environment runs on [MuJoCo](http://www.mujoco.org/) with an intergration of [OpenAI gym](https://gym.openai.com/) to facilitate the data collection and traning process. 

Requirements: [Pytorch 1.2.0](https://pytorch.org/) and Python 3.7 
# Installation

Mujoco v 1.50 (Note: the python package of this version works for python 3++)

Mujoco: 
-  http://www.mujoco.org/ (Official website)
-  https://www.roboti.us/index.html (all downloads)
1.  You will need to register a license key in order to get the simulation engine working.
2.  After you obtain your license key (mjkey.txt), place one in /mjpro131/bin
3.  Create a folder named .mujoco at your home directory
Put your mjpro150 folder (NOT mjpro150_linux) and a duplicate of your license key mjkey.txt inside this folder
    -  Now your ~/.mujoco directory will have both mjpro131 folder and mjkey.txt
4.  Use your text editor to open ~/.bashrc, for example in terminal at home directory, type subl ~/.bashrc
5.  Copy the following command at the end of the code (change “graspinglab” to your computer name)
    -  export LD_LIBRARY_PATH="/home/graspinglab/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
    -  export MUJOCO_PY_MJKEY_PATH="/home/graspinglab/.mujoco/mjkey.txt"
    -  export MUJOCO_PY_MJPRO_PATH="/home/graspinglab/.mujoco/mjpro150"
6.  Open up a terminal and use this command: source ~/.bashrc 

Mujoco-py 

Python package developed by OpenAI

Do not try pip install mujoco-py. It will not work.

1.  Download the source code from here:
https://github.com/openai/mujoco-py/releases/tag/1.50.1.0

2.  Untar / unzip the package 
3.  cd mujoco-py-1.50.1.0
4.  pip install -e. Or pip install --user -e. (if you are denied for not having permission) Or pip3 (if your pip is python 2).

Now you can use mujoco with python by…
import mujoco_py

After you have these installed, clone this repository. To check that it is working, run /KinovaGrasping/gym-kinova-gripper/teleop.py. This should show a render of the hand attempting to pick up a shape.

## Instructions
There are seven experiments to run based on the order of training. The variables we modify are the shapes used, the sizes of these shapes used and the orientation of the hand during training. Six of the experiments change the order of training, (ie we modify one of the variables, then the next, then the last) and the last experiments only uses one stage of training (all variables changed at once).

At **kinova_env_gripper.py**, look at *def randomize_all* function. 
Change the arguments of *self.experiment* for different experiment number and stage number accordingly.
For example, to run experiment 1 stage 1,
At line 581, objects = self.experiment(1, 1)  → the first number is experiment number while the second is stage number. 
Run the commands on terminal below for corresponding experiment.

## Commands on terminal
For experiments with multiple commands, run each command and wait for it to complete before running the next command. The label describes what the network is training on for that command.
### Exp1:
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp1_NO_graspclassifier_local_CubeS --saving_dir exp1_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp2:  
-  Small and Big cube for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp2_NO_graspclassifier_local_CubeS --saving_dir exp2_NO_graspclassifier_local_CubeS --shapes CubeS, CubeB --hand_orientation normal
```
-  Small and Big for all shapes for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp2_NO_graspclassifier_local_CubeS --saving_dir exp2_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation normal
```

-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp2_NO_graspclassifier_local_CubeS --saving_dir exp2_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp3:
-  Small and Big cube for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp3_NO_graspclassifier_local_CubeS --saving_dir exp3_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB --hand_orientation normal
```
-  Small and Big cube for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp3_NO_graspclassifier_local_CubeS --saving_dir exp3_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB --hand_orientation random
```
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp3_NO_graspclassifier_local_CubeS --saving_dir exp3_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp4:
-  Small all shapes for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp4_NO_graspclassifier_local_CubeS --saving_dir exp4_NO_graspclassifier_local_CubeS --shapes CubeS,CylinderS,Cube45S,Cone1S,Cone2S,Vase1S,Vase2S, --hand_orientation normal
```
-  Small and Big for all shapes for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp4_NO_graspclassifier_local_CubeS --saving_dir exp4_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation normal
```
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp4_NO_graspclassifier_local_CubeS --saving_dir exp4_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp5:
-  Small all shapes for just the normal orientation
```
python main_DDPGfD.py --tensorboardindex exp5_NO_graspclassifier_local_CubeS --saving_dir exp5_NO_graspclassifier_local_CubeS --shapes CubeS,CylinderS,Cube45S,Cone1S,Cone2S,Vase1S,Vase2S, --hand_orientation normal
```
-  Small all shapes for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp5_NO_graspclassifier_local_CubeS --saving_dir exp5_NO_graspclassifier_local_CubeS --shapes CubeS,CylinderS,Cube45S,Cone1S,Cone2S,Vase1S,Vase2S, --hand_orientation random
```
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp5_NO_graspclassifier_local_CubeS --saving_dir exp5_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp6:
-  Small cube for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp6_NO_graspclassifier_local_CubeS --saving_dir exp6_NO_graspclassifier_local_CubeS --shapes CubeS --hand_orientation random
```
-  Small all shapes for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp6_NO_graspclassifier_local_CubeS --saving_dir exp6_NO_graspclassifier_local_CubeS --shapes CubeS,CylinderS,Cube45S,Cone1S,Cone2S,Vase1S,Vase2S, --hand_orientation random
```
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp6_NO_graspclassifier_local_CubeS --saving_dir exp6_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```
### Exp7:
-  Small cube for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp7_NO_graspclassifier_local_CubeS --saving_dir exp7_NO_graspclassifier_local_CubeS --shapes CubeS --hand_orientation random
```
-  Small and big cubes for 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp7_NO_graspclassifier_local_CubeS --saving_dir exp7_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB --hand_orientation random
```
-  Small and Big for all shapes and 90% of the poses
```
python main_DDPGfD.py --tensorboardindex exp7_NO_graspclassifier_local_CubeS --saving_dir exp7_NO_graspclassifier_local_CubeS --shapes CubeS,CubeB,CylinderS,CylinderB,Cube45S,Cube45B,Cone1S,Cone1B,Cone2S,Cone2B,Vase1S,Vase1B,Vase2S,Vase2B --hand_orientation random
```

## Alternate Use
This may also be used for purposes outside of these experiments, as it contains the kinova grasping environment, which is useful for grasp classification, regrasping, grasp training and many other uses. If you are planning on using this for purposes other than running the experiments described above, what follows is a brief explanation of what some of the key files do and their interaction.
