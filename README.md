# Learning "near-contact" grasping strategy with Deep Reinforcement Learning

This is an implementation of Deep Deterministic Policy Gradient from Demonstration (DDPGfD) to train a policy to perform "near-contact" grasping tasks, where object's starting position is random within graspable region. We took one "near-contact" strategy from [this paper](https://ieeexplore.ieee.org/document/8968468) as expert demonstration and train a RL controller to handle a variety of objects with random starting position. 

This environment runs on [MuJoCo](http://www.mujoco.org/) with an intergration of [OpenAI gym](https://gym.openai.com/) to facilitate the data collection and traning process. 

Requirements: [Pytorch 1.2.0](https://pytorch.org/) and Python 3.7 

## Instructions
There are three experiments to run for two conditions: with and without grasp classifier, in this case we are using state space in global coordinate system. 
At **kinova_env_gripper.py**, look at *def randomize_all* function. 

Change the argument below to the corresponding experiment number and stage number.
For example, to run experiment 1 stage 1,
At line 581, objects = self.experiment(1, 1)  → the first number is experiment number while the second is stage number. 
Run the commands on terminal below for corresponding experiment.

## Commands on terminal
### Experiments without grasp classifier
Experiment 1 stage 1 (varying sizes) 
'''
{
	python main_DDPGfD.py --tensorboardindex exp1s1_wo_graspclassifier --saving_dir exp1s1_wo_graspclassifier
}
'''

Experiment 1 stage 2 (varying shapes)
'''
python main_DDPGfD.py --tensorboardindex exp1s2_wo_graspclassifier --saving_dir exp1s2_wo_graspclassifier
'''

Experiment 2 stage 1 (varying shapes)
'''
python main_DDPGfD.py --tensorboardindex exp2s1_wo_graspclassifier --saving_dir exp2s1_wo_graspclassifier
'''

Experiment 2 stage 2 (varying sizes)
'''
python main_DDPGfD.py --tensorboardindex exp2s2_wo_graspclassifier --saving_dir exp2s2_wo_graspclassifier
'''

Experiment 3 (all objects) 
'''
python main_DDPGfD.py --tensorboardindex exp3_wo_graspclassifier --saving_dir exp3_wo_graspclassifier
'''

### Experiments with grasp classifier
Experiment 1 stage 1 (varying sizes) 
'''
python main_DDPGfD.py --tensorboardindex exp1s1_w_graspclassifier --saving_dir exp1s1_w_graspclassifier
'''

Experiment 1 stage 2 (varying shapes)
'''
python main_DDPGfD.py --tensorboardindex exp1s2_w_graspclassifier --saving_dir exp1s2_w_graspclassifier
'''

Experiment 2 stage 1 (varying shapes)
'''
python main_DDPGfD.py --tensorboardindex exp2s1_w_graspclassifier --saving_dir exp2s1_w_graspclassifier
'''

Experiment 2 stage 2 (varying sizes)
'''
python main_DDPGfD.py --tensorboardindex exp2s2_w_graspclassifier --saving_dir exp2s2_w_graspclassifier
'''

Experiment 3 (all objects)
'''
python main_DDPGfD.py --tensorboardindex exp3_w_graspclassifier --saving_dir exp3_w_graspclassifier
'''