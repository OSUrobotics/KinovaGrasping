## Overview
Experiments are run using [main_DDPGfD.py](https://github.com/OSUrobotics/KinovaGrasping/blob/master/gym-kinova-gripper/main_DDPGfD.py), which contains the main episode loop that controls the course of generating controller experience, conducting policy learning, and evaluating grasp trial performance. 


This document serves as a simple way to quickly reference experiment commands. For more information on the commands line structure and the meaning behind each command option, please reference the [KinovaGrasping Wiki](https://github.com/OSUrobotics/KinovaGrasping/wiki/How-to-run-the-code-and-experiments#How-to-run-experiments)!

## Before you run an experiment
First, download the latest copy of the `experiments/` folder from the [Reinforcement Learning Box directory](https://oregonstate.box.com/s/56iyngvuzdjhrjhex9odgmb0x4vg32dd). Place the `experiments/` folder within `KinovaGrasping/gym-kinova-gripper/`. The experiments folder contains the latest controller experience and policies needed to generate the following experiments.

Each of the following commands will run a specific experiment type. Please replace the **`--saving_dir <saving_dir>`** option with your desired saving directory name. 

## Controller data generation
### Constant-Speed Controller:

**Variation Input (Baseline):** Medium Cube, Normal (0 deg.) hand orientation, No Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Constant-Speed_Controller --hand_orientation normal --shapes CubeM --with_orientation_noise False --max_episode 5000 --controller_type naive --mode naive`

### Variable-Speed Controller:

**Variation Input (Baseline):** Medium Cube, Normal (0 deg.) hand orientation, No Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Variable-Speed_Controller --hand_orientation normal --shapes CubeM --with_orientation_noise False --max_episode 5000 --controller_type position-dependent --mode position-dependent`

## Pre-train
**Variation Input (Baseline):** Medium Cube, Normal (0 deg.) hand orientation, No Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Pre-train_Baseline --hand_orientation normal --shapes CubeM --with_orientation_noise False --max_episode 10000 --controller_type policy --mode pre-train`

## Train
The following are examples of training the policy with each of the variation input types. For conducting a training experiment, you must determine the `pre-trained policy`, `agent replay buffer`, and `expert replay buffer` paths, otherwise they will be set to `None`.

### Randomly-initialized agent Baseline:

**Variation Input (Baseline):** Medium Cube, Normal (0 deg.) hand orientation, No Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Random_Init_Train_Baseline --hand_orientation normal --shapes CubeM --with_orientation_noise False --expert_replay_file_path "./experiments/position-dependent/no_noise/no_grasp/" --replay_buffer_sample_size None --max_episode 10000 --controller_type policy --mode train`

### Baseline + HOV:

**Variation Input (Baseline + HOV):** Medium Cube, Normal (0 deg.) hand orientation, With Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Train_Baseline_HOV --hand_orientation normal --shapes CubeM --with_orientation_noise True --expert_replay_file_path "./experiments/position-dependent/with_noise/no_grasp/" --agent_replay_buffer_path "./experiments/pre-train/Pre-train_Baseline/replay_buffer/" --pretrain_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --max_episode 10000 --controller_type policy --mode train`

### Shapes + HOV:

**Variation Input (Shapes + HOV):** Medium Cube, Med. Cylinder, Med. Vase Normal (0 deg.) hand orientation, With Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Train_Shapes_HOV --hand_orientation normal --shapes CubeM,CylinderM,Vase1M --with_orientation_noise True --expert_replay_file_path "./experiments/position-dependent/with_noise/no_grasp/" --agent_replay_buffer_path "./experiments/pre-train/Pre-train_Baseline/replay_buffer/" --pretrain_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --max_episode 10000 --controller_type policy --mode train`

### Sizes + HOV:

**Variation Input (Sizes + HOV):** Small Cube, Med. Cube, Big Cube, Normal (0 deg.) hand orientation, With Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Train_Sizes_HOV --hand_orientation normal --shapes CubeS,CubeM,CubeB --with_orientation_noise True --expert_replay_file_path "./experiments/position-dependent/with_noise/no_grasp/" --agent_replay_buffer_path "./experiments/pre-train/Pre-train_Baseline/replay_buffer/" --pretrain_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --max_episode 10000 --controller_type policy --mode train`

### Orientations + HOV:

**Variation Input (Orientations + HOV):** Medium Cube, Random (randomly select from normal 0 deg., rotated 68 deg., or top 90 deg.) hand orientation, With Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Train_Orient_HOV --hand_orientation random --shapes CubeM --with_orientation_noise True --expert_replay_file_path "./experiments/position-dependent/with_noise/no_grasp/" --agent_replay_buffer_path "./experiments/pre-train/Pre-train_Baseline/replay_buffer/" --pretrain_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --max_episode 10000 --controller_type policy --mode train`

### Kitchen Sink:

**Variation Input (Kitchen Sink):** Small Cube, Medium Cube, Big Cube, Medium Cylinder, Medium Vase, Random (randomly select from normal 0 deg., rotated 68 deg., or top 90 deg.) hand orientation, With Hand Orientation Variation (HOV)

`python main_DDPGfD.py --saving_dir Train_Kitchen_Sink_HOV --hand_orientation random --shapes CubeS,CubeM,CubeB,CylinderM,Vase1M --with_orientation_noise True --expert_replay_file_path "./experiments/position-dependent/with_noise/no_grasp/" --agent_replay_buffer_path "./experiments/pre-train/Pre-train_Baseline/replay_buffer/" --pretrain_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --max_episode 10000 --controller_type policy --mode train`

## Evaluation

### Evaluate policies over the training period with their given variation input type
Evaluate Pre-trained policy: Baseline

`python main_DDPGfD.py --saving_dir Eval_Pre-Trained_Policy --hand_orientation normal --shapes CubeM --test_policy_name "Pre-Trained Policy" --controller_type policy --test_policy_path "./experiments/pre-train/Pre-train_Baseline/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 0 --max_episode 10000 --input_variations Baseline`

Evaluate Trained Randomly-initialized policy: Baseline

`python main_DDPGfD.py --saving_dir Eval_Random_init_Policy --hand_orientation normal --shapes CubeM --test_policy_name "Randomly-initialized Policy" --controller_type policy --test_policy_path "./experiments/train/Random_Init_Train_Baseline/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 0 --max_episode 10000 --input_variations Baseline`

Evaluate Trained Policy: Baseline_HOV

`python main_DDPGfD.py --saving_dir Eval_Baseline_HOV_Policy --hand_orientation normal --shapes CubeM --with_orientation_noise True --test_policy_name "Baseline + HOV Policy" --controller_type policy --test_policy_path "./experiments/train/Baseline_HOV/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 0 --max_episode 10000 --input_variations Baseline_HOV`

Evaluate Trained Policy: Sizes_HOV

`python main_DDPGfD.py --saving_dir Eval_Sizes_HOV_Policy --hand_orientation normal --shapes CubeS,CubeM,CubeB --with_orientation_noise True --test_policy_name "Sizes + HOV Policy" --controller_type policy --test_policy_path "./experiments/train/Sizes_HOV/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 0 --max_episode 10000 --input_variations Sizes_HOV`

Evaluate Trained Policy: Shapes_HOV

`python main_DDPGfD.py --saving_dir Eval_Shapes_HOV_Policy --hand_orientation normal --shapes CubeM,CylinderM,Vase1M --with_orientation_noise True --test_policy_name "Shapes + HOV Policy" --controller_type policy --test_policy_path "./experiments/train/Shapes_HOV/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 0 --max_episode 10000 --input_variations Shapes_HOV`

### Evaluate final (best performing) trained policies given a specific variation input type:
INPUT VARIATION: Baseline

REGIONS OF INTEREST: center, extreme left, extreme right

Pre-trained Policy:

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Pre_train_policy_FINAL --test_policy_name "Orientations + HOV" --controller_type policy --test_policy_path "./experiments/pre-train/Pre-train_Baseline/policy/pre-train_DDPGfD_kinovaGrip" --mode eval --eval_freq 0 --eval_num 400 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`

Randomly-Initialized Policy:

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Random_Init_policy_FINAL --test_policy_name "Orientations + HOV" --controller_type policy --test_policy_path "./experiments/train/Random_Init_Train_Baseline/policy/train_DDPGfD_kinovaGrip" --mode eval --eval_freq 0 --eval_num 400 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`

Policy: Baseline_HOV

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Baseline_HOV_policy_FINAL --test_policy_name "Baseline + HOV" --controller_type policy --test_policy_path "./experiments/train/Baseline_HOV/output/results/" --mode eval --eval_freq 2000 --eval_num 400 --start_episode 8800 --max_episode 8800 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`

Policy: Sizes_HOV

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Sizes_HOV_policy_FINAL --test_policy_name "Sizes + HOV" --controller_type policy --test_policy_path "./experiments/train/Sizes_HOV/policy/train_DDPGfD_kinovaGrip" --mode eval --eval_freq 0 --max_episode 10000 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`

Policy: Shapes_HOV

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Shapes_HOV_policy_FINAL --test_policy_name "Shapes + HOV" --controller_type policy --test_policy_path "./experiments/train/Shapes_HOV/policy/train_DDPGfD_kinovaGrip" --mode eval --eval_freq 0 --eval_num 400 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`

Policy: Orientations_HOV

`python main_DDPGfD.py --saving_dir Eval_Train_IV_Baseline_w_Orientations_HOV_policy_FINAL --test_policy_name "Orientations + HOV" --controller_type policy --test_policy_path "./experiments/train/Orientations_HOV/policy/train_DDPGfD_kinovaGrip" --mode eval --eval_freq 0 --eval_num 400 --input_variations Baseline,Baseline_HOV,Shapes_HOV,Orientations_HOV --regions_of_interest center,extreme_right,extreme_left`
