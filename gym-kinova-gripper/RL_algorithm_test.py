## Isolate and test certain functionality within RL algorithm ##

import argparse


def test_naive_controller(test_args=None):
    """ Test the naive controller input and output
    test_args: Specific arguments to be altered from default for testing
    """
    print("Testing the Naive Controller")
    # Pass in arguments we would like to test, otherwise arguments are set to default
    # FROM MAIN_DDPG setup_args(args)
    # Call mode naive from main_DDPG
    # Run for set number of episodes
    # Examine output from generated plots
    # Add extra test cases, checks, output analysis where needed


def test_training(test_args=None):
    """ Test the RL policy training input and output
    test_args: Specific arguments to be altered from default for testing
    """
    print("Testing Policy training")
    # Pass in arguments we would like to test, otherwise arguments are set to default
    # FROM MAIN_DDPG setup_args(args)
    # Call mode train from main_DDPG
    # Run for set number of episodes
    # Examine output from generated plots
    # Add extra test cases, checks, output analysis where needed

#def get_test_args(filename) # Get arguments to test with from file
#def test_replay_buffer():   # Test replay buffer
#def test_training_sample(): # Test policy sampling
#def test_training_update(): # Test policy update


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, action='store', default=10)
    parser.add_argument("--test_mode", type=str, action='store', default="train")
    parser.add_argument("--shapes", default='CubeS', action='store', type=str)  # Requested shapes to use
    parser.add_argument("--orientation", type=str, action='store', default="normal")
    parser.add_argument("--policy_name", default="DDPGfD")              # Policy name
    parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0") # OpenAI gym environment name
    args = parser.parse_args()

    print("------ Testing ", args.test_mode," ------")