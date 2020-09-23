# Setup Kinova Grasping

### Step 1: Set up Mujoco

#### Install Mujoco v1.50 (Note: the python package of this version works for python 3++)

http://www.mujoco.org/ (Official website)

https://www.roboti.us/index.html (all downloads)

Note: You will need to register a license key in order to get the simulation engine working. Register as a student and you will get 1 year of free license. Non-commercial license costs $500 for a research lab. (We might want to consider that if we are going to use it in the long run)

After you obtain your license key (`mjkey.txt`), place one in `mjpro150/bin`

#### Configuring environment variables for Mujoco

Create a folder named `.mujoco` at your home directory

Put your `mjpro150` folder (NOT `mjpro150_linux`) and a duplicate of your license key `mjkey.txt` inside this folder

Now your `~/.mujoco` directory will have both `mjpro150` folder and `mjkey.txt`

Use your text editor to open `~/.bashrc`, for example in terminal at home directory, type `vim ~/.bashrc`

Copy the following command at the end of the code (change “`graspinglab`” to your computer name)

```
export LD_LIBRARY_PATH="/home/graspinglab/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_MJKEY_PATH="/home/graspinglab/.mujoco/mjkey.txt"
export MUJOCO_PY_MJPRO_PATH="/home/graspinglab/.mujoco/mjpro150"
```

### Step 2: Set up Virtual Environment

Next you will set up your virtual environment for the Python packages needed for the project.

#### Intializing the virtual environment
In the root directory, run:

```
python3 -m venv venv  # create virtual env
source venv/bin/activate
```

#### Setting up the `mujoco-py` package

Next, you will need to install the `mujoco-py` package from OpenAI. You'll need to build the package yourself.

Download the source code from https://github.com/openai/mujoco-py/releases/tag/1.50.1.0 and unzip the downloaded folder somewhere

Then, while still in your virtual environment, install the `mujoco-py` as follows:

```
cd /path/to/your/mujoco-py-1.50.1.0

pip install -e .
```

Ensure that `mujoco-py` is working by opening up a shell with your virtualenv and importing the package:

```
python  # enter python shell

# now in python shell
import mujoco_py  # make sure this import works!
```

After getting `mujoco-py` up and running, you can install the rest of the needed packages:

```
# make sure you're in the root directory of the project!
pip install -r requirements.txt
```

Congrats! Your environment should be up and running now! You can try running the following code as a final sanity check:

```
cd gym-kinova-gripper  # have to be inside this folder for the script to work

python teleop.py
```

### Step 3: Enjoy!
