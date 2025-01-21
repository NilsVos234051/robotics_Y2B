import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
from task10 import OT2Env
import argparse
import os
from clearml import Task
import typing_extensions
from typing_extensions import TypeIs
import tensorboard

# Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# It can also be helpful to include the hyperparameters in the task name
task = Task.init(project_name='Mentor Group M/Group 1', task_name='Experiment1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


os.environ['WANDB_API_KEY'] = '96266883381464de7d6208aa6214a51135f9b646'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

# Initialize Weights & Biases
run = wandb.init(project="OT2-Hyperparameter-Search", reinit=True)

# Initialize the environment
env = OT2Env(render=False)

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=args.gamma,
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",
            )

timesteps = 100000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

# Save the final model
model.save(f"ppo_ot2_lr{args.learning_rate}_ns{args.n_steps}_bs{args.batch_size}")

# Close the environments
env.close()

wandb.finish()