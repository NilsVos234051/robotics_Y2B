import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
from task10 import OT2Env
import argparse
import os
from clearml import Task

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
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

# # Define hyperparameter configurations
# hyperparameter_configs = [
#     {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "gamma": 0.99, "gae_lambda": 0.95},
#     {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 32, "gamma": 0.95, "gae_lambda": 0.90},
#     {"learning_rate": 5e-4, "n_steps": 4096, "batch_size": 128, "gamma": 0.98, "gae_lambda": 0.92},
#     {"learning_rate": 2e-4, "n_steps": 512, "batch_size": 64, "gamma": 0.97, "gae_lambda": 0.93},
#     {"learning_rate": 1e-3, "n_steps": 2048, "batch_size": 128, "gamma": 0.99, "gae_lambda": 0.99},
# ]

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
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",
            )

# Train the model and track it with WandB
model.learn(
            total_timesteps=500000,
            callback=wandb_callback,
            progress_bar=True,
            )

# Save the final model
model.save(f"ppo_ot2_lr{args.learning_rate}_ns{args.n_steps}_bs{args.batch_size}")

# Close the environments
env.close()

# # Iterate through hyperparameter configurations
# for config in hyperparameter_configs:
#     wandb_config = {
#         "learning_rate": config["learning_rate"],
#         "n_steps": config["n_steps"],
#         "batch_size": config["batch_size"],
#         "gamma": config["gamma"],
#         "gae_lambda": config["gae_lambda"],
#     }
#     wandb.config.update(wandb_config)
#
#     # Initialize the environment
#     env = OT2Env(render=False)
#
#     # Create an evaluation environment
#     eval_env = OT2Env(render=False)
#
#     # Set up the EvalCallback to evaluate the agent periodically
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path="./logs/best_model",
#         log_path="./logs/",
#         eval_freq=10_000,
#         deterministic=True,
#     )
#
#     model = PPO('MlpPolicy', env, verbose=1,
#                 learning_rate=args.learning_rate,
#                 batch_size=args.batch_size,
#                 n_steps=args.n_steps,
#                 n_epochs=args.n_epochs,
#                 tensorboard_log=f"runs/{run.id}", )
#
#     # Train the model and track it with WandB
#     model.learn(
#         total_timesteps=500_000,
#         callback=[eval_callback, WandbCallback()],
#     )
#
#     # Save the final model
#     model.save(f"ppo_ot2_lr{config['learning_rate']}_ns{config['n_steps']}_bs{config['batch_size']}")
#
#     # Close the environments
#     env.close()
#     eval_env.close()

wandb.finish()