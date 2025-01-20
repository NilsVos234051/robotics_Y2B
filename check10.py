from stable_baselines3.common.env_checker import check_env
from task10 import OT2Env

# instantiate your custom environment
wrapped_env = OT2Env()

# Assuming 'wrapped_env' is your wrapped environment instance
check_env(wrapped_env)