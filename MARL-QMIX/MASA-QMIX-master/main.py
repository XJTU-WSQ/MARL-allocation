from environment import ScheduleEnv
from runner import Runner
from common.arguments import get_common_args, get_mixer_args
import sys
import numpy as np
import torch
import os
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


np.random.seed(42)
torch.manual_seed(42)


def marl_agent_wrapper():
    args = get_common_args()
    args = get_mixer_args(args)
    env = ScheduleEnv()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    print("是否加载模型（测试必须）：", args.load_model, "是否训练：", args.learn)
    runner = Runner(env, args)

    if args.log_step_data:
        os.makedirs("episode_logs", exist_ok=True)  # 创建保存路径
    if args.use_tensorboard:
        print(f"TensorBoard logs at: runs")

    if args.learn:
        runner.run()
    else:
        epoch_rewards, _ = runner.evaluate()
        print('The ave_reward of {} is  {}'.format(args.alg, epoch_rewards))


if __name__ == "__main__":
    marl_agent_wrapper()
