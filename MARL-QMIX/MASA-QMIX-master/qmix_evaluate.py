from environment import ScheduleEnv
from task.task_generator import generate_tasks
from common.arguments import get_common_args, get_mixer_args
from runner import Runner
from common.rollout import RolloutWorker
from agent.agent import Agents
import pickle
import numpy as np


def load_model_and_evaluate(file_path):
    # 加载固定任务集
    with open(file_path, "rb") as f:
        all_task_sets = pickle.load(f)

    # 设置参数
    args = get_common_args()
    args = get_mixer_args(args)
    env = ScheduleEnv()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    # 初始化 Runner 和模型
    runner = Runner(env, args)
    args.load_model = True  # 确保加载模型
    args.learn = False  # 设置为评估模式
    args.cuda = True  # 根据需求选择是否使用 GPU

    print("是否加载模型（测试必须）：", args.load_model, "是否训练：", args.learn)
    agents = Agents(args)
    rolloutWorker = RolloutWorker(env, agents, args)

    # 评估任务集
    total_wait_times = []
    completion_rates = []

    for i, tasks in enumerate(all_task_sets):
        _, _, _, stats = rolloutWorker.generate_episode(evaluate=True, tasks=tasks)  # 传入固定任务集
        total_wait_times.append(stats["wait_time"])
        completion_rates.append(stats["completion_rate"])
        print(f"Run {i + 1}: Total wait time = {stats['wait_time']}, Completion rate = {stats['completion_rate']:.2%}")

    average_wait_time = np.mean(total_wait_times)
    average_completion_rate = np.mean(completion_rates)

    print(f"\nAverage wait time over {len(all_task_sets)} runs: {average_wait_time}")
    print(f"Average completion rate over {len(all_task_sets)} runs: {average_completion_rate:.2%}")

    return average_wait_time, average_completion_rate


if __name__ == "__main__":
    task_file_path = "task/task.pkl"  # 替换为任务文件路径
    load_model_and_evaluate(task_file_path)