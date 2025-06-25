from environment import ScheduleEnv
from task.task_generator import generate_tasks
from common.arguments import get_common_args, get_mixer_args
from runner import Runner
from common.rollout import RolloutWorker
from agent.agent import Agents
import pickle
import os
import numpy as np
from collections import defaultdict
from loguru import logger


def load_model_and_evaluate(file_path):
    # 加载固定任务集
    with open(file_path, "rb") as f:
        all_task_sets = pickle.load(f)
    # 只比较qmix和greedy算法
    task_type_list = ['qmix', 'greedy']

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

    # 评估任务集 - 只关注完成率和完成时间
    completion_rates = defaultdict(list)
    avg_completion_times = defaultdict(list)
    total_tasks = len(all_task_sets)

    for i, tasks in enumerate(all_task_sets):
            _, _, _, stats = rolloutWorker.generate_episode(
                evaluate=True,
                tasks=tasks,
                task_type=qmix
            )

            # 记录完成率
            completion_rates[task_type].append(stats["completion_rate"])

            # 记录平均完成时间（仅对已完成任务）
            if stats["total_completed_num"] > 0:
                avg_completion_time = stats["total_completion_time"] / stats["total_completed_num"]
            else:
                avg_completion_time = 0
            avg_completion_times[task_type].append(avg_completion_time)

            logger.info(
                f"任务集 {i + 1}/{total_tasks} | 算法 {task_type} | "
                f"完成率: {stats['completion_rate']:.2%} | "
                f"平均完成时间: {avg_completion_time:.2f}秒"
            )

    # 计算整体平均值
    results = {}
    for task_type in task_type_list:
        avg_completion_rate = np.mean(completion_rates[task_type])
        avg_completion_time = np.mean(avg_completion_times[task_type])

        results[task_type] = {
            "completion_rate": avg_completion_rate,
            "avg_completion_time": avg_completion_time
        }

        logger.info(f"算法 {task_type} 在 {total_tasks} 个任务集上的平均表现:")
        logger.info(f"  平均任务完成率: {avg_completion_rate:.2%}")
        logger.info(f"  平均任务完成时间: {avg_completion_time:.2f}秒")
        logger.info("")

    # 比较结果
    logger.info("QMIX vs Greedy 比较结果:")
    qmix_rate = results['qmix']['completion_rate']
    greedy_rate = results['greedy']['completion_rate']
    qmix_time = results['qmix']['avg_completion_time']
    greedy_time = results['greedy']['avg_completion_time']

    logger.info(f"QMIX 完成率比 Greedy {'高' if qmix_rate > greedy_rate else '低'}: "
                f"{abs(qmix_rate - greedy_rate):.2%}")
    logger.info(f"QMIX 完成时间比 Greedy {'少' if qmix_time < greedy_time else '多'}: "
                f"{abs(qmix_time - greedy_time):.2f}秒")

    return results


if __name__ == "__main__":
    task_file_path = os.environ['PYTHONPATH'] + "\\MARL-QMIX\\MASA-QMIX-master\\task\\task.pkl"
    results = load_model_and_evaluate(task_file_path)

    # 打印最终比较结果
    print("\n最终比较结果:")
    print(f"QMIX 平均完成率: {results['qmix']['completion_rate']:.2%}")
    print(f"Greedy 平均完成率: {results['greedy']['completion_rate']:.2%}")
    print(f"QMIX 平均完成时间: {results['qmix']['avg_completion_time']:.2f}秒")
    print(f"Greedy 平均完成时间: {results['greedy']['avg_completion_time']:.2f}秒")