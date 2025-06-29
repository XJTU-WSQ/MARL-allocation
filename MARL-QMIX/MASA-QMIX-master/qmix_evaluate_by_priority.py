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
    avg_wait_times = defaultdict(list)
    completion_rates = defaultdict(list)
    max_wait_times = defaultdict(list)
    allocated_rates = defaultdict(list)
    avg_service_time = defaultdict(list)
    avg_completed_time = defaultdict(list)



    for i, tasks in enumerate(all_task_sets[:50]):
        
        _, _, _, stats = rolloutWorker.generate_episode(evaluate=True, 
                                                tasks=tasks, task_type='qmix')  # 传入固定任务集

        for task_type in [0,1,2,3,4,5]:
            idx = np.array([t[3] for t in tasks])==task_type
            
            
            tasks_allocated_part = np.array(stats['stats_dict'][119]['tasks_allocated'])[idx]
            tasks_completed_part = np.array(stats['stats_dict'][119]['tasks_completed'])[idx]
            wait_times_part = np.array(stats['stats_dict'][119]['time_wait'])[idx]
            service_times_part = np.array(stats['stats_dict'][119]['service_time'])[idx]
            time_on_road_part = np.array(stats['stats_dict'][119]['time_on_road'])[idx]
            completed_tasks_time_part = wait_times_part + time_on_road_part + service_times_part

            
            completion_rates[task_type].append(tasks_completed_part.sum()/tasks_completed_part.shape[0])
            allocated_rates[task_type].append(tasks_allocated_part.sum()/tasks_allocated_part.shape[0])
            avg_wait_times[task_type].append(wait_times_part.mean())
            avg_service_time[task_type].append(service_times_part.mean())
            avg_completed_time[task_type].append(completed_tasks_time_part.mean())
            
            logger.info(f"Run {task_type} {i + 1}: allocated_rate = {allocated_rates[task_type][-1]:.2%}, completion_rate = {completion_rates[task_type][-1]:.2%}")
            logger.info(f"Run {task_type} {i + 1}: AVG wait time = {avg_wait_times[task_type][-1]:.2}, AVG service time = {avg_service_time[task_type][-1]:.2}, AVG completed time = {avg_completed_time[task_type][-1]:.2}")
    for task_type in [0,1,2,3,4,5]:
        average_wait_time = np.mean(avg_wait_times[task_type])
        average_max_wait_times = np.mean(max_wait_times[task_type])
        average_completion_rate = np.mean(completion_rates[task_type])
        average_allocated_rate = np.mean(allocated_rates[task_type])
        average_service_time = np.mean(avg_service_time[task_type])
        average_completed_time = np.mean(avg_completed_time[task_type])
        
        logger.info(f"Average wait time over {len(all_task_sets)} runs with {task_type}: {average_wait_time:.2}")
        logger.info(f"Average completion rate over {len(all_task_sets)} runs with {task_type}: {average_completion_rate:.2%}")
        logger.info(f"Average allocated rate over {len(all_task_sets)} runs with {task_type}: {average_allocated_rate:.2%}")
        logger.info(f"Average service time over {len(all_task_sets)} runs with {task_type}: {average_service_time:.2}")
        logger.info(f"Average completed time over {len(all_task_sets)} runs with {task_type}: {average_completed_time:.2}")


    return average_wait_time, average_completion_rate


if __name__ == "__main__":
    task_file_path = os.environ['PYTHONPATH']+"task/task.pkl"  # 替换为任务文件路径
    load_model_and_evaluate(task_file_path)