import os
import sys
import numpy as np
from datetime import datetime
from loguru import logger
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
from policy.qmix import QMIX
from utils.util import timer


# 配置日志文件路径，包含当前日期
current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file_path = f"logs/log_{current_date}.log"
logger.add(log_file_path, rotation="00:00", retention="10 days", level="INFO")

# 计算并记录平均统计量
group_mapping = {
    'epoch_rewards': '0_Reward',
    'completion_rate': '1_Completion',
    'allocated_rate': '1_Completion',
    'total_completed_num': '1_Completion',
    'avg_completion_time': '2_Time',
    'avg_time_wait': '2_Time',
    'avg_time_on_road': '2_Time',
    'avg_service_time': '2_Time',
    'max_wait_time': '2_Time',
    'avg_service_coff': '4_Training',
    'avg_random_completion_time': '3_Baselines',
    'avg_greedy_completion_time': '3_Baselines',
    'random_completion_rate': '3_Baselines',
    'greedy_completion_rate': '3_Baselines',
    'epsilon_value': '4_Training',
    'total_allocated_num': '1_Completion',
    'total_random_completed_num': '3_Baselines',
    'total_greedy_completed_num': '3_Baselines'
}

class Runner:
    def __init__(self, env, args):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式化为常见的时间格式
        self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}_{timestamp}")
        self.qmix = QMIX(args, writer=self.writer)
        self.episode_count = 0
        self.test_episode_count = 0
        self.env = env
        self.agents = Agents(args, writer=self.writer)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            # 动态调整权重
            if epoch < 1000:  # 初期阶段
                self.agents.policy.args.completion_weight = 0.7
                self.agents.policy.args.time_weight = 0.3
            elif epoch < 5000:  # 中期阶段
                self.agents.policy.args.completion_weight = 0.5
                self.agents.policy.args.time_weight = 0.5
            else:  # 后期阶段
                self.agents.policy.args.completion_weight = 0.3
                self.agents.policy.args.time_weight = 0.7
            # 记录权重到TensorBoard
            self.writer.add_scalar("Train/4_Training/Completion_Weight",
                                   self.agents.policy.args.completion_weight, epoch)
            self.writer.add_scalar("Train/4_Training/Time_Weight",
                                   self.agents.policy.args.time_weight, epoch)

            # 传递权重到环境
            self.rolloutWorker.args.completion_weight = self.agents.policy.args.completion_weight
            self.rolloutWorker.args.time_weight = self.agents.policy.args.time_weight
            # 显示输出
            if epoch > 100 and epoch % self.args.evaluate_cycle == 1:  # 确保测试在模型文件更新之后
                self.evaluate()
            # 初始化统计字典
            epoch_info_dict = defaultdict(list)

            # 每个 epoch 收集多个 episode 的各类统计信息
            with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
                for _ in range(self.args.n_episodes):
                    episode, episode_reward, terminated, episode_stats = self.rolloutWorker.generate_episode(epoch)

                    # 收集基础统计量
                    epoch_info_dict['epoch_rewards'].append(episode_reward)
                    for key in [
                        'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                        'avg_service_coff', 'epsilon_value', 'total_random_completed_num', 'total_greedy_completed_num'
                    ]:
                        if key in episode_stats:
                            epoch_info_dict[key].append(episode_stats[key])

                    # 计算时间相关统计量
                    if episode_stats['total_completed_num'] > 0:
                        epoch_info_dict['avg_completion_time'].append(episode_stats["total_completion_time"] / episode_stats["total_completed_num"])
                        epoch_info_dict['avg_time_wait'].append(episode_stats["total_time_wait"] / episode_stats["total_completed_num"])
                        epoch_info_dict['avg_time_on_road'].append(episode_stats["total_time_on_road"] / episode_stats["total_completed_num"])
                        epoch_info_dict['avg_service_time'].append(episode_stats["total_service_time"] / episode_stats["total_completed_num"])
                    else:
                        for key in ['avg_completion_time', 'avg_time_wait', 'avg_time_on_road', 'avg_service_time']:
                            epoch_info_dict[key].append(0)

                    # 记录最长等待时间
                    epoch_info_dict['max_wait_time'].append(episode_stats.get('max_wait_time', 0))

                    # 计算对比算法统计量
                    if episode_stats['total_random_completed_num'] > 0:
                        epoch_info_dict['avg_random_completion_time'].append(episode_stats["total_random_completion_time"] / episode_stats["total_random_completed_num"])
                    else:
                        epoch_info_dict['avg_random_completion_time'].append(0)

                    if episode_stats['total_greedy_completed_num'] > 0:
                        epoch_info_dict['avg_greedy_completion_time'].append(episode_stats["total_greedy_completion_time"] / episode_stats["total_greedy_completed_num"])
                    else:
                        epoch_info_dict['avg_greedy_completion_time'].append(0)

                    # 计算完成率
                    total_tasks = episode_stats.get('total_tasks_num', 1)
                    epoch_info_dict['random_completion_rate'].append(episode_stats['total_random_completed_num'] / total_tasks)
                    epoch_info_dict['greedy_completion_rate'].append(episode_stats['total_greedy_completed_num'] / total_tasks)
                    # 保存 episode 到缓冲区
                    if _ == 0:
                        episode_batch = episode
                    else:
                        for key in episode_batch.keys():
                            episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 计算并记录平均统计量

            for stat_type in [
                'epoch_rewards', 'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                'avg_completion_time', 'avg_time_wait', 'avg_time_on_road', 'avg_service_time', 'max_wait_time',
                'avg_service_coff', 'avg_random_completion_time', 'avg_greedy_completion_time',
                'total_random_completed_num', 'total_greedy_completed_num',
                'random_completion_rate', 'greedy_completion_rate', 'epsilon_value'
            ]:
                if stat_type in epoch_info_dict:
                    avg_value = np.mean(epoch_info_dict[stat_type])
                    # 添加分组前缀
                    group = group_mapping.get(stat_type, "5_Other")
                    self.writer.add_scalar(f"Train/{group}/AVG {stat_type}", avg_value, epoch)

                    # 对于关键指标记录最大最小值
                    if stat_type in ['epoch_rewards', 'avg_completion_time']:
                        max_value = np.max(epoch_info_dict[stat_type])
                        min_value = np.min(epoch_info_dict[stat_type])
                        self.writer.add_scalar(f"Train/{group}/MAX {stat_type}", max_value, epoch)
                        self.writer.add_scalar(f"Train/{group}/MIN {stat_type}", min_value, epoch)

            # 记录对比算法的性能
            avg_completion_time = np.mean(epoch_info_dict['avg_completion_time'])
            avg_random_completion_time = np.mean(epoch_info_dict['avg_random_completion_time'])
            avg_greedy_completion_time = np.mean(epoch_info_dict['avg_greedy_completion_time'])

            qmix_completion_rate = np.mean(epoch_info_dict['completion_rate'])
            random_completion_rate = np.mean(epoch_info_dict['random_completion_rate'])
            greedy_completion_rate = np.mean(epoch_info_dict['greedy_completion_rate'])

            logger.info(f'Compare average_completion_time: QMIX={avg_completion_time:.2f} '
                        f'random={avg_random_completion_time:.2f} greedy={avg_greedy_completion_time:.2f}')
            logger.info(f'Compare completion_rate: QMIX={qmix_completion_rate * 100:.2f}% '
                        f'random={random_completion_rate * 100:.2f}% greedy={greedy_completion_rate * 100:.2f}%')

            # 保存 episode 到缓冲区
            self.buffer.store_episode(episode_batch)

            # 记录当前 epoch 的平均奖励
            epoch_avg_rewards = np.mean(epoch_info_dict['epoch_rewards'])
            logger.info(f"Epoch {epoch}: AVG Reward={epoch_avg_rewards:.2f}, "
                        f"AVG Total_completion_time={avg_completion_time:.2f}, "
                        f"AVG completion_rate={qmix_completion_rate * 100:.2f}%")

            # 执行训练步骤(buffer 小于200 没必要训练)
            if self.buffer.current_size > 200:
                for train_step in range(self.args.train_steps):
                    logger.info(f" buffer size={self.buffer.current_size}, batch_size={self.args.batch_size}")
                    mini_batch = self.buffer.sample_probabilistic(min(self.buffer.current_size, self.args.batch_size))
                    with timer(f'train with train_steps={self.args.train_steps}'):
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        self.writer.close()

    def evaluate(self):
        """
        测试阶段：记录测试的指标
        """
        # 保存原始权重
        original_completion_weight = self.agents.policy.args.completion_weight
        original_time_weight = self.agents.policy.args.time_weight

        # 评估时使用平衡权重
        self.agents.policy.args.completion_weight = 0.5
        self.agents.policy.args.time_weight = 0.5
        self.rolloutWorker.args.completion_weight = 0.5
        self.rolloutWorker.args.time_weight = 0.5

        # 初始化统计字典
        epoch_info_dict = defaultdict(list)

        # 每个评估周期收集多个episode的统计信息
        with timer(f'generate_episode with n_episodes={self.args.evaluate_epoch}'):
            for _ in range(self.args.evaluate_epoch):
                _, episode_reward, _, episode_stats = self.rolloutWorker.generate_episode(evaluate=True)

                # 收集基础统计量
                epoch_info_dict['epoch_rewards'].append(episode_reward)
                for key in [
                    'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                    'avg_service_coff', 'epsilon_value',
                    'total_random_completed_num', 'total_greedy_completed_num'
                ]:
                    if key in episode_stats:
                        epoch_info_dict[key].append(episode_stats[key])

                # 计算时间相关统计量
                if episode_stats['total_completed_num'] > 0:
                    epoch_info_dict['avg_completion_time'].append(
                        episode_stats["total_completion_time"] / episode_stats["total_completed_num"]
                    )
                    epoch_info_dict['avg_time_wait'].append(
                        episode_stats["total_time_wait"] / episode_stats["total_completed_num"]
                    )
                    epoch_info_dict['avg_time_on_road'].append(
                        episode_stats["total_time_on_road"] / episode_stats["total_completed_num"]
                    )
                    epoch_info_dict['avg_service_time'].append(
                        episode_stats["total_service_time"] / episode_stats["total_completed_num"]
                    )
                else:
                    for key in ['avg_completion_time', 'avg_time_wait', 'avg_time_on_road', 'avg_service_time']:
                        epoch_info_dict[key].append(0)

                # 记录最长等待时间
                epoch_info_dict['max_wait_time'].append(episode_stats.get('max_wait_time', 0))

                # 计算对比算法统计量
                if episode_stats['total_random_completed_num'] > 0:
                    epoch_info_dict['avg_random_completion_time'].append(
                        episode_stats["total_random_completion_time"] / episode_stats["total_random_completed_num"]
                    )
                else:
                    epoch_info_dict['avg_random_completion_time'].append(0)

                if episode_stats['total_greedy_completed_num'] > 0:
                    epoch_info_dict['avg_greedy_completion_time'].append(
                        episode_stats["total_greedy_completion_time"] / episode_stats["total_greedy_completed_num"]
                    )
                else:
                    epoch_info_dict['avg_greedy_completion_time'].append(0)

                # 计算完成率
                total_tasks = episode_stats.get('total_tasks_num', 1)
                epoch_info_dict['random_completion_rate'].append(
                    episode_stats['total_random_completed_num'] / total_tasks
                )
                epoch_info_dict['greedy_completion_rate'].append(
                    episode_stats['total_greedy_completed_num'] / total_tasks
                )

        # 计算并记录平均统计量
        for stat_type in [
            'epoch_rewards', 'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
            'avg_completion_time', 'avg_time_wait', 'avg_time_on_road', 'avg_service_time', 'max_wait_time',
            'avg_service_coff', 'avg_random_completion_time', 'avg_greedy_completion_time',
            'total_random_completed_num', 'total_greedy_completed_num',
            'random_completion_rate', 'greedy_completion_rate', 'epsilon_value'
        ]:
            if stat_type in epoch_info_dict:
                avg_value = np.mean(epoch_info_dict[stat_type])
                # 添加分组前缀
                group = group_mapping.get(stat_type, "5_Other")
                self.writer.add_scalar(f"Test/{group}/AVG {stat_type}", avg_value, self.test_episode_count)

                # 对于关键指标记录最大最小值
                if stat_type in ['epoch_rewards', 'avg_completion_time']:
                    max_value = np.max(epoch_info_dict[stat_type])
                    min_value = np.min(epoch_info_dict[stat_type])
                    self.writer.add_scalar(f"Test/{group}/MAX {stat_type}", max_value, self.test_episode_count)
                    self.writer.add_scalar(f"Test/{group}/MIN {stat_type}", min_value, self.test_episode_count)

        # 计算整体平均值用于返回
        avg_reward = np.mean(epoch_info_dict['epoch_rewards'])
        avg_completion_time = np.mean(epoch_info_dict['avg_completion_time'])
        avg_completion_rate = np.mean(epoch_info_dict['completion_rate'])

        # 记录对比算法性能
        avg_random_completion_time = np.mean(epoch_info_dict['avg_random_completion_time'])
        avg_greedy_completion_time = np.mean(epoch_info_dict['avg_greedy_completion_time'])
        random_completion_rate = np.mean(epoch_info_dict['random_completion_rate'])
        greedy_completion_rate = np.mean(epoch_info_dict['greedy_completion_rate'])

        logger.info(f'Test Compare average_completion_time: QMIX={avg_completion_time:.2f} '
                    f'random={avg_random_completion_time:.2f} greedy={avg_greedy_completion_time:.2f}')
        logger.info(f'Test Compare completion_rate: QMIX={avg_completion_rate * 100:.2f}% '
                    f'random={random_completion_rate * 100:.2f}% greedy={greedy_completion_rate * 100:.2f}%')

        # 增加测试计数器
        self.test_episode_count += 1

        # 恢复原始权重
        self.agents.policy.args.completion_weight = original_completion_weight
        self.agents.policy.args.time_weight = original_time_weight
        self.rolloutWorker.args.completion_weight = original_completion_weight
        self.rolloutWorker.args.time_weight = original_time_weight

        return avg_reward, avg_completion_time, avg_completion_rate





