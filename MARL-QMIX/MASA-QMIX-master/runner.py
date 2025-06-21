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
            # 显示输出
            if epoch > 100 and epoch % self.args.evaluate_cycle == 1:  # 确保测试在模型文件更新之后
                episode_reward, avg_completion_time, avg_completion_rate = self.evaluate()
                logger.info(f'Test evaluate_average_reward:{episode_reward:.2f}, evaluate_average_completion_time:{avg_completion_time:.2f},'
                            f' evaluate_average_completion_rate: {np.mean(completion_rate) * 100:.2f}% epoch:{epoch:.2f}')
            epoch_info_dict = defaultdict(list)
            epoch_stats_type = ['total_completion_time', 'total_random_completion_time', 'total_greedy_completion_time', 'total_random_completed_num',
                                'total_greedy_completed_num', 'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                                'epsilon_value', 'total_time_wait', 'total_time_on_road', 'total_service_time']
            # 每个 epoch 收集多个 episode 的各类统计信息
            episodes = []
            completion_rate = []
            random_completion_rate = []
            greedy_completion_rate = []
            with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
                for _ in range(self.args.n_episodes):
                    episode, episode_reward, terminated, episode_stats = self.rolloutWorker.generate_episode(epoch)
                    # 累加统计量
                    episodes.append(episode)
                    epoch_info_dict['epoch_rewards'].append(episode_reward)
                    for stat_type in epoch_stats_type:
                        epoch_info_dict[stat_type].append(episode_stats[stat_type])
                    epoch_info_dict['avg_completion_time'].append(episode_stats["total_completion_time"]/episode_stats["total_completed_num"])
                    completion_rate.append(episode_stats['completion_rate'])
                    epoch_info_dict['avg_random_completion_time'].append(episode_stats["total_random_completion_time"] / episode_stats["total_random_completed_num"])
                    random_completion_rate.append(episode_stats['total_random_completed_num'] / episode_stats['total_tasks_num'])  # 假设tasks_array可用
                    epoch_info_dict['avg_greedy_completion_time'].append(episode_stats["total_greedy_completion_time"] / episode_stats["total_greedy_completed_num"])
                    greedy_completion_rate.append(episode_stats['total_greedy_completed_num'] / episode_stats['total_tasks_num'])

            epoch_avg_dict = defaultdict(int)
            avg_stats_type = ['total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                              'epsilon_value', 'epoch_rewards',  'total_time_wait', 'total_time_on_road', 'total_service_time',
                              'total_completion_time', 'avg_completion_time', 'avg_random_completion_time', 'avg_greedy_completion_time']
            for avg_stat in avg_stats_type:
                epoch_avg_dict[avg_stat] = np.mean(epoch_info_dict[avg_stat])
                self.writer.add_scalar(f"Train/AVG {avg_stat}", epoch_avg_dict[avg_stat], epoch)
            
            epoch_max_dict = defaultdict(int)
            epoch_min_dict = defaultdict(int)
            maxmin_stats_type = ['epoch_rewards', 'avg_completion_time']
            for maxmin_stat in maxmin_stats_type:
                epoch_max_dict[maxmin_stat] = np.max(epoch_info_dict[maxmin_stat])
                epoch_min_dict[maxmin_stat] = np.min(epoch_info_dict[maxmin_stat])
                self.writer.add_scalar(f"Train/MAX {maxmin_stat}", epoch_max_dict[maxmin_stat], epoch)
                self.writer.add_scalar(f"Train/MIN {maxmin_stat}", epoch_min_dict[maxmin_stat], epoch)

            avg_completion_time = epoch_avg_dict['avg_completion_time']
            avg_random_completion_time = epoch_avg_dict['avg_random_completion_time']
            avg_greedy_completion_time = epoch_avg_dict['avg_greedy_completion_time']
            logger.info(f'Compare average_completion_time: QMIX={avg_completion_time:.2f} random={avg_random_completion_time:.2f} greedy={avg_greedy_completion_time:.2f}')
            logger.info(
                f'Compare completion_rate: QMIX={np.mean(completion_rate) * 100:.2f}% random={np.mean(random_completion_rate) * 100:.2f}% greedy={np.mean(greedy_completion_rate) * 100:.2f}%')
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 保存 episode 到缓冲区
            self.buffer.store_episode(episode_batch)
            
            epoch_avg_rewards = epoch_avg_dict['epoch_rewards']
            logger.info(f"Epoch {epoch}: AVG Reward={epoch_avg_rewards:.2f}, AVG Total_completion_time={avg_completion_time:.2f}, AVG completion_rate={np.mean(completion_rate) * 100:.2f}%")
            # 执行训练步骤(buffer 小于200 没必要训练)
            if self.buffer.current_size > 200:
                for train_step in range(self.args.train_steps):
                    logger.info(f" buffer size={self.buffer.current_size:.2f}, batch_size={self.args.batch_size:.2f}")
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    with timer(f'train with train_steps={self.args.train_steps}'):
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        self.writer.close()

    def evaluate(self):
        """
        测试阶段：记录测试的指标
        """
        epoch_info_dict = defaultdict(list)
        epoch_stats_type = ['total_completion_time', 'total_random_completion_time', 'total_greedy_completion_time', 'total_random_completed_num',
                            'total_greedy_completed_num', 'total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                            'epsilon_value', 'total_time_wait', 'total_time_on_road', 'total_service_time']
        # 每个 epoch 收集多个 episode 的各类统计信息
        completion_rate = []
        random_completion_rate = []
        greedy_completion_rate = []

        with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
            for _ in range(self.args.evaluate_epoch):
                _, episode_reward, _, episode_stats = self.rolloutWorker.generate_episode(evaluate=True)

                # 累加统计量
                epoch_info_dict['epoch_rewards'].append(episode_reward)
                for stat_type in epoch_stats_type:
                    epoch_info_dict[stat_type].append(episode_stats[stat_type])
                epoch_info_dict['avg_completion_time'].append(episode_stats["total_completion_time"]/episode_stats["total_completed_num"])
                completion_rate.append(episode_stats['completion_rate'])
                epoch_info_dict['avg_random_completion_time'].append(episode_stats["total_random_completion_time"] / episode_stats["total_random_completed_num"])
                random_completion_rate.append(episode_stats['total_random_completed_num'] / episode_stats['total_tasks_num'])
                epoch_info_dict['avg_greedy_completion_time'].append(episode_stats["total_greedy_completion_time"] / episode_stats["total_greedy_completed_num"])
                greedy_completion_rate.append(episode_stats['total_greedy_completed_num'] / episode_stats['total_tasks_num'])

        epoch_avg_dict = defaultdict(int)
        avg_stats_type = ['total_completed_num', 'completion_rate', 'total_allocated_num', 'allocated_rate',
                          'epsilon_value', 'epoch_rewards', 'total_time_wait', 'total_time_on_road', 'total_service_time',
                          'total_completion_time', 'avg_completion_time', 'avg_random_completion_time', 'avg_greedy_completion_time']
        for avg_stat in avg_stats_type:
            epoch_avg_dict[avg_stat] = np.mean(epoch_info_dict[avg_stat])
            self.writer.add_scalar(f"Test/AVG {avg_stat}", epoch_avg_dict[avg_stat], self.test_episode_count)
        
        epoch_max_dict = defaultdict(int)
        epoch_min_dict = defaultdict(int)
        maxmin_stats_type = ['epoch_rewards', 'avg_completion_time']
        for maxmin_stat in maxmin_stats_type:
            epoch_max_dict[maxmin_stat] = np.max(epoch_info_dict[maxmin_stat])
            epoch_min_dict[maxmin_stat] = np.min(epoch_info_dict[maxmin_stat])
            self.writer.add_scalar(f"Test/MAX {maxmin_stat}", epoch_max_dict[maxmin_stat], self.test_episode_count)
            self.writer.add_scalar(f"Test/MIN {maxmin_stat}", epoch_min_dict[maxmin_stat], self.test_episode_count)

        self.test_episode_count += 1
        return epoch_avg_dict['epoch_rewards'], epoch_avg_dict['avg_completion_time'], epoch_avg_dict['completion_rate']





