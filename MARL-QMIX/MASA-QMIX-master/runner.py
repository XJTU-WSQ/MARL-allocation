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
                episode_reward, wait_time = self.evaluate()
                logger.info(f'Test evaluate_average_reward:{episode_reward:.2f}, evaluate_average_wait_time:{wait_time:.2f} epoch:{epoch:.2f}')
            epoch_info_dict = defaultdict(list)
            epoch_stats_type = ['concurrent_rewards','conflict_penalty','wait_penalty','service_cost_penalty',
                                'conflicts','shift_time_wait','random_shift_time_wait','greedy_shift_time_wait',
                                'completed_tasks','completion_rate','allocated_rate','epsilon_value','max_shift_time_wait',
                                'shift_time_on_road','shift_service_time','total_completed_time']
            # 每个 epoch 收集多个 episode 的各类统计信息
            episodes = []
            with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
                for _ in range(self.args.n_episodes):
                    episode, episode_reward, terminated, episode_stats = self.rolloutWorker.generate_episode(epoch)
                    # 累加统计量
                    episodes.append(episode)
                    epoch_info_dict['epoch_rewards'].append(episode_reward)
                    epoch_info_dict['wait_times_random'].append(episode_stats["random_shift_time_wait"])
                    epoch_info_dict['wait_times_greedy'].append(episode_stats["greedy_shift_time_wait"])
                    for stat_type in epoch_stats_type:
                        epoch_info_dict[stat_type].append(episode_stats[stat_type])
                    epoch_info_dict['total_avg_time_wait'].append(episode_stats["shift_time_wait"]/episode_stats["total_allocated_num"])

            epoch_avg_dict = defaultdict(int)
            avg_stats_type = ['conflicts','shift_time_wait','max_shift_time_wait','completed_tasks','completion_rate','epsilon_value',
                              'epoch_rewards','concurrent_rewards','conflict_penalty','wait_penalty','service_cost_penalty',
                              'total_avg_time_wait','wait_times_random','wait_times_greedy','shift_time_on_road','shift_service_time','total_completed_time']
            for avg_stat in avg_stats_type:
                epoch_avg_dict[avg_stat] = np.mean(epoch_info_dict[avg_stat])
                self.writer.add_scalar(f"Train/AVG {avg_stat}", epoch_avg_dict[avg_stat], epoch)
            
            epoch_max_dict = defaultdict(int)
            epoch_min_dict = defaultdict(int)
            maxmin_stats_type = ['epoch_rewards','concurrent_rewards','conflict_penalty','wait_penalty','service_cost_penalty','total_avg_time_wait']
            for maxmin_stat in maxmin_stats_type:
                epoch_max_dict[maxmin_stat] = np.max(epoch_info_dict[maxmin_stat])
                epoch_min_dict[maxmin_stat] = np.min(epoch_info_dict[maxmin_stat])
                self.writer.add_scalar(f"Train/MAX {maxmin_stat}", epoch_max_dict[maxmin_stat], epoch)
                self.writer.add_scalar(f"Train/MIN {maxmin_stat}", epoch_min_dict[maxmin_stat], epoch)

            qmix_avg_time = epoch_avg_dict['shift_time_wait']
            random_avg_time = epoch_avg_dict['wait_times_random']
            greedy_avg_time = epoch_avg_dict['wait_times_greedy']
            logger.info(f'Compare shift_time_wait: QMIX={qmix_avg_time:.2f} random={random_avg_time:.2f} greedy={greedy_avg_time:.2f}')
            # 写入 TensorBoard
            
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 保存 episode 到缓冲区
            self.buffer.store_episode(episode_batch)
            
            epoch_avg_rewards = epoch_avg_dict['epoch_rewards']
            avg_qmix_max_time = epoch_avg_dict['max_shift_time_wait']
            logger.info(f"Epoch {epoch}: AVG Reward={epoch_avg_rewards:.2f}, AVG Wait_time={qmix_avg_time:.2f}, MAX Wait_time={avg_qmix_max_time:.2f}")
            # 执行训练步骤(buffer 小于200 没必要训练)
            if self.buffer.current_size > 200:
                for train_step in range(self.args.train_steps):
                    logger.info(f" buffer size={self.buffer.current_size:.2f}, batch_size={self.args.batch_size:.2f}")
                    mini_batch = self.buffer.sample_probabilistic(min(self.buffer.current_size, self.args.batch_size))
                    with timer(f'train with train_steps={self.args.train_steps}'):
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        self.writer.close()

    def evaluate(self):
        """
        测试阶段：记录测试的指标
        """
        epoch_info_dict = defaultdict(list)
        epoch_stats_type = ['concurrent_rewards','conflict_penalty','shift_time_on_road','shift_service_time','total_completed_time',#'wait_penalty','service_cost_penalty',
                            'conflicts','shift_time_wait','random_shift_time_wait','greedy_shift_time_wait',
                            'completed_tasks','completion_rate','allocated_rate','epsilon_value','max_shift_time_wait']
        # 每个 epoch 收集多个 episode 的各类统计信息
        with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
            for _ in range(self.args.evaluate_epoch):
                _, episode_reward, _, episode_stats = self.rolloutWorker.generate_episode(evaluate=True)

                # 累加统计量
                epoch_info_dict['epoch_rewards'].append(episode_reward)
                epoch_info_dict['wait_times_random'].append(episode_stats["random_shift_time_wait"])
                epoch_info_dict['wait_times_greedy'].append(episode_stats["greedy_shift_time_wait"])
                for stat_type in epoch_stats_type:
                    epoch_info_dict[stat_type].append(episode_stats[stat_type])
                epoch_info_dict['total_avg_time_wait'].append(episode_stats["shift_time_wait"]/episode_stats["total_allocated_num"])

        epoch_avg_dict = defaultdict(int)
        avg_stats_type = ['conflicts','shift_time_wait','max_shift_time_wait','completed_tasks','completion_rate','epsilon_value',
                            'epoch_rewards','concurrent_rewards','conflict_penalty','shift_time_on_road','shift_service_time','total_completed_time',#'wait_penalty','service_cost_penalty',
                            'total_avg_time_wait','wait_times_random','wait_times_greedy','total_allocated_num']
        for avg_stat in avg_stats_type:
            epoch_avg_dict[avg_stat] = np.mean(epoch_info_dict[avg_stat])
            self.writer.add_scalar(f"Test/AVG {avg_stat}", epoch_avg_dict[avg_stat], self.test_episode_count)
        
        epoch_max_dict = defaultdict(int)
        epoch_min_dict = defaultdict(int)
        maxmin_stats_type = ['epoch_rewards','concurrent_rewards','conflict_penalty',#'wait_penalty','service_cost_penalty',
                             'total_avg_time_wait']
        for maxmin_stat in maxmin_stats_type:
            epoch_max_dict[maxmin_stat] = np.max(epoch_info_dict[maxmin_stat])
            epoch_min_dict[maxmin_stat] = np.min(epoch_info_dict[maxmin_stat])
            self.writer.add_scalar(f"Test/MAX {maxmin_stat}", epoch_max_dict[maxmin_stat], self.test_episode_count)
            self.writer.add_scalar(f"Test/MIN {maxmin_stat}", epoch_min_dict[maxmin_stat], self.test_episode_count)

        self.test_episode_count += 1
        return epoch_avg_dict['epoch_rewards'], epoch_avg_dict['shift_time_wait']





