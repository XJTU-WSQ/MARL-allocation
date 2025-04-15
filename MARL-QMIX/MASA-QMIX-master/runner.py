import os
import sys
import numpy as np
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


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

    def run(self, generate_episode_type='parallel'):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            # 显示输出
            if epoch % self.args.evaluate_cycle == 0:  # 测试
                episode_reward, wait_time = self.evaluate()
                print("\nevaluate_average_reward:", episode_reward, "evaluate_average_wait_time:", wait_time, "epoch:", epoch)

            # 每个 epoch 收集多个 episode
            concurrent_rewards = []
            conflict_penalties = []
            wait_penalties = []
            service_cost_penalties = []
            epoch_rewards = []

            total_conflicts = []
            total_wait_times = []
            total_completed_tasks = []
            total_completion_rates = []
            episodes = []
            with timer(f'generate_episode with n_episodes={self.args.n_episodes}'):
                if generate_episode_type == 'origin':
                    for _ in range(self.args.n_episodes):
                        episode, episode_reward, terminated, episode_stats = self.rolloutWorker.generate_episode(epoch)
                        # 累加统计量
                        episodes.append(episode)
                        epoch_rewards.append(episode_reward)
                        concurrent_rewards.append(episode_stats["concurrent_rewards"])
                        conflict_penalties.append(episode_stats["conflict_penalty"])
                        wait_penalties.append(episode_stats["wait_penalty"])
                        service_cost_penalties.append(episode_stats["service_cost_penalty"])
                        total_conflicts.append(episode_stats["conflicts"])
                        total_wait_times.append(episode_stats["wait_time"])
                        total_completed_tasks.append(episode_stats["completed_tasks"])
                        total_completion_rates.append(episode_stats["completion_rate"])
                elif generate_episode_type == 'parallel':
                    with ThreadPoolExecutor(max_workers=self.args.n_episodes) as executor:
                        futures = [executor.submit(self.rolloutWorker.generate_episode) for _ in range(self.args.n_episodes)]
                        results = [future.result() for future in futures]
                        for result in results:
                            episode = result[0]
                            episode_reward = result[1]
                            episode_stats = result[-1]
                            episodes.append(episode)
                            epoch_rewards.append(episode_reward)
                            concurrent_rewards.append(episode_stats["concurrent_rewards"])
                            conflict_penalties.append(episode_stats["conflict_penalty"])
                            wait_penalties.append(episode_stats["wait_penalty"])
                            service_cost_penalties.append(episode_stats["service_cost_penalty"])
                            total_conflicts.append(episode_stats["conflicts"])
                            total_wait_times.append(episode_stats["wait_time"])
                            total_completed_tasks.append(episode_stats["completed_tasks"])
                            total_completion_rates.append(episode_stats["completion_rate"])

            avg_conflicts = np.mean(total_conflicts)
            avg_wait_time = np.mean(total_wait_times)
            avg_completed_tasks = np.mean(total_completed_tasks)
            avg_completion_rate = np.mean(total_completion_rates)

            # Compute statistics for this epoch
            avg_reward = np.mean(epoch_rewards)
            max_reward = np.max(epoch_rewards)
            min_reward = np.min(epoch_rewards)

            avg_concurrent_rewards = np.mean(concurrent_rewards)
            max_concurrent_rewards = np.max(concurrent_rewards)
            min_concurrent_rewards = np.min(concurrent_rewards)

            avg_conflict_penalties = np.mean(conflict_penalties)
            max_conflict_penalties = np.max(conflict_penalties)
            min_conflict_penalties = np.min(conflict_penalties)

            avg_wait_penalty = np.mean(wait_penalties)
            max_wait_penalty = np.max(wait_penalties)
            min_wait_penalty = np.min(wait_penalties)

            avg_service_cost_penalty = np.mean(service_cost_penalties)
            max_service_cost_penalty = np.max(service_cost_penalties)
            min_service_cost_penalty = np.min(service_cost_penalties)

            # 写入 TensorBoard
            self.writer.add_scalar("Train/Conflicts", avg_conflicts, epoch)
            self.writer.add_scalar("Train/Wait Time", avg_wait_time, epoch)
            self.writer.add_scalar("Train/Completed Tasks", avg_completed_tasks, epoch)
            self.writer.add_scalar("Train/Completion Rate", avg_completion_rate, epoch)

            self.writer.add_scalar("Train/Average Reward", avg_reward, epoch)
            self.writer.add_scalar("Train/Max Reward", max_reward, epoch)
            self.writer.add_scalar("Train/Min Reward", min_reward, epoch)

            self.writer.add_scalar("Train/Average Wait Penalty", avg_wait_penalty, epoch)
            self.writer.add_scalar("Train/Max Wait Penalty", max_wait_penalty, epoch)
            self.writer.add_scalar("Train/Min Wait Penalty", min_wait_penalty, epoch)

            self.writer.add_scalar("Train/Average Service Cost Penalty", avg_service_cost_penalty, epoch)
            self.writer.add_scalar("Train/Max Service Cost Penalty", max_service_cost_penalty, epoch)
            self.writer.add_scalar("Train/Min Service Cost Penalty", min_service_cost_penalty, epoch)

            self.writer.add_scalar("Train/Average Concurrent Rewards", avg_concurrent_rewards, epoch)
            self.writer.add_scalar("Train/Max Concurrent Rewards", max_concurrent_rewards, epoch)
            self.writer.add_scalar("Train/Min Concurrent Rewards", min_concurrent_rewards, epoch)

            self.writer.add_scalar("Train/Average Conflict Penalty", avg_conflict_penalties, epoch)
            self.writer.add_scalar("Train/Max Conflict Penalty", max_conflict_penalties, epoch)
            self.writer.add_scalar("Train/Min Conflict Penalty", min_conflict_penalties, epoch)

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 保存 episode 到缓冲区
            self.buffer.store_episode(episode_batch)

            logger.info(f"Epoch {epoch}: Average Reward={avg_reward:.2f}, Average Wait_time={avg_wait_time:.2f}, \
                        Estimated Wait_time per_task={avg_wait_time/avg_completed_tasks:.2f}")
            # 执行训练步骤
            
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
        total_rewards = []
        total_wait_penalties = []
        total_service_cost_penalties = []
        total_conflicts = []
        total_wait_times = []
        total_completed_tasks = []
        total_completion_rates = []

        for _ in range(self.args.evaluate_epoch):
            _, episode_reward, _, episode_stats = self.rolloutWorker.generate_episode(evaluate=True)

            total_rewards.append(episode_reward)
            total_wait_penalties.append(episode_stats["wait_penalty"])
            total_service_cost_penalties.append(episode_stats["service_cost_penalty"])
            total_conflicts.append(episode_stats["conflicts"])
            total_wait_times.append(episode_stats["wait_time"])
            total_completed_tasks.append(episode_stats["completed_tasks"])
            total_completion_rates.append(episode_stats["completion_rate"])

        avg_conflicts = np.mean(total_conflicts)
        avg_wait_time = np.mean(total_wait_times)
        avg_completed_tasks = np.mean(total_completed_tasks)
        avg_completion_rate = np.mean(total_completion_rates)
        # Compute statistics for this epoch
        avg_reward = np.mean(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)

        avg_wait_penalty = np.mean(total_wait_penalties)
        max_wait_penalty = np.max(total_wait_penalties)
        min_wait_penalty = np.min(total_wait_penalties)

        avg_service_cost_penalty = np.mean(total_service_cost_penalties)
        max_service_cost_penalty = np.max(total_service_cost_penalties)
        min_service_cost_penalty = np.min(total_service_cost_penalties)

        # 写入 TensorBoard
        self.writer.add_scalar("Test/Conflicts", avg_conflicts, self.test_episode_count)
        self.writer.add_scalar("Test/Wait Time", avg_wait_time, self.test_episode_count)
        self.writer.add_scalar("Test/Completed Tasks", avg_completed_tasks, self.test_episode_count)
        self.writer.add_scalar("Test/Completion Rate", avg_completion_rate, self.test_episode_count)

        self.writer.add_scalar("Test/Average Reward", avg_reward, self.test_episode_count)
        self.writer.add_scalar("Test/Max Reward", max_reward, self.test_episode_count)
        self.writer.add_scalar("Test/Min Reward", min_reward, self.test_episode_count)

        self.writer.add_scalar("Test/Average Wait Penalty", avg_wait_penalty, self.test_episode_count)
        self.writer.add_scalar("Test/Max Wait Penalty", max_wait_penalty, self.test_episode_count)
        self.writer.add_scalar("Test/Min Wait Penalty", min_wait_penalty, self.test_episode_count)

        self.writer.add_scalar("Test/Average Service Cost Penalty", avg_service_cost_penalty, self.test_episode_count)
        self.writer.add_scalar("Test/Max Service Cost Penalty", max_service_cost_penalty, self.test_episode_count)
        self.writer.add_scalar("Test/Min Service Cost Penalty", min_service_cost_penalty, self.test_episode_count)

        self.test_episode_count += 1
        return avg_reward, avg_wait_time





