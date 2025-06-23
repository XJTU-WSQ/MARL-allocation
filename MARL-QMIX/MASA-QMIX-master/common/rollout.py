import copy
import json
import math
import random
import numpy as np
from task import task_generator
from loguru import logger


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.max_epsilon
        self.anneal_steps = args.anneal_steps
        self.min_epsilon = args.min_epsilon
        self.max_epsilon = args.max_epsilon
        self.current_step = 0
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False, 
                         tasks=None, task_type='qmix', evalue_epsilon=0):
        # 如果提供了固定任务集，则使用，否则生成新任务
        global total_greedy_completion_time, total_greedy_completed_num
        global fixed_initial_pos
        if tasks is not None:
            self.env.tasks_array = tasks
        elif evaluate:
            self.env.tasks_array = task_generator.generate_tasks(task_num=5)
        else:
            self.env.tasks_array = task_generator.generate_tasks(task_num=random.randint(4, 6))

        episode_data = []
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()

        if task_type == 'qmix':
            # 保存初始位置到实例变量
            fixed_initial_pos = copy.deepcopy(self.env.robots.robot_pos)
        # === 关键修改：其他算法使用保存的位置 ===
        # 对于其他算法，且任务集相同（tasks不为None），使用保存的位置
        if task_type != 'qmix' and tasks is not None and hasattr(self, 'fixed_initial_pos'):
            self.env.robots.robot_pos = copy.deepcopy(fixed_initial_pos)
        terminated = False
        step = 0
        episode_reward = 0
        self.agents.policy.init_hidden(1)
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        # epsilon
        epsilon = evalue_epsilon if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # 初始化统计量
        stats_dict = {}
        # qmix算法 统计量
        total_time_wait = 0               # 总体等待时长（已完成任务）
        total_time_on_road = 0            # 总体路程时长（已完成任务）
        total_service_time = 0            # 总体服务时长（已完成任务）
        total_completion_time = 0         # 总体完成时长（已完成任务）
        total_greedy_completion_time = 0
        total_greedy_completed_num = 0

        max_wait_time = 0  # 最长等待时间
        avg_service_coff = 0  # 平均服务系数
        all_reward_components = []  # 存储所有步骤的奖励组成

        while not terminated and step < self.episode_limit:

            self.env.update_task_window()
            self.env.renew_wait_time()
            obs = self.env.get_obs()
            state = self.env.get_state()

            avail_actions = [self.env.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]
            actions = self.agents.choose_actions_batch(obs, avail_actions, epsilon)
            actions_onehot = []
            for action in actions:
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions_onehot.append(action_onehot)
            if task_type == 'qmix':
                reward, terminated, info = self.env.step(actions)
            elif task_type == 'greedy':
                greedy_actions = self.env.assign_tasks_baseline()
                reward, terminated, info = self.env.step(greedy_actions)
            # 记录奖励组成
            all_reward_components.append(info["reward_components"])
            stats_dict[step] = info

            # 如果开启数据记录，将数据存入缓冲区
            if self.args.log_step_data:
                episode_data.append({
                    "step": step,
                    "state": convert_to_native(state),
                    "obs": convert_to_native(obs),
                    "actions": convert_to_native(actions),
                    "reward": convert_to_native(reward),
                    "avail_actions": convert_to_native(avail_actions),
                    "robots_state": convert_to_native(info["robots_state"]),
                    "robots_work_times": convert_to_native(info["robots_work_times"]),
                    "task_window": convert_to_native(info["task_window"]),
                    "done": convert_to_native(info["done"])
                })

            # 保存观测、状态和动作信息
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step' and evaluate != True and task_type == 'qmix':
                # anneal_epsilon = (self.epsilon - self.min_epsilon) / self.anneal_steps
                # epsilon = epsilon - anneal_epsilon if epsilon > self.min_epsilon else epsilon
                self.current_step += 1
                # 改为平方衰减探索率
                progress = min(1.0, self.current_step / self.anneal_steps)
                epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * ((1 - progress)**2)
        # === 在循环结束后添加统计量计算 ===
        total_allocated_num = sum(self.env.tasks_allocated)
        total_completed_num = sum(self.env.tasks_completed)

        # 计算已完成任务的时间统计
        for i in range(len(self.env.tasks_array)):
            if self.env.tasks_completed[i] == 1:
                total_time_wait += self.env.time_wait[i]
                total_time_on_road += self.env.time_on_road[i]
                total_service_time += self.env.service_time[i]
                if self.env.time_wait[i] > max_wait_time:
                    max_wait_time = self.env.time_wait[i]
                avg_service_coff += self.env.service_coff[i]

        # 计算平均服务系数
        if total_completed_num > 0:
            avg_service_coff = avg_service_coff / total_completed_num

        # 计算总体完成时间
        if total_completed_num > 0:
            total_completion_time = sum(self.env.completed_tasks_time)

        total_tasks = len(self.env.tasks_array)
        completion_rate = total_completed_num / total_tasks
        allocated_rate = total_allocated_num / total_tasks

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])  # 如果没有padding的情况下是没有terminal的

        if task_type == 'qmix':
            _, _, _, greedy_stats =  self.generate_episode(episode_num=episode_num, evaluate=evaluate, 
                            tasks=tasks, task_type='greedy', evalue_epsilon=evalue_epsilon)

            total_greedy_completion_time = greedy_stats['total_completion_time']
            total_greedy_completed_num = greedy_stats["total_completed_num"]

            episode_reward = relative_reward(total_completed_num, total_greedy_completed_num, total_random_completed_num, epsilon)
            r = [[(r[i][0]-0.5)*abs(episode_reward)/2+episode_reward] for i in range(len(r))]
            # 全局奖励：考虑平均等待时间，已分配任务数和已完成任务数

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]], dtype=object)
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        if terminated or step == self.episode_limit:
            if self.args.log_step_data:
                with open(f"./episode_logs/episode_{episode_num}.json", "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=4)
                print(f"Episode {episode_num} data saved to episode_{episode_num}.json")
        # 构建统计量
        stats = {
            # 任务完成相关
            "total_completed_num": total_completed_num,
            "completion_rate": completion_rate,
            "total_allocated_num": total_allocated_num,
            "allocated_rate": allocated_rate,
            "total_tasks_num": total_tasks,

            # 时间相关
            "total_time_wait": total_time_wait,
            "max_wait_time": max_wait_time,
            "total_time_on_road": total_time_on_road,
            "total_service_time": total_service_time,
            "total_completion_time": total_completion_time,

            # 效率相关
            "avg_service_coff": avg_service_coff,

            # 奖励和学习相关
            "episode_reward": episode_reward,
            "epsilon_value": epsilon,

            # 对比算法统计
            "total_greedy_completion_time": total_greedy_completion_time,
            "total_greedy_completed_num": total_greedy_completed_num,

            "stats_dict": stats_dict,
            "reward_components": all_reward_components,  # 所有步骤的奖励组成
            "episode_immediate_reward": self.env.episode_immediate_reward,  # 整个episode的即时奖励总和
            "episode_final_reward": self.env.episode_final_reward  # 整个episode的最终奖励
        }

        return episode, episode_reward, terminated, stats


def convert_to_native(obj):
    """
    将 numpy 类型转换为 Python 原生类型，确保 JSON 序列化兼容。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 转为列表
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)  # 转为 Python 布尔类型
    elif isinstance(obj, (np.integer, int)):
        return int(obj)  # 转为 Python 整数
    elif isinstance(obj, (np.floating, float)):
        return float(obj)  # 转为 Python 浮点数
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]  # 递归处理列表
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}  # 递归处理字典
    else:
        return obj  # 返回



