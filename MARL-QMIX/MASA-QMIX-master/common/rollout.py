import copy
import json
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

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False, 
                         tasks=None, task_type = 'qmix', evalue_epsilon=0):
        # 如果提供了固定任务集，则使用，否则生成新任务
        if tasks is not None:
            self.env.tasks_array = tasks
        else:
            self.env.tasks_array = task_generator.generate_tasks()

        episode_data = []
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
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
        total_conflicts = 0
        total_wait_penalty = 0
        total_service_cost_penalty = 0
        total_conflict_penalty = 0
        total_concurrent_rewards = 0

        total_shift_time_wait = 0
        total_random_shift_time_wait = 0
        total_greedy_shift_time_wait = 0
        
        max_shift_time_wait = 0
        max_random_time_wait = 0
        max_greedy_time_wait = 0

        max_shift_time_list = []
        max_random_time_list = []
        max_greedy_time_list = []       

        total_shift_allocated_num = 0
        total_random_allocated_num = 0
        total_greedy_allocated_num = 0

        total_shift_completed_num = 0
        total_random_completed_num = 0
        total_greedy_completed_num = 0
        tmp_num = 0
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
            random_actions = self.env.assign_tasks_baseline(baseline_type='random')
            greedy_actions = self.env.assign_tasks_baseline(baseline_type='greedy')
            _, _, random_info = self.env.step(random_actions, freeze_env=True)
            _, _, greedy_info = self.env.step(greedy_actions, freeze_env=True)
            if task_type == 'qmix':
                reward, terminated, info = self.env.step(actions)
            elif task_type == 'random':
                reward, terminated, info = self.env.step(random_actions)
            elif task_type == 'greedy':
                reward, terminated, info = self.env.step(greedy_actions)
            # 临时取消了局部奖励，后续待优化
            # reward1 = relative_reward(info,random_info,greedy_info,'concurrent_rewards', self.epsilon)
            # reward2 = relative_reward(info,random_info,greedy_info,'conflict_penalty', self.epsilon)
            # reward3 = relative_reward(info,random_info,greedy_info,'total_service_cost_penalty', self.epsilon)
            # reward4 = relative_reward(info,random_info,greedy_info,'total_wait_penalty', self.epsilon)
            # reward4 = relative_reward(info,random_info,greedy_info,'total_wait_penalty', self.epsilon)
            # reward = 0.2*reward1 + 0.2*reward2 + 0.2*reward3 + 0.1*reward4
            # 累积统计量
            total_concurrent_rewards += info["concurrent_rewards"]
            total_conflicts += info["conflict_count"]
            total_conflict_penalty += info["conflict_penalty"]
            total_service_cost_penalty += info["total_service_cost_penalty"]
            total_wait_penalty += info["total_wait_penalty"]

            total_shift_time_wait += info["shift_time_wait"]
            total_random_shift_time_wait += random_info["shift_time_wait"]
            total_greedy_shift_time_wait += greedy_info["shift_time_wait"]

            max_shift_time_wait = max(max_shift_time_wait, info["max_time_wait"])
            max_random_time_wait = max(max_random_time_wait, random_info["max_time_wait"])
            max_greedy_time_wait = max(max_greedy_time_wait, greedy_info["max_time_wait"])
            max_shift_time_list.append(info["max_time_wait"])
            max_random_time_list.append(random_info["max_time_wait"])
            max_greedy_time_list.append(greedy_info["max_time_wait"])

            total_shift_allocated_num += info["shift_allocated_num"]
            total_random_allocated_num += random_info["shift_allocated_num"]
            total_greedy_allocated_num += greedy_info["shift_allocated_num"]

            total_shift_completed_num += info["shift_completed_num"]
            total_random_completed_num += random_info["shift_completed_num"]
            total_greedy_completed_num += greedy_info["shift_completed_num"]            
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
                    "task_window": convert_to_native(info["task_window"]),
                    "conflict_count": convert_to_native(info["conflict_count"]),
                    "total_wait_penalty": convert_to_native(info["total_wait_penalty"]),
                    "total_service_cost_penalty": convert_to_native(info["total_service_cost_penalty"]),
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
            step += 1

            if self.args.epsilon_anneal_scale == 'step' and evaluate!=True:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
                tmp_num += 1
        avg_greedy_time_wait = total_greedy_shift_time_wait/total_greedy_allocated_num
        avg_random_time_wait = total_random_shift_time_wait/total_random_allocated_num
        avg_qmix_time_wait = total_shift_time_wait/total_shift_allocated_num
        episode_reward = relative_reward(1/avg_qmix_time_wait, 1/avg_greedy_time_wait, 1/avg_random_time_wait, self.epsilon)
        episode_reward2 = relative_reward(1/max_shift_time_wait, 1/max_greedy_time_wait, 1/max_random_time_wait, self.epsilon)
        episode_reward3 = relative_reward(total_shift_allocated_num, total_random_allocated_num, total_greedy_allocated_num, self.epsilon)
        episode_reward4 = relative_reward(total_shift_completed_num, total_random_completed_num, total_greedy_completed_num, self.epsilon)
        episode_reward = episode_reward + episode_reward2
        # 全局奖励：考虑平均等待时间，已分配任务数和已完成任务数
        # last obs
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
        total_wait_time = self.env.total_time_wait  # 累计等待时间
        total_completed_tasks = sum(self.env.tasks_completed)
        total_allocated_tasks = sum(self.env.tasks_allocated)
        total_tasks = len(self.env.tasks_array)
        completion_rate = total_completed_tasks / total_tasks
        allocated_rate = total_allocated_tasks / total_tasks
        # 构建统计量
        stats = {
            "concurrent_rewards": total_concurrent_rewards,
            "conflicts": total_conflicts,
            "conflict_penalty": total_conflict_penalty,
            "wait_time": total_wait_time,
            "wait_penalty": total_wait_penalty,
            "service_cost_penalty": total_service_cost_penalty,
            "completed_tasks": total_completed_tasks,
            "completion_rate": completion_rate,
            "allocated_rate": allocated_rate,
            "episode_reward": episode_reward,
            "epsilon_value":epsilon,
            "shift_time_wait":total_shift_time_wait,
            "max_shift_time_wait":max_shift_time_wait,
            "total_allocated_num":total_shift_allocated_num,
            "total_completed_num":total_shift_completed_num,
            "random_shift_time_wait":total_random_shift_time_wait,
            "greedy_shift_time_wait":total_greedy_shift_time_wait
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

def relative_reward(a, b, c, epsilon):
    # 根据 b 和 c 构建基线，评估 a 所处位置并用于奖励的计算
    baseline_lower = min(b, c)+(0.9-epsilon)*abs(b-c)
    baseline_upper = max(b, c)+(0.9-epsilon)*abs(b-c)
    one_level_range = max(a,b)
    if a < baseline_lower:
        return -2*(baseline_lower - a)/(one_level_range)
    elif a > baseline_upper:
        return 2*(a - baseline_upper)/(one_level_range)
    else:
        return (a - baseline_lower)/one_level_range
