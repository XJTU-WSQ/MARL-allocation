from abc import ABC
import gym
import random

import numpy as np
from gym.utils import seeding
from copy import deepcopy
from utils.sites import Sites
from utils.robots import Robots
from utils.tasks import Tasks
from utils.util import *
from task import task_generator


class ScheduleEnv(gym.Env, ABC):

    def __init__(self, episode_limit=120):
        # 实例化
        self.sites = Sites()
        self.robots = Robots()
        self.tasks = Tasks()
        
        # self.robots_state = [0 for _ in range(self.robots.num_robots)]
        # self.robots_work_times = [0 for _ in range(self.robots.num_robots)]
        # 奖励函数的权重
        self.episode_limit = episode_limit
        self.current_step = 0  # 新增：记录当前步数
        # 任务相关参数
        self.tasks_array = task_generator.generate_tasks()
        # task： Task index|Time|Location|Task Type|Target|Duration
        self.task_window_size = 10

        self.reward_components = []  # 新增：记录每个步骤的奖励组成
        self.episode_immediate_reward = 0  # 新增：记录整个episode的即时奖励总和
        self.episode_final_reward = 0  # 新增：记录整个episode的最终奖励
        self.reset()  # 初始化环境

    def reset(self):
        """
        环境重置，初始化所有参数。
        """
        # 任务信息初始化
        self.current_step = 0  # 重置步数计数器
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.time_on_road = [0 for _ in range(len(self.tasks_array))]
        self.service_time = [0 for _ in range(len(self.tasks_array))]
        self.service_coff = [0 for _ in range(len(self.tasks_array))]
        self.completed_tasks_time = []
        self.total_time_wait = 0
        self.total_time_on_road = 0
        self.total_time_on_road2 = 0
        self.total_service_time = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.unallocated_tasks = set(range(len(self.tasks_array))) # 未分配任务集合
        self.task_window = [[0 for _ in range(6)] for _ in range(self.task_window_size)]

        # 机器人信息初始化
        self.robots.robot_pos = self.robots.robot_sites_pos
        random.shuffle(self.robots.robot_pos)
        self.robots_state = [0 for _ in range(self.robots.num_robots)] # 机器人状态信息 : 1占用，0空闲
        self.robots_work_times = [0 for _ in range(self.robots.num_robots)]
        self.robots.robots_tasks_info = np.zeros([self.robots.num_robots, 6], dtype=int)
        
        # 动态获取 obs_shape
        temp_avail_action = [0] * (self.task_window_size + 1)
        temp_obs = self.get_agent_obs(0, temp_avail_action)
        self.obs_shape = len(temp_obs)  # 动态观测维度

        # 环境参数初始化
        self.time = 0.0
        self.done = False
        self.state4marl = [0 for _ in range(len(self.get_state()))]
        self.obs4marl = np.zeros((self.robots.num_robots, self.obs_shape), dtype=np.float32)
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]
        self.update_task_window()

        self.reward_components = []  # 重置奖励记录
        self.episode_immediate_reward = 0
        self.episode_final_reward = 0

        return self.get_obs(), self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def renew_wait_time(self):
        """
        优化后的 renew_wait_time：只遍历未分配任务集合。
        """
        tasks_to_remove = []
        for task_index in list(self.unallocated_tasks):  # 强制拷贝避免修改迭代错误
            if self.tasks_allocated[task_index] == 1:
                tasks_to_remove.append(task_index)
                continue  # ⚠️ 已分配任务跳过等待时间计算

            if self.tasks_array[task_index][1] <= self.time:
                self.time_wait[task_index] = self.time - self.tasks_array[task_index][1]
            else:
                break

        # 从未分配任务集合中移除已分配任务
        self.unallocated_tasks -= set(tasks_to_remove)

    def update_task_window(self):
        """
        更新任务窗：只在每个 step 开始时生成一次任务窗。
        """
        m = 0
        self.task_window = [[0 for _ in range(6)] for _ in range(self.task_window_size)]
        for task_index in range(len(self.tasks_array)):
            if self.tasks_array[task_index][1] <= self.time and self.tasks_allocated[task_index] == 0:
                self.task_window[m] = self.tasks_array[task_index].tolist()
                m += 1
                if m == self.task_window_size:
                    break
            if self.tasks_array[task_index][1] > self.time:
                break

    def get_state(self):
        """
        获取全局状态，包含机器人和任务的全局信息。
        现在包含：机器人状态、工作时间、位置、任务进度
        """
        state = []
        max_pos_value = 60.0  # 坐标范围为 [0, 60]

        # 1. 添加机器人状态信息（每个机器人5个特征）
        for robot_id in range(self.robots.num_robots):
            # 基础信息
            state.append(self.robots_state[robot_id])  # 状态(0/1)
            state.append(self.robots_work_times[robot_id])  # 工作时间

            # 位置信息
            state.append(self.robots.robot_pos[robot_id][0] / max_pos_value)  # x坐标
            state.append(self.robots.robot_pos[robot_id][1] / max_pos_value)  # y坐标

            # 任务进度(0-1)
            if self.robots_state[robot_id] == 1:  # 忙碌状态
                task_info = self.robots.robots_tasks_info[robot_id]
                complete_time = task_info[5] if task_info[5] > 0 else 1  # 防止除零
                total_time = np.ceil(complete_time / 30)
                progress = min(1.0, self.robots_work_times[robot_id] / total_time)
            else:
                progress = 0.0
            state.append(progress)

        # 2. 添加任务信息（任务的归一化位置信息，任务的归一化等待时长）
        waiting_tasks = [
            task for task_index, task in enumerate(self.tasks_array)
            if self.tasks_allocated[task_index] == 0 and task[1] <= self.time
        ]

        for i in range(min(self.task_window_size, len(waiting_tasks))):
            task = waiting_tasks[i]
            [task_index, _, site_id, task_type, destination_id, _] = task
            if task_type == 2 or task_type == 3:
                destination_pos = self.sites.sites_pos[destination_id]
            else:
                destination_pos = self.sites.sites_pos[site_id]
            state.append(destination_pos[0] / max_pos_value)  # x 坐标归一化
            state.append(destination_pos[1] / max_pos_value)  # y 坐标归一化
            state.append(self.time_wait[task_index] / 30)  # 动态归一化等待时间

        # 填充空位
        for _ in range(self.task_window_size - len(waiting_tasks)):
            state.extend([0., 0., 0.])

        return np.array(state, dtype=np.float32)

    def get_obs(self):
        """
        获取所有智能体的观测信息。
        """
        for agent_id in range(self.robots.num_robots):
            avail_action = self.get_avail_agent_actions(agent_id)
            observation = self.get_agent_obs(agent_id, avail_action)
            self.obs4marl[agent_id] = observation
        return self.obs4marl

    def get_agent_obs(self, robot_id, avail_action):
        """
        获取单个机器人的观测，包含机器人自身信息、任务窗口信息、其他机器人信息。
        """
        observation = []
        max_pos_value = 60.0  # 坐标最大值
        max_distance = 100.0  # 环境中最大曼哈顿距离
        max_task_window_size = self.task_window_size  # 假设任务窗口大小固定

        # 1. 添加机器人自身信息(位置（x,y), 状态，工作时间，机器人任务进度（任务已耗时/预计完成时间）（机器人空闲置0），机器人类别信息（one-hot, nums=5）)
        robot_pos = self.robots.robot_pos[robot_id]
        normalized_robot_pos = [robot_pos[0] / max_pos_value, robot_pos[1] / max_pos_value]
        robot_type_id = self.robots.robots_type_id[robot_id]
        robot_type_onehot = [0 for i in range(len(self.robots.robot_info))]
        robot_type_onehot[robot_type_id] = 1
        robot_work_times = self.robots_work_times[robot_id]

        # 计算任务完成进度 (0-1范围)
        if self.robots_state[robot_id] == 1:  # 忙碌状态
            task_info = self.robots.robots_tasks_info[robot_id]
            robot_complete_times = task_info[5] if task_info[5] > 0 else 1  # 防止除零
            total_times = np.ceil(robot_complete_times / 30)
            task_progress = min(1.0, robot_work_times / total_times)  # 限制最大为1
        else:  # 空闲状态
            task_progress = 0.0

        observation.extend(normalized_robot_pos)  # 归一化位置
        observation.append(self.robots_state[robot_id])  # 状态(0/1)
        observation.append(robot_work_times)
        observation.append(task_progress)  # 任务进度(0-1)
        observation.extend(robot_type_onehot)  # 机器人类别(one-hot)

        # 2. 添加任务窗口信息（任务信息（x,y,距离，已等待时间, 经验耗时均值，经验耗时标准差，任务优先级）+ 任务类型（one-hot, nums=6））
        task_features = []
        for i, task in enumerate(self.task_window):
            if all(x == 0 for x in task) or avail_action[i] == 0:  # 占位符任务或不可执行
                task_features.append([0.0]*13)  # 占位符
            else:
                [task_index, _, site_id, task_type, destination_id, _] = task
                if task_type == 2 or task_type == 3:
                    destination_pos = self.sites.sites_pos[destination_id]
                else:
                    destination_pos = self.sites.sites_pos[site_id]

                # 归一化坐标信息
                task_x = destination_pos[0] / max_pos_value
                task_y = destination_pos[1] / max_pos_value
                # 归一化距离和等待时间
                dis = (abs(robot_pos[0] - destination_pos[0]) + abs(robot_pos[1] - destination_pos[1])) / max_distance
                wait_time = self.time_wait[task_index] / 30
                task_type_features = [0]*len(self.tasks.task_info)
                task_type_features[task_type] = 1
                mean = self.robots.get_buff_skills_coff_mean(robot_id, task_type)
                std = self.robots.get_buff_skills_coff_std(robot_id, task_type)
                task_priority = self.tasks.task_priority[task_type]
                task_features.append([task_x, task_y, dis, wait_time, mean, std, task_priority]+task_type_features)

        # 填充任务窗口
        task_features += [[0.0] * 13] * (max_task_window_size - len(task_features))
        for feature in task_features[:max_task_window_size]:
            observation.extend(feature)

        # 3. 添加其他机器人信息（排除当前机器人）
        for r_id in range(self.robots.num_robots):
            if r_id == robot_id:  # 跳过当前机器人
                continue

            # 获取其他机器人信息
            other_pos = self.robots.robot_pos[r_id]
            other_state = self.robots_state[r_id]
            other_type = self.robots.robots_type_id[r_id]
            other_work_times = self.robots_work_times[r_id]
            # 计算任务进度
            if other_state == 1:  # 忙碌状态
                other_task_info = self.robots.robots_tasks_info[r_id]
                other_complete_time = other_task_info[5] if other_task_info[5] > 0 else 1
                other_total_time = np.ceil(other_complete_time / 30)
                other_progress = min(1.0, other_work_times / other_total_time)
            else:
                other_progress = 0.0

            # 机器人类别one-hot
            other_type_onehot = [0] * len(self.robots.robot_info)
            other_type_onehot[other_type] = 1

            # 添加到观测
            observation.extend([other_pos[0] / max_pos_value,  # 归一化x
                                other_pos[1] / max_pos_value,  # 归一化y
                                other_work_times,  # 工作时间
                                other_state,  # 状态(0/1)
                                other_progress])  # 任务进度(0-1)
            observation.extend(other_type_onehot)  # 类别one-hot

        # observation 280 =
        #  10 = 5  + 5      -> 当前机器人位置（x,y)，状态，工作时间，机器人任务进度（任务已耗时/预计完成时间）（机器人空闲置0），机器人类别信息（one-hot, nums=5）
        # 130 = 10 * 13     -> 10 * （任务信息（x,y,距离，已等待时间, 经验耗时均值，经验耗时标准差，任务优先级）+ 任务类型（one-hot, nums=6））
        # 140 = 14 * (5+5)  -> 其他所有机器人位置（x,y), 状态，工作时间，机器人任务进度（任务已耗时/预计完成时间）（机器人空闲置0），机器人类别信息（one-hot, nums=5）
        return np.array(observation, dtype=np.float32)

    def get_avail_agent_actions(self, agent_id):
        """
        获取当前机器人可执行的任务，并对无效任务动态掩码，返回动作掩码和惩罚权重。
        """
        avail_actions = [0] * (self.task_window_size + 1)  # 初始化动作掩码，+1 表示 "不执行任务"
        avail_actions[-1] = 1  # 默认 "不执行任务" 动作可选 0-9：任务窗对应位置任务，10：不执行任务

        # 如果机器人忙碌，直接返回
        if self.robots_state[agent_id] == 1:
            return avail_actions

        # 机器人技能
        robot_skill = self.robots.get_skills(agent_id)

        # 遍历任务窗，检查任务是否有效
        for j, task in enumerate(self.task_window):
            if task == [0, 0, 0, 0, 0, 0]:  # 判断任务是否为占位符
                continue

            task_skill_list = self.tasks.required_skills[task[3]]  # 获取任务需求技能

            # 判断技能是否匹配
            if all(rs >= ts for rs, ts in zip(robot_skill, task_skill_list)):
                avail_actions[j] = 1  # 设置任务为可选
            else:
                avail_actions[j] = 0  # 机器人技能不匹配，任务不可选
        return avail_actions

    def assign_tasks_baseline(self):
        self.update_task_window()
        self.renew_wait_time()

        actions = [-1 for i in range(self.robots.num_robots)]
        robot_ids_list = list(range(self.robots.num_robots))

        robot_positions = self.robots.robot_pos
        task_window = self.task_window

        # Assign the closest task to each robot
        for robot_id in robot_ids_list:
            robot_pos = robot_positions[robot_id]
            closest_task = None
            min_distance = float('inf')

            # Get available actions for the robot
            available_actions = self.get_avail_agent_actions(robot_id)

            # Iterate through the task window to find the closest task
            for task_index, task in enumerate(task_window):
                if available_actions[task_index] == 0:  # Skip if the task is not executable
                    continue

                task_pos = self.sites.sites_pos[task[2]]
                distance = np.linalg.norm(np.array(robot_pos) - np.array(task_pos))

                # Update the closest task
                if distance < min_distance:
                    closest_task = task_index
                    min_distance = distance
            # If no suitable task is found, choose the "do nothing" action
            actions[robot_id] = closest_task if closest_task is not None else len(task_window)
        return actions

    def step(self, actions, task_priority_reward = False):
        """
        执行智能体的动作，更新环境状态，并计算综合奖励。
        """
        self.current_step += 1
        time_step = 30  # 每个 step 的时间间隔(额外冗余 30*120=3600s， 120对应episode_limit)
        conflict_penalty = 0
        total_service_cost_penalty = 0
        service_coff_list = []
        task_priority_list = []

        freeze_dict = {
            'robots_state': self.robots_state.copy(),
            'robots_work_times': self.robots_work_times,
            'tasks_completed': self.tasks_completed.copy(),
            'tasks_allocated': self.tasks_allocated.copy(),
            'time': self.time,
            'time_wait': self.time_wait.copy(),
            'total_time_wait': self.total_time_wait,
            'total_time_on_road': self.total_time_on_road,
            'total_service_time': self.total_service_time,
            'time_on_road': self.time_on_road.copy(),
            'service_time': self.service_time.copy(),
            'service_coff': self.service_coff.copy(),
            'robots': deepcopy(self.robots),
        }   

        conflict_count = 0  # 记录冲突数量
        # 用于记录任务分配情况，检测冲突
        task_allocation = {}

        # 获取任务窗中未分配的任务集合，排除当前动作中涉及的任务
        allocated_tasks = {
            action for action in actions if action < self.task_window_size
        }
        unallocated_tasks = {
            task_index for task_index, task in enumerate(self.task_window)
            if all(x != 0 for x in task) and self.tasks_allocated[task[0]] == 0 and task_index not in allocated_tasks
        }

        # 更新所有忙碌机器人状态
        for robot_id in range(self.robots.num_robots):
            if self.robots_state[robot_id] == 1:  # 忙碌机器人
                self.robots_work_times[robot_id] += 1
                task_info = self.robots.robots_tasks_info[robot_id]
                finished = self.robots.renew_position(robot_id, task_info[3], task_info[2], task_info[4], time_step)
                if finished:
                    self.robots_state[robot_id] = 0
                    self.robots_work_times[robot_id] = 0
                    self.tasks_completed[task_info[0]] = 1  # 标记任务完成
                    task_index = task_info[0]
                    # 计算任务完成时间 = 等待时间 + 在途时间 + 服务时间
                    completion_time = (self.time_wait[task_index] +
                                       self.time_on_road[task_index] +
                                       self.service_time[task_index])
                    self.completed_tasks_time.append(completion_time)
                    task_priority_list.append(self.tasks.task_priority[task_info[3]])

        # 遍历所有机器人动作，分配任务并记录任务分配
        for robot_id, action in enumerate(actions):
            if action < self.task_window_size:  # 如果动作合法
                task_index = action
                # 检测是否发生冲突
                if task_index in task_allocation:
                    task_allocation[task_index].append(robot_id)
                else:
                    task_allocation[task_index] = [robot_id]
        # 处理冲突任务并检查未分配任务情况
        for task_index, agents in task_allocation.items():
            time_on_road = 0
            if len(agents) > 1:  # 发生冲突
                conflict_count += 1
                task = self.task_window[task_index]
                task_pos = self.sites.sites_pos[task[2]]
                agents.sort(key=lambda agent_id: abs(self.robots.robot_pos[agent_id][0] - task_pos[0]) +
                                                 abs(self.robots.robot_pos[agent_id][1] - task_pos[1]))

                chosen_agent = agents[0]  # 选择距离最近的机器人执行任务
                for agent_id in agents:
                    if agent_id == chosen_agent:
                        time_on_road, service_time, time_on_road2, service_coff = self.robots.execute_task(agent_id, task)
                        self.robots_state[agent_id] = 1
                        self.tasks_allocated[task[0]] = 1
                        self.time_on_road[task[0]] = time_on_road
                        self.service_time[task[0]] = service_time
                        self.service_coff[task[0]] = service_coff
                        service_coff_list.append(service_coff)
                    else:
                        # 对冲突机器人进行检查，如果可以执行未分配任务但未执行
                        for unallocated_task_index in unallocated_tasks:
                            unallocated_task = self.task_window[unallocated_task_index]
                            required_skills = self.tasks.required_skills[unallocated_task[3]]
                            if all(
                                rs >= ts for rs, ts in zip(self.robots.get_skills(agent_id), required_skills)
                            ):
                                break
            else:  # 没有冲突
                agent_id = agents[0]
                task = self.task_window[task_index]
                time_on_road, service_time, time_on_road2, service_coff = self.robots.execute_task(agent_id, task)
                self.robots_state[agent_id] = 1
                self.tasks_allocated[task[0]] = 1
                self.time_on_road[task[0]] = time_on_road
                self.service_time[task[0]] = service_time
                self.service_coff[task[0]] = service_coff
                service_coff_list.append(service_coff)

            self.total_time_on_road += time_on_road
            self.total_time_on_road2 += time_on_road2
            self.total_service_time += service_time

        # 更新时间步
        self.time += time_step
        self.total_time_wait = sum(self.time_wait) + self.total_time_on_road  # 累计等待时间
        shift_allocated_num = sum(self.tasks_allocated)-sum(freeze_dict['tasks_allocated'])
        shift_completed_num = sum(self.tasks_completed)-sum(freeze_dict['tasks_completed'])

        # 只计算本步骤新完成任务的完成时间惩罚
        new_completed_time_penalty = 0
        for task_index in range(len(self.tasks_array)):
            # 只处理本步骤新完成的任务
            if self.tasks_completed[task_index] == 1 and freeze_dict['tasks_completed'][task_index] == 0:
                # 惩罚与完成时间成正比，但归一化处理
                completion_time = self.completed_tasks_time[-1]  # 获取最新完成的任务时间
                normalized_time = completion_time / 1000  # 假设最大完成时间为1800秒(30分钟)
                new_completed_time_penalty -= 1.5 * normalized_time  # 最大惩罚-1.5

        if task_priority_reward and len(task_priority_list)>0:
            avg_task_priority = np.mean(task_priority_list)
        else:
            avg_task_priority = 1
        # ===== 重构奖励组件 =====
        # 1. 分配奖励：鼓励分配任务
        allocation_reward = 0.5 * shift_allocated_num

        # 2. 完成奖励：鼓励完成任务（主要目标）
        completion_reward = 2.0 * shift_completed_num * avg_task_priority

        # 3. 效率奖励：与服务系数挂钩
        efficiency_reward = np.mean(service_coff_list) if service_coff_list else 0

        # 4. 时间惩罚：仅对新完成的任务
        time_penalty = new_completed_time_penalty * avg_task_priority

        # 5. 等待惩罚：惩罚未分配任务
        wait_penalty = -0.01 * len(self.unallocated_tasks)

        # 综合全局奖励（去除冲突惩罚）
        immediate_reward = (
            allocation_reward +
            completion_reward +
            efficiency_reward +
            time_penalty +
            wait_penalty
        )

        step_reward_components = {
            "allocation": allocation_reward,
            "completion": completion_reward,
            "efficiency": efficiency_reward,
            "time_penalty": time_penalty,
            "wait_penalty": wait_penalty,
            "avg_task_priority": avg_task_priority,
            "immediate": immediate_reward
        }

        self.reward_components.append(step_reward_components)
        self.episode_immediate_reward += immediate_reward
        total_reward = immediate_reward

        # 修改done判断逻辑：达到最大步长或任务全部完成
        max_steps_reached = self.current_step >= self.episode_limit  # 新增：检查是否达到最大步长
        all_tasks_completed = sum(self.tasks_completed) == len(self.tasks_array)

        done = all_tasks_completed or max_steps_reached  # 任一条件满足即结束

        final_reward = 0
        if done:
            # 1. 完成率奖励（核心目标）
            total_tasks = len(self.tasks_array)
            task_density = total_tasks / self.robots.num_robots
            completion_rate = sum(self.tasks_completed) / len(self.tasks_array)

            # 完成率奖励函数：
            completion_bonus = 40 * completion_rate * task_density

            # 2. 时间效率奖励（次要目标）
            if sum(self.tasks_completed) > 0:
                if task_priority_reward:
                    task_priority_weight = [task_info[3] for task_info in self.tasks_array]
                    avg_completion_time = sum(time * weight for time, weight in zip(self.completed_tasks_time, task_priority_weight)) / sum(task_priority_weight)
                else:
                    avg_completion_time = sum(self.completed_tasks_time) / sum(self.tasks_completed)
                # 时间奖励函数：指数衰减奖励
                time_bonus = 300 * math.exp(-0.005 * avg_completion_time) * task_density  # 每增加100秒，奖励减半
            else:
                time_bonus = 0

            # 最终奖励
            final_reward = completion_bonus + time_bonus
            total_reward += final_reward
            self.episode_final_reward = final_reward

            # 记录最终奖励组成
            step_reward_components["final_completion"] = completion_bonus
            step_reward_components["final_time"] = time_bonus
            step_reward_components["total_final"] = final_reward
            step_reward_components["total_immediate"] = self.episode_immediate_reward

        info = {
            "done": done,
            "robots_state": self.robots_state,
            "robots_work_times": self.robots_work_times,
            "task_window": self.task_window,
            "tasks_completed": self.tasks_completed,
            "tasks_allocated": self.tasks_allocated,
            "time_wait": self.time_wait,
            "time_on_road": self.time_on_road,
            "service_time": self.service_time,
            'completed_tasks_time': self.completed_tasks_time.copy(),
            "service_coff": self.service_coff,
            'shift_allocated_num': shift_allocated_num,
            'shift_completed_num': shift_completed_num,
            'num_completed_tasks': len(self.completed_tasks_time),
            'total_reward': total_reward,
            "reward_components": step_reward_components,  # 当前步骤的奖励组成
            "episode_immediate_reward": self.episode_immediate_reward,  # 整个episode的即时奖励总和
            "episode_final_reward": self.episode_final_reward  # 整个episode的最终奖励
        }
        return total_reward, done, info

    def get_env_info(self):
        """
        动态获取环境信息，包括 n_actions, n_agents, state_shape, obs_shape 和 episode_limit。
        """
        return {
            "n_actions": self.task_window_size + 1,  # 动作数量 = 任务窗大小 + 1
            "n_agents": self.robots.num_robots,  # 机器人数量
            "state_shape": len(self.get_state()),  # 全局状态向量的长度
            "obs_shape": self.obs_shape,  # 动态观测维度
            "episode_limit": self.episode_limit
        }

