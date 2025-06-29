"""
养老服务机器人类：机器人的种类，技能和位置信息
robot类属性：
位置信息、类型 robotID、空闲的标识符isIdle、0-1技能集合
"""
import numpy as np
from utils.util import *
from utils.sites import Sites
from collections import defaultdict, deque
from utils.util import count_path_on_road


def cal_pos(start_pos, target_pos, speed, time_span):
    renew_pos = list(start_pos)  # 创建新列表以防止修改原始列表
    x_d = abs(start_pos[0] - target_pos[0])
    y_d = abs(start_pos[1] - target_pos[1])
    d = time_span * speed

    if d <= x_d:
        renew_pos[0] += (d if start_pos[0] < target_pos[0] else -d)
    elif x_d < d <= x_d + y_d:
        res = d - x_d
        renew_pos[0] = target_pos[0]
        renew_pos[1] += (res if start_pos[1] < target_pos[1] else -res)
    else:
        renew_pos = target_pos
    return tuple(renew_pos)  # 将结果转换为元组返回


class Robots:

    def __init__(self):
        # 机器人数
        self.sites = Sites()
        self.n_wheelchair = 4
        self.n_delivery = 1
        self.n_private_delivery = 1
        self.n_company = 2
        self.n_walking = 2
        # 技能数
        self.num_skills = 5
        # 机器人名称
        self.robot_info = ["智能轮椅机器人", "开放式递送机器人", "箱式递送机器人", "通用型机器人", "辅助行走机器人"]
        # 机器人速度
        self.speed = [1.5, 1.0, 1.0, 0.8, 1.0]
        # 机器人总数
        self.num_robots = 15
        # 机器人的类型
        self.robots_type_id = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        # 机器人的技能信息
        self.robots_skills = np.zeros([self.num_robots, self.num_skills])
        # robot_sites_pos 分别对应①-⑤类型机器人的停靠点
        self.robot_sites_pos = np.array([
            [8, 30], [52, 30], [8, 10],            # 智能轮椅机器人
            [52, 10], [15, 20], [15, 20],         # 开放式递送机器人
            [45, 20], [45, 20], [45, 20],           # 私人递送机器人
            [25, 30], [35, 20], [35, 10],          # 通用型机器人
            [25, 10], [35, 30], [35, 30]           # 辅助行走机器人
        ])
        # [time_on_road_1, service_time, time_on_road_2]
        self.robots_time = np.zeros([self.num_robots, 3])
        # 机器人位置初始化
        self.robot_pos = self.robot_sites_pos
        # 机器人执行任务的信息 robots_tasks_info[i] = [task_index, requests_time, site_id, task_id, destination_id, task_complete_time]
        self.robots_tasks_info = np.zeros([self.num_robots, 6], dtype=int)
        # 获取每个机器人所掌握的技能集
        for i in range(self.num_robots):
            self.robots_skills[i] = self.get_skills(i)

        self.skill_buffers = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,  # 最多存储100条记录，自动淘汰旧数据
            'mean': 0.0,
            'M2': 0
        }))

    # 机器人能力：①辅助老人移动能力 ② 送餐能力 ③递送能力 ④ 陪伴能力 ⑤康复训练能力
    def get_skills(self, agent_id):
        robot_type_id = self.robots_type_id[agent_id]
        if robot_type_id == 0:
            return [1, 1, 1, 1, 0]   # 类型 0：智能轮椅机器人
        if robot_type_id == 1:
            return [0, 1, 1, 1, 0]   # 类型 1：开放式递送机器人
        if robot_type_id == 2:
            return [1, 1, 1, 1, 1]   # 类型 2：通用箱式机器人
        if robot_type_id == 3:
            return [1, 1, 1, 1, 1]   # 类型 3：通用人型机器人
        if robot_type_id == 4:
            return [1, 1, 1, 1, 1]   # 类型 4：辅助行走机器人

    # 机器人能力系数：①辅助老人移动能力 ② 送餐能力 ③递送能力 ④ 陪伴能力 ⑤康复训练能力
    def get_skills_coff(self, agent_id):
        robot_type_id = self.robots_type_id[agent_id]
        if robot_type_id == 0:
            return [1.2, 0.3, 0.3, 0.6, 0]   # 类型 0：智能轮椅机器人
        if robot_type_id == 1:
            return [0, 1.2, 0.6, 0.6, 0]   # 类型 1：开放式递送机器人
        if robot_type_id == 2:
            return [0.1, 0.6, 1.2, 0.6, 0.1]   # 类型 2：通用箱式机器人
        if robot_type_id == 3:
            return [0.3, 0.1, 0.1, 1.2, 0.3]   # 类型 3：通用人型机器人
        if robot_type_id == 4:
            return [0.6, 0.1, 0.1, 1, 1.2]   # 类型 4：辅助行走机器人

    def get_task_service_coff(self, agent_id, task_type):
        # 针对特定机器人和特定任务，返回具体的能力系数（除了紧急任务，任务系数与能力系数是一一对应的）
        return self.get_skills_coff(agent_id)[task_type-1] if task_type>0 else self.get_skills_coff(agent_id)[3]

    def get_buff_skills_coff_mean(self, agent_id, task_id):
        """获取技能系数（经验均值）"""
        return self.skill_buffers[agent_id][task_id]['mean']

    def get_buff_skills_coff_std(self, agent_id, task_id):
        """获取技能系数（经验标准差）"""
        buffer_info = self.skill_buffers[agent_id][task_id]
        count = buffer_info.get('count', 0)
        
        if count < 2:
            return 0.0  # 样本数小于2时标准差为0
        
        # 计算样本标准差 (使用n-1作为分母)
        return math.sqrt(buffer_info.get('M2', 0.0) / (count - 1))

    def save_buff_skills(self, agent_id, task_id, task_times):
        task_times = task_times
        buffer_info = self.skill_buffers[agent_id][task_id]
        # 获取当前统计信息
        count = buffer_info['count']
        mean = buffer_info['mean']
        M2 = buffer_info['M2']

        count += 1  # 更新计数
        delta = task_times/30 - mean  # 计算增量
        mean += delta / count  # 更新均值
        
        # 更新二阶矩 (用于计算方差和标准差)
        delta2 = task_times/30 - mean
        M2 += delta * delta2
        
        # 保存更新后的统计量
        buffer_info['count'] = count
        buffer_info['mean'] = mean
        buffer_info['M2'] = M2

    # 执行任务（暂时没有考虑任务类型为1时，紧急情况的处理）
    # 输入的范围：task_id:0-6, robot_id:0-14, site_id:0-25, destination_id:0-25
    # 返回：机器人执行完全部任务所需时长
    def execute_task(self, robot_id, task):
        # 机器人的任务信息：
        [task_index, requests_time, site_id, task_id, destination_id, service_time] = task
        # 根据能力系数计算实际服务时长(临时处理， 兼容紧急任务的情况)
        service_coff = self.get_task_service_coff(robot_id, task_id)
        real_service_time = np.ceil(service_time/service_coff)

        # 机器人的类型和速度
        robot_type_id = self.robots_type_id[robot_id]
        speed = self.speed[robot_type_id]
        # task_types = ["紧急事件", "移动辅助任务", "送餐", "私人物品递送", "情感陪护", "康复训练"]
        if task_id == 2 or task_id == 3:  # 送餐/私人物品配送：先去目标点，再去任务请求点
            time_on_road_1 = count_path_on_road(self.robot_pos[robot_id], self.sites.sites_pos[destination_id],
                                                speed)
            time_on_road_2 = count_path_on_road(self.sites.sites_pos[destination_id], self.sites.sites_pos[site_id],
                                                speed)
        else:  # 其他：先去任务请求点，再去目标点
            time_on_road_1 = count_path_on_road(self.robot_pos[robot_id], self.sites.sites_pos[site_id], speed)
            time_on_road_2 = count_path_on_road(self.sites.sites_pos[site_id], self.sites.sites_pos[destination_id],
                                                speed)

        self.save_buff_skills(robot_id, task_id, real_service_time)
        self.robots_time[robot_id] = [time_on_road_1, real_service_time, time_on_road_2]

        task_complete_time = time_on_road_1 + real_service_time + time_on_road_2
        self.robots_tasks_info[robot_id] = [task_index, requests_time, site_id, task_id, destination_id, task_complete_time]
        return time_on_road_1, real_service_time, time_on_road_2, service_coff  # 去程耗时，服务耗时，返程耗时

    # 更新机器人位置信息
    # 输入：上一个step的多智能体的联合动作，选择去哪个任务请求点，同时要有上一个step的任务列表和任务目标点列表。上一个step花费的时间。
    # 代码中需要获得机器人的ID，机器人选取的任务，机器人的目标点，设计一个机器人移动路径，并根据step移动的时间更新机器人的位置信息。
    def renew_position(self, robot_id, task_id, site_id, destination_id, time_span):
        time_table = self.robots_time[robot_id]
        time_phase1, time_phase2, time_phase3 = time_table
        robot_type_id = self.robots_type_id[robot_id]
        speed = self.speed[robot_type_id]

        if task_id == 2 or task_id == 3:
            site_pos, destination_pos = self.sites.sites_pos[destination_id], self.sites.sites_pos[site_id]
        else:
            site_pos, destination_pos = self.sites.sites_pos[site_id], self.sites.sites_pos[destination_id]

        if time_span < time_phase1:
            t_1 = time_span
            self.robot_pos[robot_id] = cal_pos(self.robot_pos[robot_id], site_pos, speed, t_1)
            self.robots_time[robot_id][0] = time_phase1 - time_span
            return 0

        elif time_phase1 <= time_span < time_phase1 + time_phase2:
            self.robot_pos[robot_id] = site_pos
            self.robots_time[robot_id][0] = 0
            self.robots_time[robot_id][1] = time_phase1 + time_phase2 - time_span
            return 0

        elif time_phase1 + time_phase2 <= time_span < time_phase1 + time_phase2 + time_phase3:
            t_3 = time_span - (time_phase1 + time_phase2)
            if time_phase1 > 0 or time_phase2 > 0:
                self.robot_pos[robot_id] = site_pos
            self.robot_pos[robot_id] = cal_pos(self.robot_pos[robot_id], destination_pos, speed, t_3)
            self.robots_time[robot_id][0] = 0
            self.robots_time[robot_id][1] = 0
            self.robots_time[robot_id][2] = time_phase3 - t_3
            return 0
        else:
            self.robot_pos[robot_id] = destination_pos
            self.robots_time[robot_id] = [0, 0, 0]
            return 1
