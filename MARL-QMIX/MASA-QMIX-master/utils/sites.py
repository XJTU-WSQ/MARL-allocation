"""
记录老人房间位置以及公共活动区域的位置
A:老人房间: 分别对应老人的1-18号房间
B:公共活动区域: 餐厅，室外活动区南，室外活动区北，活动室，浴室，值班室，西北卫生间，东南卫生间。
"""
import numpy as np


# Sites里面存储的是任务请求点和任务目标点的位置信息
class Sites:
    def __init__(self):
        # 老人房间: 分别对应老人的1-12号房间
        self.rooms_pos = np.array([
            [5, 5], [5, 15], [5, 25], [5, 35],  # 左侧房间
            [55, 5], [55, 15], [55, 25], [55, 35],  # 右侧房间
            [27.5, 5], [32.5, 5], [27.5, 35], [32.5, 35]
        ])
        # 公共活动区域：餐厅，室外活动区，活动室，医护室，卫生间。
        self.public_sites_pos = np.array([
            [15, 10],  # 餐厅中心
            [15, 30],  # 医护室中心
            [30, 20],  # 室外活动区中心
            [45, 10],  # 活动室中心
            [45, 30]   # 卫生间中心
        ])
        # 养老机构中所有任务请求点和任务目标点位置：
        self.sites_pos = np.concatenate((self.rooms_pos, self.public_sites_pos), axis=0)
        self.n_rooms = len(self.rooms_pos)
        self.n_public_sites = len(self.public_sites_pos)
        self.num_sites = self.n_public_sites + self.n_rooms
