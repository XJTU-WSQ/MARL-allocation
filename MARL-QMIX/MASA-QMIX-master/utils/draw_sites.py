from sites import Sites
import numpy as np
import matplotlib.pyplot as plt
from robots import Robots

# 设置绘图范围
plt.xlim((0, 60))
plt.ylim((0, 40))

# 初始化地点和机器人数据
s1 = Sites()
s2 = Robots()

# 绘制老人房间位置
rooms_pos = s1.rooms_pos
x_rooms, y_rooms = np.split(rooms_pos, 2, axis=1)
plt.scatter(x_rooms, y_rooms, s=100, color='hotpink', alpha=0.8, label='Elder Rooms')

# 绘制公共活动区域位置
public_sites_pos = s1.public_sites_pos
x_sites, y_sites = np.split(public_sites_pos, 2, axis=1)
plt.scatter(x_sites, y_sites, s=150, color='blue', alpha=0.7, label='Public Areas')

# 绘制机器人初始位置
robot_pos = s2.robot_sites_pos
robot_colors = ["red", "green", "orange", "purple", "cyan"]
robot_types = ["Wheelchair", "Delivery", "Private Delivery", "Companion", "Walking"]

for i, robot_type in enumerate(robot_types):
    robot_positions = robot_pos[np.array(s2.robots_type_id) == i]
    if len(robot_positions) > 0:
        x_robots, y_robots = np.split(robot_positions, 2, axis=1)
        plt.scatter(x_robots, y_robots, s=150, color=robot_colors[i], alpha=0.8, label=f'{robot_type} Robots')

# 添加图例、标题和网格
plt.title('Updated Elderly Care Facility Layout with Robots and Task Sites')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
