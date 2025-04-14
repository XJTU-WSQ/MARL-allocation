import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义养老院布局的大小
width, height = 60, 40

# 老人房间分布在外围，共12个房间
elder_rooms = [
    (5, i, f"Room {idx+1}") for idx, i in enumerate(range(5, 40, 10))
] + [
    (60, i, f"Room {idx+5}") for idx, i in enumerate(range(5, 40, 10))
] + [
    (30, 2.5, "Room 9"), (35, 2.5, "Room 10"),  # 上方房间
    (30, 37.5, "Room 11"), (35, 37.5, "Room 12")  # 下方房间
]

# 调整活动室位置使其与老人房间中心对称
public_areas = [
    (10, 5, 10, 10, "Dining Room"),       # 餐厅
    (40, 5, 10, 10, "Activity Room"),      # 室外活动区
    (20, 15, 20, 10, "Outdoor Area"),     # 活动室调整到中心
    (10, 25, 10, 10, "Medical Room"),      # 医护值班室
    (40, 25, 10, 10, "Restroom")           # 卫生间
]

# 机器人初始位置及类型
robots = {
    "Wheelchair": [(8, 30), (52, 30), (8, 10), (52, 10)],
    "Delivery": [(15, 20)],
    "Private Delivery": [(45, 20)],
    "Companion": [(25, 30), (35, 10)],
    "Walking": [(25, 10), (35, 30)]
}

# 为每种类型的机器人按照原始顺序编号
labeled_robots = {}
for robot_type, positions in robots.items():
    labeled_robots[robot_type] = [(pos, f"{robot_type} {i+1}") for i, pos in enumerate(positions)]

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制外墙
ax.add_patch(patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='black', facecolor='none'))

# 绘制老人房间
for x, y, name in elder_rooms:
    ax.add_patch(patches.Rectangle((x - 5, y - 2.5), 5, 5, linewidth=2, edgecolor='red', facecolor='none'))
    plt.text(x - 2.5, y, name, fontsize=8, ha='center', va='center', color='red')

# 绘制公共区域
for x, y, w, h, name in public_areas:
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none'))
    plt.text(x + w / 2, y + h / 2, name, fontsize=8, ha='center', va='center', color='green')

# 绘制机器人位置
robot_colors = {"Wheelchair": "blue", "Delivery": "orange", "Private Delivery": "purple", "Companion": "cyan", "Walking": "pink"}
for robot_type, robot_list in labeled_robots.items():
    for (x, y), label in robot_list:
        plt.scatter(x, y, c=robot_colors[robot_type], s=100, label=robot_type if robot_type not in ax.get_legend_handles_labels()[1] else "")
        plt.text(x, y + 1, label, fontsize=8, ha='center', va='center', color=robot_colors[robot_type])

# 添加边界
plt.xlim(0, width)
plt.ylim(0, height)

# 添加图例和标题
plt.title('Elderly Care Facility Layout with Rooms, Robots, and Activity Areas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
# plt.grid(True)
plt.show()
plt.show()
plt.show()
