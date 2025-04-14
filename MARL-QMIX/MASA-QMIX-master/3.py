import json
import csv

# 加载 JSON 文件
file_path = './episode_logs/episode_1.json'
output_txt_path = './episode_logs/episode_analysis.txt'
output_csv_path = './episode_logs/episode_analysis.csv'

# 加载 JSON 数据
with open(file_path, 'r') as file:
    data = json.load(file)

# 创建并写入优化的 TXT 文件
with open(output_txt_path, 'w') as txt_file:
    txt_file.write("Step-by-Step Analysis\n")
    txt_file.write("=" * 100 + "\n")
    for step_data in data:
        step = step_data['step']
        state = step_data['state'][:10]  # 显示前10维
        obs = step_data['obs']  # 不简化，逐行显示
        actions = step_data['actions']
        reward = step_data['reward']
        avail_actions = step_data['avail_actions']  # 不简化，逐行显示
        robots_state = step_data['robots_state']
        task_window = step_data['task_window']
        conflict_count = step_data['conflict_count']
        task_rewards = step_data['task_rewards']
        wait_penalty_raw = step_data['wait_penalty_raw']
        total_wait_penalty = step_data['total_wait_penalty']
        service_cost_penalty_raw = step_data['service_cost_penalty_raw']
        total_service_cost_penalty = step_data['total_service_cost_penalty']
        done = step_data['done']

        txt_file.write(f"Step: {step}\n")
        txt_file.write(f"State (first 10 dims): {state}\n")

        txt_file.write("Obs per robot:\n")
        for i, robot_obs in enumerate(obs):
            txt_file.write(f"  Robot {i}: {robot_obs}\n")

        txt_file.write(f"Actions: {actions}\n")
        txt_file.write(f"Reward: {reward}\n")

        txt_file.write("Available Actions per robot:\n")
        for i, robot_avail in enumerate(avail_actions):
            txt_file.write(f"  Robot {i}: {robot_avail}\n")

        txt_file.write(f"Robots State: {robots_state}\n")
        txt_file.write(f"Task Window: {task_window}\n")
        txt_file.write(f"Conflict Count: {conflict_count}\n")
        txt_file.write(f"Task Rewards: {task_rewards}\n")
        txt_file.write(f"Wait Penalty Raw: {wait_penalty_raw}\n")
        txt_file.write(f"Total Wait Penalty: {total_wait_penalty}\n")
        txt_file.write(f"Service Cost Penalty Raw: {service_cost_penalty_raw}\n")
        txt_file.write(f"Total Service Cost Penalty: {total_service_cost_penalty}\n")
        txt_file.write(f"Done: {done}\n")
        txt_file.write("=" * 100 + "\n")

print(f"Optimized TXT output saved to: {output_txt_path}")

# 创建并写入 CSV 文件
with open(output_csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入表头
    writer.writerow([
        "Step", "Robot ID", "Obs", "Actions", "Reward", "Available Actions",
        "Robots State", "Task Window", "Conflict Count", "Task Rewards",
        "Wait Penalty Raw", "Total Wait Penalty", "Service Cost Penalty Raw",
        "Total Service Cost Penalty", "Done"
    ])

    # 写入每步数据
    for step_data in data:
        step = step_data['step']
        actions = step_data['actions']
        reward = step_data['reward']
        robots_state = step_data['robots_state']
        task_window = step_data['task_window']
        conflict_count = step_data['conflict_count']
        task_rewards = step_data['task_rewards']
        wait_penalty_raw = step_data['wait_penalty_raw']
        total_wait_penalty = step_data['total_wait_penalty']
        service_cost_penalty_raw = step_data['service_cost_penalty_raw']
        total_service_cost_penalty = step_data['total_service_cost_penalty']
        done = step_data['done']

        # 遍历每个机器人写入其 obs 和 available actions
        for robot_id, (robot_obs, robot_avail) in enumerate(zip(step_data['obs'], step_data['avail_actions'])):
            writer.writerow([
                step,
                robot_id,
                robot_obs,
                actions[robot_id] if robot_id < len(actions) else None,
                reward,
                robot_avail,
                robots_state,
                task_window,
                conflict_count,
                task_rewards,
                wait_penalty_raw,
                total_wait_penalty,
                service_cost_penalty_raw,
                total_service_cost_penalty,
                done
            ])

print(f"CSV output saved to: {output_csv_path}")
