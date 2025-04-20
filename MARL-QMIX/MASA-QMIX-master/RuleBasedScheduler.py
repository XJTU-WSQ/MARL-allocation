import numpy as np
import pickle
from environment import ScheduleEnv
from task.task_generator import generate_tasks


class RuleBasedScheduler:
    def __init__(self, env: ScheduleEnv):
        """
        Initialize the rule-based scheduler.
        :param env: Instance of the scheduling environment.
        """
        self.env = env

    def assign_tasks(self):
        """
        Assign tasks to robots based on the closest observable task.
        """
        # Update the environment's task window and wait times
        self.env.update_task_window()
        self.env.renew_wait_time()

        robot_positions = self.env.robots.robot_pos
        task_window = self.env.task_window
        actions = []

        # Assign the closest task to each robot
        for robot_id, robot_pos in enumerate(robot_positions):
            closest_task = None
            min_distance = float('inf')

            # Get available actions for the robot
            available_actions = self.env.get_avail_agent_actions(robot_id)

            # Iterate through the task window to find the closest task
            for task_index, task in enumerate(task_window):
                if available_actions[task_index] == 0:  # Skip if the task is not executable
                    continue

                task_pos = self.env.sites.sites_pos[task[2]]
                distance = np.linalg.norm(np.array(robot_pos) - np.array(task_pos))

                # Update the closest task
                if distance < min_distance:
                    closest_task = task_index
                    min_distance = distance

            # If no suitable task is found, choose the "do nothing" action
            actions.append(closest_task if closest_task is not None else len(task_window))

        return actions

    def step(self, actions):
        """
        Execute the assigned tasks and update the environment state.
        :param actions: List of actions for each robot.
        :return: Total reward, whether the episode is done, and additional info.
        """
        time_step = 30  # Time step duration
        conflict_penalty = 0
        total_service_cost_penalty = 0
        conflict_count = 0
        task_allocation = {}

        # Update busy robots and check task completion
        for robot_id in range(self.env.robots.num_robots):
            if self.env.robots_state[robot_id] == 1:  # If the robot is busy
                task_info = self.env.robots.robots_tasks_info[robot_id]
                finished = self.env.robots.renew_position(
                    robot_id, task_info[3], task_info[2], task_info[4], time_step
                )
                if finished:
                    self.env.robots_state[robot_id] = 0
                    self.env.tasks_completed[task_info[0]] = 1

        # Assign tasks to robots and handle conflicts
        for robot_id, action in enumerate(actions):
            if action < self.env.task_window_size:  # Valid task
                task_index = action
                if task_index in task_allocation:
                    task_allocation[task_index].append(robot_id)
                else:
                    task_allocation[task_index] = [robot_id]

        for task_index, agents in task_allocation.items():
            time_on_road = 0
            if len(agents) > 1:  # Conflict occurred
                conflict_count += 1
                # Randomly choose one robot to execute the task
                chosen_agent = np.random.choice(agents)

                for agent_id in agents:
                    if agent_id == chosen_agent:
                        time_on_road, _ = self.env.robots.execute_task(
                            agent_id, self.env.task_window[task_index]
                        )
                        self.env.robots_state[agent_id] = 1
                        self.env.tasks_allocated[self.env.task_window[task_index][0]] = 1
                        total_service_cost_penalty += time_on_road * self.env.service_cost_penalty_weight
                    else:
                        conflict_penalty += self.env.conflict_penalty_weight  # Penalty for conflict
            else:  # No conflict
                agent_id = agents[0]
                time_on_road, _ = self.env.robots.execute_task(
                    agent_id, self.env.task_window[task_index]
                )
                self.env.robots_state[agent_id] = 1
                self.env.tasks_allocated[self.env.task_window[task_index][0]] = 1
                total_service_cost_penalty += time_on_road * self.env.service_cost_penalty_weight
            self.env.total_time_on_road += time_on_road

        # Update global time and calculate rewards
        self.env.time += time_step
        self.env.total_time_wait = sum(self.env.time_wait) + self.env.total_time_on_road # Update total wait time

        concurrent_rewards = self.env.calc_concurrent_rewards(task_allocation)
        total_wait_penalty = self.env.calc_total_wait_penalty()
        total_reward = concurrent_rewards + conflict_penalty + total_service_cost_penalty + total_wait_penalty
        # total_reward = (2 * len(task_allocation) - conflict_penalty - total_service_cost_penalty)
        done = self.env.time > self.env.tasks_array[-1][1] and sum(self.env.tasks_completed) == len(self.env.tasks_array)

        info = {
            "conflict_count": conflict_count,
            "concurrent_rewards": concurrent_rewards,
            "conflict_penalty": conflict_penalty,
            "total_service_cost_penalty": total_service_cost_penalty,
            "tasks_completed": sum(self.env.tasks_completed),
            "total_time_wait": self.env.total_time_wait,  # Include total wait time in info
            "total_wait_penalty": total_wait_penalty,
            "done": done,
        }

        return total_reward, done, info

    def evaluate_scheduler(self, tasks):
        """
        Evaluate the rule-based scheduler: compute total wait time and task completion rate.
        :param tasks: Array of tasks.
        :return: Total wait time and completion rate.
        """
        # Reset environment with new tasks
        self.env.tasks_array = tasks
        self.env.reset()  # Reset the environment to initial state

        total_wait_time = 0
        tasks_completed = 0
        total_tasks = len(tasks)
        info_list = []
        for _ in range(self.env.get_env_info()['episode_limit']):
            actions = self.assign_tasks()
            _, _, info = self.step(actions)
            info_list.append(info)
            tasks_completed = sum(self.env.tasks_completed)
            if info['done']:
                break

        total_wait_time = self.env.total_time_wait  # Update total wait time from step info
        completion_rate = tasks_completed / total_tasks
        return total_wait_time, completion_rate, info_list, total_tasks

    def evaluate_multiple_runs(self, task_file_path):
        """
        Run the scheduler on multiple task sets and compute average metrics.
        :param task_file_path: Path to the task file.
        """
        with open(task_file_path, "rb") as file:
            all_task_sets = pickle.load(file)

        total_wait_times = []
        completion_rates = []
        concurrent_rewards_list = []
        conflict_penalty_list = []
        total_service_cost_penalty_list = []
        total_wait_penalty_list = []

        for index, tasks in enumerate(all_task_sets):
            wait_time, completion_rate, info_list, total_tasks = self.evaluate_scheduler(tasks)
            total_wait_times.append(wait_time)
            completion_rates.append(completion_rate)
            print(f"Run {index + 1}: Total wait time = {wait_time}, Completion rate = {completion_rate:.2%}")
            concurrent_rewards = sum([info['concurrent_rewards'] for info in info_list])/total_tasks
            conflict_penalty = sum([info['conflict_penalty'] for info in info_list])/total_tasks
            total_service_cost_penalty = sum([info['total_service_cost_penalty'] for info in info_list])/total_tasks
            total_wait_penalty = sum([info['total_wait_penalty'] for info in info_list])/total_tasks
            concurrent_rewards_list.append(concurrent_rewards)
            conflict_penalty_list.append(conflict_penalty)
            total_service_cost_penalty_list.append(total_service_cost_penalty)
            total_wait_penalty_list.append(total_wait_penalty)
            print(f"### concurrent_rewards = {concurrent_rewards:.2}, conflict_penalty = {conflict_penalty:.2}")
            print(f"### total_service_cost_penalty = {total_service_cost_penalty:.2}, total_wait_penalty = {total_wait_penalty:.2}")

        avg_wait_time = np.mean(total_wait_times)
        avg_completion_rate = np.mean(completion_rates)

        print(f"\nAverage wait time over {len(all_task_sets)} runs: {avg_wait_time}")
        print(f"Average completion rate over {len(all_task_sets)} runs: {avg_completion_rate:.2%}")
        # 计算不同奖励或惩罚的最小值，常见分位数，最大值
        quantiles = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        print(f'## concurrent_rewards_list {np.quantile(concurrent_rewards_list, quantiles)}')
        print(f'## conflict_penalty_list {np.quantile(conflict_penalty_list, quantiles)}')
        print(f'## total_service_cost_penalty_list {np.quantile(total_service_cost_penalty_list, quantiles)}')
        print(f'## total_wait_penalty_list {np.quantile(total_wait_penalty_list, quantiles)}')
        return avg_wait_time, avg_completion_rate


if __name__ == "__main__":
    # Generate and evaluate tasks
    task_file_path = "task/task.pkl"  # Replace with the actual path to the task file
    env = ScheduleEnv()
    scheduler = RuleBasedScheduler(env)

    # Evaluate the scheduler
    scheduler.evaluate_multiple_runs(task_file_path)