import numpy as np
import pickle
from environment import ScheduleEnv


class RandomActionScheduler:
    def __init__(self, env: ScheduleEnv):
        """
        Initialize the random action scheduler.
        :param env: Instance of the scheduling environment.
        """
        self.env = env

    def assign_tasks(self):
        """
        Randomly assign tasks to robots from available actions.
        """
        # Update the environment's task window and wait times
        self.env.update_task_window()
        self.env.renew_wait_time()

        actions = []
        for robot_id in range(self.env.robots.num_robots):
            available_actions = self.env.get_avail_agent_actions(robot_id)

            # Randomly choose a valid action (including "do nothing")
            valid_actions = [action for action, available in enumerate(available_actions) if available == 1]
            actions.append(np.random.choice(valid_actions))

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
                        total_service_cost_penalty += time_on_road * -0.02
                    else:
                        conflict_penalty += -3  # Penalty for conflict
            else:  # No conflict
                agent_id = agents[0]
                time_on_road, _ = self.env.robots.execute_task(
                    agent_id, self.env.task_window[task_index]
                )
                self.env.robots_state[agent_id] = 1
                self.env.tasks_allocated[self.env.task_window[task_index][0]] = 1
                total_service_cost_penalty += time_on_road * -0.02
            self.env.total_time_on_road += time_on_road

        # Update global time and calculate rewards
        self.env.time += time_step
        self.env.total_time_wait = sum(self.env.time_wait) + self.env.total_time_on_road  # Update total wait time
        total_reward = (2 * len(task_allocation) - conflict_penalty - total_service_cost_penalty)
        done = self.env.time > self.env.tasks_array[-1][1] and sum(self.env.tasks_completed) == len(self.env.tasks_array)

        info = {
            "conflict_count": conflict_count,
            "conflict_penalty": conflict_penalty,
            "total_service_cost_penalty": total_service_cost_penalty,
            "tasks_completed": sum(self.env.tasks_completed),
            "total_time_wait": self.env.total_time_wait,  # Include total wait time in info
            "done": done,
        }

        return total_reward, done, info

    def evaluate_scheduler(self, tasks):
        """
        Evaluate the random action scheduler: compute total wait time and task completion rate.
        :param tasks: Array of tasks.
        :return: Total wait time and completion rate.
        """
        # Reset environment with new tasks
        self.env.tasks_array = tasks
        self.env.reset()  # Reset the environment to initial state

        total_wait_time = 0
        tasks_completed = 0
        total_tasks = len(tasks)

        for _ in range(self.env.get_env_info()['episode_limit']):
            actions = self.assign_tasks()
            _, _, info = self.step(actions)

            tasks_completed = sum(self.env.tasks_completed)
            if info['done']:
                break

        total_wait_time = self.env.total_time_wait  # Update total wait time from step info
        completion_rate = tasks_completed / total_tasks
        return total_wait_time, completion_rate

    def evaluate_multiple_runs(self, task_file_path):
        """
        Run the random scheduler on multiple task sets and compute average metrics.
        :param task_file_path: Path to the task file.
        """
        with open(task_file_path, "rb") as file:
            all_task_sets = pickle.load(file)

        total_wait_times = []
        completion_rates = []

        for index, tasks in enumerate(all_task_sets):
            wait_time, completion_rate = self.evaluate_scheduler(tasks)
            total_wait_times.append(wait_time)
            completion_rates.append(completion_rate)
            print(f"Run {index + 1}: Total wait time = {wait_time}, Completion rate = {completion_rate:.2%}")

        avg_wait_time = np.mean(total_wait_times)
        avg_completion_rate = np.mean(completion_rates)

        print(f"\nAverage wait time over {len(all_task_sets)} runs: {avg_wait_time}")
        print(f"Average completion rate over {len(all_task_sets)} runs: {avg_completion_rate:.2%}")
        return avg_wait_time, avg_completion_rate

if __name__ == "__main__":
    # Generate and evaluate tasks
    task_file_path = "task/task.pkl"  # Replace with the actual path to the task file
    env = ScheduleEnv()

    # Evaluate Random Action Scheduler
    print("Evaluating Random Action Scheduler")
    random_scheduler = RandomActionScheduler(env)
    random_scheduler.evaluate_multiple_runs(task_file_path)