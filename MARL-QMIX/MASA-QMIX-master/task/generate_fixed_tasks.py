# Generate 100 sets of random tasks using the provided task generation function
from task_generator import generate_tasks
import pickle

# Create 100 sets of tasks
all_task_sets = []
for _ in range(1000):
    tasks = generate_tasks()
    all_task_sets.append(tasks)

# Save the task sets as a .pkl file
tasks_pkl_path = "task.pkl"
with open(tasks_pkl_path, "wb") as f:
    pickle.dump(all_task_sets, f)


