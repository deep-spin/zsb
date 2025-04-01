from zsb.tasks import available_tasks

if __name__ == "__main__":
    print("Available tasks:\n")
    for task_name, task in available_tasks.items():
        print(f"{task_name}: {task.description}")
