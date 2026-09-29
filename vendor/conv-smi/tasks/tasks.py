from tasks.layered_vqa import TaskLayeredVQA


def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name == "layered_vqa":
        return TaskLayeredVQA(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")

if __name__ == "__main__":
    task = get_task("layered_vqa")
    print(len(task.get_samples()))
