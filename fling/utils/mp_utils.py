import multiprocessing


def client_trainer(client, kwargs):
    # This is the function that each client will execute
    # It will receive a task and its arguments, execute it, and return its result
    return client.train(**kwargs)


def client_tester(client, kwargs):
    # This is the function that each client will execute
    # It will receive a task and its arguments, execute it, and return its result
    return client.test(**kwargs)


def client_finetuner(client, kwargs):
    # This is the function that each client will execute
    # It will receive a task and its arguments, execute it, and return its result
    return client.finetune(**kwargs)


class MultiProcessLauncher:

    def __init__(self, num_proc: int, task_name: str):
        self.num_proc = num_proc
        self.task_name = task_name

    def launch(self, clients, **kwargs) -> list:
        tasks = [(client, kwargs) for client in clients]
        with multiprocessing.Pool(self.num_proc) as pool:
            # Use starmap to apply the worker function to every task
            # Each task is a tuple that contains the task object and the arguments
            if self.task_name == 'train':
                results = pool.starmap(client_trainer, tasks)
            elif self.task_name == 'test':
                results = pool.starmap(client_tester, tasks)
            elif self.task_name == 'finetuner':
                results = pool.starmap(client_finetuner, tasks)
            else:
                raise ValueError(f'Unrecognized task name: {self.task_name}')
        return results
