"""
This module implements a `ThreadPool` class to manage asynchronous task execution using a thread pool.
It provides functionality to submit tasks, process them in the background, and handle exceptions.
The thread pool supports a configurable number of workers and ensures tasks are completed during shutdown.
"""
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue

class ThreadPool:
    """
    A class to manage a pool of worker threads that process tasks from a queue.

    This class uses a `ThreadPoolExecutor` to manage a fixed number of threads 
    that process tasks concurrently. Tasks are added to a queue, and a background 
    worker thread processes the tasks from the queue.

    Args:
        max_workers (int): The maximum number of worker threads in the pool. 
            Defaults to 5.
    """
    def __init__(self, max_workers=5):
        """
        Initialize the ThreadPool.

        Args:
            max_workers (int): Maximum number of threads in the pool.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue()
        self._stop_event = threading.Event()

        # Start a background thread to process the task queue
        self.worker_thread = threading.Thread(
            target=self._process_tasks, daemon=True)
        self.worker_thread.start()

    def submit_task(self, func, *args, **kwargs) -> Future:
        """
        Submit a task to the thread pool.

        Args:
            func (callable): The function to execute asynchronously.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Future: A Future object representing the execution of the task.
        """
        future = self.executor.submit(func, *args, **kwargs)
        self.task_queue.put(future)
        return future

    def _process_tasks(self):
        """
        Process tasks from the task queue and handle exceptions.
        """
        while not self._stop_event.is_set():
            try:
                # Get a task from the queue
                future = self.task_queue.get(timeout=0.1)
                if future.done() and future.exception():
                    logging.error("Task failed with exception: %s", future.exception())
            except Queue.Empty:
                continue

    def shutdown(self, wait=True):
        """
        Shutdown the ThreadPool and wait for all tasks to complete.

        Args:
            wait (bool): If True, wait for tasks to complete before shutting down.
        """
        self._stop_event.set()
        self.worker_thread.join()

        # Shutdown the executor
        self.executor.shutdown(wait=wait)

    def __del__(self):
        """
        """
        self.shutdown()


# Define a default global thread pool
global_thread_pool = None # pylint: disable=C0103

def get_thread_pool():
    """
    """
    global global_thread_pool # pylint: disable=global-statement
    if global_thread_pool is None:
        global_thread_pool = ThreadPool()
    return global_thread_pool
 
# import time
# import logging
# from thread_pool_module import ThreadPool  # Assume your ThreadPool class is in this module
# 
# # Set up logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# 
# # Define a sample task
# def sample_task(name, duration):
#     logging.info("Task %s started, will take %d seconds.", name, duration)
#     time.sleep(duration)
#     logging.info("Task %s completed.", name)
#     return f"Result from {name}"
# 
# # Create a ThreadPool instance
# thread_pool = ThreadPool(max_workers=3)
# 
# # Submit tasks to the pool
# futures = []
# for i in range(5):
#     future = thread_pool.submit_task(sample_task, f"Task-{i+1}", i+1)
#     futures.append(future)
# 
# # Wait for futures and handle results
# for future in futures:
#     try:
#         # Wait for the task to complete and get the result
#         result = future.result(timeout=10)  # Timeout in seconds
#         logging.info("Future result: %s", result)
#     except Exception as e:
#         logging.error("Exception while executing a task: %s", e)
# 
# # Shutdown the thread pool
# thread_pool.shutdown(wait=True)
# 
# # Define a callback function
# def task_callback(future):
#     try:
#         result = future.result()
#         logging.info("Callback: Task completed with result: %s", result)
#     except Exception as e:
#         logging.error("Callback: Task failed with exception: %s", e)
# 
# # Submit tasks and add callbacks
# for i in range(3):
#     future = thread_pool.submit_task(sample_task, f"Callback-Task-{i+1}", i+1)
#     future.add_done_callback(task_callback)
# 
# # Shutdown the pool after ensuring all tasks are handled
# thread_pool.shutdown(wait=True)
