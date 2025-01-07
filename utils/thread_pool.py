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
