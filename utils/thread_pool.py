from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
import threading
import logging
import time

class ThreadPool:
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
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
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
                    logging.error(f"Task failed with exception: {future.exception()}")
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

# Example Usage
if __name__ == "__main__":
    def save_frame(frame_id):
        logging.info(f"Saving frame {frame_id}")
        time.sleep(0.5)  # Simulate saving time
        logging.info(f"Frame {frame_id} saved.")

    def log_message(message):
        logging.info(f"Log: {message}")

    logging.basicConfig(level=logging.INFO)

    pool = ThreadPool(max_workers=3)

    try:
        # Submit tasks
        for i in range(10):
            pool.submit_task(save_frame, i)

        pool.submit_task(log_message, "Background logging test.")

        time.sleep(2)  # Simulate main loop processing
    finally:
        pool.shutdown()
