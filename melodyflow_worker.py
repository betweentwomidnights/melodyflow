# rq_worker.py
import os
from redis import Redis
from rq import Worker, Queue
import torch
import gc
from contextlib import contextmanager
import psutil

@contextmanager
def gpu_task():
    """Context manager for GPU tasks"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

class GPUWorker(Worker):
    def __init__(self, queues, connection, *args, **kwargs):
        super().__init__(queues, connection=connection, *args, **kwargs)
        self.gpu_memory_threshold = 9600  # MiB - threshold for MelodyFlow

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # Convert to MiB
        return 0

    def can_process_job(self, job):
        """Check if there's enough GPU memory for the job"""
        current_usage = self.get_gpu_memory_usage()
        required_memory = job.meta.get('gpu_memory_required', self.gpu_memory_threshold)
        return current_usage < required_memory

    def perform_job(self, *args, **kwargs):
        with gpu_task():
            return super().perform_job(*args, **kwargs)

def run_worker():
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/1')
    redis_conn = Redis.from_url(redis_url)
    
    # Create queue
    queue = Queue('melodyflow', connection=redis_conn)
    
    # Create and start worker
    worker = GPUWorker([queue], connection=redis_conn)
    worker.work(with_scheduler=True)

if __name__ == '__main__':
    run_worker()
