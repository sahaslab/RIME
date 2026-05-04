import sys
import asyncio
import contextlib
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass
class AudioTask:
    func: Callable
    args: tuple
    kwargs: dict
    future: asyncio.Future


class AudioProcessingQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = None

    async def start(self):
        """Start the background worker."""
        self.worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        """Stop the background worker."""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _worker(self):
        """Continually process tasks from the queue one by one."""
        while True:
            task: AudioTask = await self.queue.get()
            try:
                # The MCP stdio transport owns stdout. Any library print during a
                # queued task will corrupt the JSONRPC stream, so forward stray
                # stdout to stderr while the task runs.
                with contextlib.redirect_stdout(sys.stderr):
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        # Offload synchronous torch ops to a thread to keep server responsive
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None, lambda: task.func(*task.args, **task.kwargs)
                        )

                if not task.future.done():
                    task.future.set_result(result)
            except Exception as e:
                if not task.future.done():
                    task.future.set_exception(e)
            finally:
                self.queue.task_done()

    @property
    def is_running(self) -> bool:
        """Check if the background worker is currently active."""
        return self.worker_task is not None and not self.worker_task.done()

    async def enqueue(self, func: Callable, *args, **kwargs) -> Any:
        """Add a task to the queue and wait for the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        task = AudioTask(func=func, args=args, kwargs=kwargs, future=future)
        await self.queue.put(task)

        # Wait for the worker to process this specific task
        return await future
