# -*- coding: utf-8 -*-
"""对话文本落盘工具：异步追加写入 dialog.txt"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DialogFileLogger:
    """将 USER/BOT 文本按行追加写入文件（不覆盖）"""

    def __init__(self, file_path: str = "dialog.txt", max_queue_size: int = 1000):
        self.file_path = Path(file_path)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._queue.put(None)
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def log_line(self, speaker: str, text: str) -> None:
        """写入一行：`SPEAKER: text`"""
        if not text:
            return
        line = f"{speaker}: {text.strip()}\n"
        await self._queue.put(line)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            try:
                await asyncio.to_thread(self._append_line, item)
            except OSError as e:
                logger.warning("写入 dialog.txt 失败: %s", e)

    def _append_line(self, line: str) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line)
