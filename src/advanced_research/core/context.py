"""Core context management system."""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContextBlock:
    key: str
    content: str
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    expires: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return self.expires is not None and datetime.now() > self.expires

    @property
    def age_seconds(self) -> int:
        return int((datetime.now() - self.timestamp).total_seconds())


class ContextInjector:
    def __init__(self, max_context_size: int = 10000):
        self.context_blocks: Dict[str, ContextBlock] = {}
        self.max_context_size = max_context_size
        self.conversation_history: List[Dict[str, Any]] = []

    def add_context(
        self,
        key: str,
        content: str,
        priority: Priority = Priority.MEDIUM,
        expires_in_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        expires = None
        if expires_in_seconds:
            expires = datetime.now() + timedelta(seconds=expires_in_seconds)

        block = ContextBlock(
            key=key,
            content=content,
            priority=priority,
            expires=expires,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.context_blocks[key] = block
        self._cleanup_expired()

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        self._cleanup_expired()

        # Sort by priority (descending) then by timestamp (newest first)
        sorted_blocks = sorted(
            self.context_blocks.values(),
            key=lambda b: (b.priority.value, -b.timestamp.timestamp()),
            reverse=True,
        )

        context_parts = []
        current_size = 0
        max_size = max_tokens or self.max_context_size

        for block in sorted_blocks:
            if current_size + len(block.content) > max_size:
                break
            context_parts.append(f"## {block.key}\n{block.content}\n")
            current_size += len(block.content)

        return "\n".join(context_parts)

    def get_context_blocks(
        self, filter_tags: Optional[List[str]] = None
    ) -> List[ContextBlock]:
        blocks = list(self.context_blocks.values())

        if filter_tags:
            blocks = [
                block
                for block in blocks
                if any(tag in block.tags for tag in filter_tags)
            ]

        return sorted(
            blocks,
            key=lambda b: (b.priority.value, -b.timestamp.timestamp()),
            reverse=True,
        )

    def remove_context(self, key: str) -> bool:
        return self.context_blocks.pop(key, None) is not None

    def clear_context(self) -> None:
        self.context_blocks.clear()

    def add_conversation_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.conversation_history.append(message)

    def get_conversation_history(
        self,
        limit: Optional[int] = None,
        role_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        history = self.conversation_history

        if role_filter:
            history = [msg for msg in history if msg["role"] == role_filter]

        if limit:
            history = history[-limit:]

        return history

    def _cleanup_expired(self) -> None:
        expired_keys = [
            key for key, block in self.context_blocks.items() if block.is_expired
        ]
        for key in expired_keys:
            del self.context_blocks[key]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_blocks": len(self.context_blocks),
            "total_conversation_messages": len(self.conversation_history),
            "blocks_by_priority": {
                priority.name: sum(
                    1
                    for block in self.context_blocks.values()
                    if block.priority == priority
                )
                for priority in Priority
            },
            "oldest_block_age": min(
                (block.age_seconds for block in self.context_blocks.values()),
                default=0,
            ),
            "newest_block_age": max(
                (block.age_seconds for block in self.context_blocks.values()),
                default=0,
            ),
        }
