"""Enhanced context injector with LRS integration."""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .context import ContextInjector, ContextBlock, Priority
from .integrations import LRSIntegration


class LRSContextInjector(ContextInjector):
    def __init__(self, lrs_integration: LRSIntegration, max_context_size: int = 10000):
        super().__init__(max_context_size)
        self.lrs_integration = lrs_integration
        self.user_id = None

    def set_user(self, user_id: str, user_profile: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.user_profile = user_profile or {"id": user_id}

    def add_context(
        self,
        key: str,
        content: str,
        priority: Priority = Priority.MEDIUM,
        expires_in_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        track_learning: bool = True,
    ) -> None:
        super().add_context(key, content, priority, expires_in_seconds, tags, metadata)

        # Track learning interaction if enabled and user is set
        if track_learning and self.user_id and self.lrs_integration.enabled:
            context_block = self.context_blocks[key]
            # Record context access for learning analytics
            asyncio.create_task(
                self.lrs_integration.record_context_interaction(
                    context_block, self.user_id, "created"
                )
            )

    def get_context(
        self, max_tokens: Optional[int] = None, track_access: bool = True
    ) -> str:
        context = super().get_context(max_tokens)

        # Track context access for learning analytics
        if track_access and self.user_id and self.lrs_integration.enabled:
            # Track which context blocks were accessed
            accessed_blocks = self.get_context_blocks()
            for block in accessed_blocks:
                asyncio.create_task(
                    self.lrs_integration.record_context_interaction(
                        block, self.user_id, "accessed"
                    )
                )

        return context

    def add_conversation_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        track_learning: bool = True,
    ) -> None:
        super().add_conversation_message(role, content, metadata)

        # Track conversation participation
        if track_learning and self.user_id and self.lrs_integration.enabled:
            actor = {"account": {"homePage": "advanced-research", "name": self.user_id}}

            object_data = {
                "id": f"urn:conversation:{len(self.conversation_history)}",
                "objectType": "Activity",
                "definition": {
                    "name": {"en-US": f"Conversation - {role}"},
                    "description": {"en-US": content[:100]},
                    "type": "http://adlnet.gov/expapi/activities/conversation",
                },
            }

            asyncio.create_task(
                self.lrs_integration.record_learning_event(
                    actor, "communicated", object_data
                )
            )
