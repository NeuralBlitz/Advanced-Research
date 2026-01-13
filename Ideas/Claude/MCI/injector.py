#!/usr/bin/env python3
“””
Markdown Context Injector - Practical Implementation
A system for injecting structured context into LLM prompts using pure Markdown
“””

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class Priority(Enum):
“”“Context priority levels mapped to Markdown formatting”””
CRITICAL = (“🔴”, “#”, 100)
HIGH = (“🟠”, “##”, 80)
MEDIUM = (“🟡”, “###”, 60)
LOW = (“🟢”, “####”, 40)
BACKGROUND = (“⚪”, “#####”, 20)

```
def __init__(self, emoji, heading, weight):
    self.emoji = emoji
    self.heading = heading
    self.weight = weight
```

@dataclass
class ContextBlock:
“”“Represents a single context block”””
key: str
value: Any
priority: Priority = Priority.MEDIUM
scope: str = “session”
expires_after: Optional[int] = None  # turns until expiration
timestamp: datetime = field(default_factory=datetime.now)

```
def to_markdown(self, compact: bool = False) -> str:
    """Convert context block to Markdown format"""
    if compact:
        return f"`{self.key}:{self.value}`"
    
    md = f"{self.priority.heading} {self.priority.emoji} {self.key.replace('_', ' ').title()}\n\n"
    
    if isinstance(self.value, dict):
        md += self._dict_to_table(self.value)
    elif isinstance(self.value, list):
        md += self._list_to_markdown(self.value)
    else:
        md += f"**Value**: {self.value}\n"
    
    md += f"\n**Scope**: {self.scope}"
    if self.expires_after:
        md += f" | **Expires In**: {self.expires_after} turns"
    md += "\n"
    
    return md

def _dict_to_table(self, data: dict) -> str:
    """Convert dictionary to Markdown table"""
    md = "| Key | Value |\n|-----|-------|\n"
    for k, v in data.items():
        md += f"| **{k}** | {v} |\n"
    return md

def _list_to_markdown(self, data: list) -> str:
    """Convert list to Markdown bullets"""
    return "\n".join([f"- {item}" for item in data]) + "\n"
```

@dataclass
class ConversationState:
“”“Tracks conversation state over time”””
turn_number: int = 0
topics: List[str] = field(default_factory=list)
user_mood: str = “neutral”
last_intent: str = “”

```
def to_markdown(self) -> str:
    """Convert conversation state to Markdown"""
    md = "### 📊 Conversation State\n\n"
    md += f"**Turn**: {self.turn_number} | "
    md += f"**Mood**: {self.user_mood} | "
    md += f"**Last Intent**: {self.last_intent}\n\n"
    md += f"**Topics Discussed**: {' → '.join(self.topics[-5:])}\n\n"
    return md
```

class ContextInjector:
“”“Main context injection system”””

```
def __init__(self):
    self.contexts: Dict[str, ContextBlock] = {}
    self.conversation_state = ConversationState()
    self.context_history: List[str] = []
    
def add_context(
    self,
    key: str,
    value: Any,
    priority: Priority = Priority.MEDIUM,
    scope: str = "session",
    expires_after: Optional[int] = None
):
    """Add or update a context block"""
    self.contexts[key] = ContextBlock(
        key=key,
        value=value,
        priority=priority,
        scope=scope,
        expires_after=expires_after
    )
    
def remove_context(self, key: str):
    """Remove a context block"""
    if key in self.contexts:
        del self.contexts[key]

def update_conversation_state(self, **kwargs):
    """Update conversation state"""
    for key, value in kwargs.items():
        if hasattr(self.conversation_state, key):
            setattr(self.conversation_state, key, value)
    
    if 'turn_number' not in kwargs:
        self.conversation_state.turn_number += 1
    
    # Expire temporary contexts
    self._expire_contexts()

def _expire_contexts(self):
    """Remove expired contexts"""
    to_remove = []
    for key, ctx in self.contexts.items():
        if ctx.expires_after is not None:
            ctx.expires_after -= 1
            if ctx.expires_after <= 0:
                to_remove.append(key)
    
    for key in to_remove:
        del self.contexts[key]

def inject_context(
    self,
    user_message: str,
    compact: bool = False,
    include_state: bool = True
) -> str:
    """
    Inject context into a user message
    
    Args:
        user_message: The raw user message
        compact: Use compact format for tokens efficiency
        include_state: Include conversation state
        
    Returns:
        Markdown-formatted string with injected context
    """
    
    # Sort contexts by priority
    sorted_contexts = sorted(
        self.contexts.items(),
        key=lambda x: x[1].priority.weight,
        reverse=True
    )
    
    # Build the context injection
    md_parts = []
    
    # Header
    md_parts.append("# ⚡ ACTIVE CONTEXT INJECTION\n")
    md_parts.append("---\n")
    
    # Add conversation state if requested
    if include_state:
        md_parts.append(self.conversation_state.to_markdown())
        md_parts.append("---\n")
    
    # Add contexts by priority
    if compact:
        md_parts.append("## 🎯 Context Stack\n\n")
        compact_contexts = " | ".join([
            ctx.to_markdown(compact=True) 
            for _, ctx in sorted_contexts
        ])
        md_parts.append(compact_contexts + "\n\n")
    else:
        for key, ctx in sorted_contexts:
            md_parts.append(ctx.to_markdown())
            md_parts.append("\n")
    
    md_parts.append("---\n")
    md_parts.append("## 💬 USER MESSAGE\n\n")
    md_parts.append(user_message + "\n\n")
    md_parts.append("---\n")
    
    # Add response directives
    md_parts.append(self._generate_response_directives())
    
    result = "".join(md_parts)
    self.context_history.append(result)
    
    return result

def _generate_response_directives(self) -> str:
    """Generate response directives based on current context"""
    md = "## 🎯 Response Directives\n\n"
    md += "> **Based on current context, optimize response for:**\n"
    
    # Infer directives from context
    directives = []
    
    if any(ctx.priority == Priority.CRITICAL for ctx in self.contexts.values()):
        directives.append("- ⚡ **Urgency**: Address critical context immediately")
    
    if self.conversation_state.user_mood in ["frustrated", "confused"]:
        directives.append("- 🤝 **Tone**: Be empathetic and clear")
    
    if len(self.conversation_state.topics) > 3:
        directives.append("- 🔄 **Continuity**: Reference previous discussion points")
    
    if not directives:
        directives.append("- 💡 **Standard**: Provide helpful, accurate response")
    
    md += "\n".join(directives) + "\n\n"
    
    return md

def get_context_summary(self) -> str:
    """Get a summary of all active contexts"""
    md = "# 📋 Active Context Summary\n\n"
    md += f"**Total Contexts**: {len(self.contexts)}\n"
    md += f"**Conversation Turn**: {self.conversation_state.turn_number}\n\n"
    
    # Group by priority
    by_priority = {}
    for ctx in self.contexts.values():
        if ctx.priority not in by_priority:
            by_priority[ctx.priority] = []
        by_priority[ctx.priority].append(ctx.key)
    
    md += "## Contexts by Priority\n\n"
    for priority in Priority:
        if priority in by_priority:
            keys = ", ".join(by_priority[priority])
            md += f"- **{priority.emoji} {priority.name}**: {keys}\n"
    
    return md
```

def demo_basic_usage():
“”“Demonstrate basic context injection”””
print(”=”*70)
print(“DEMO 1: Basic Context Injection”)
print(”=”*70 + “\n”)

```
injector = ContextInjector()

# Add various contexts
injector.add_context(
    "user_role",
    "Senior Backend Developer",
    priority=Priority.HIGH,
    scope="session"
)

injector.add_context(
    "current_task",
    "Debugging authentication endpoint",
    priority=Priority.CRITICAL,
    scope="conversation"
)

injector.add_context(
    "tech_stack",
    {
        "Language": "Python",
        "Framework": "FastAPI",
        "Database": "PostgreSQL",
        "Auth": "JWT"
    },
    priority=Priority.MEDIUM,
    scope="project"
)

injector.add_context(
    "user_mood",
    "Frustrated - issue unresolved for 2 hours",
    priority=Priority.HIGH,
    scope="turn",
    expires_after=1
)

# Update conversation state
injector.update_conversation_state(
    topics=["authentication", "JWT", "token validation"],
    user_mood="frustrated",
    last_intent="debug_help"
)

# Inject context
user_message = "Why does my JWT validation keep failing with 401?"

result = injector.inject_context(user_message, compact=False)
print(result)
```

def demo_compact_mode():
“”“Demonstrate compact context injection for token efficiency”””
print(”\n” + “=”*70)
print(“DEMO 2: Compact Mode (Token Efficient)”)
print(”=”*70 + “\n”)

```
injector = ContextInjector()

# Add contexts
injector.add_context("user", "dev:senior", priority=Priority.HIGH)
injector.add_context("topic", "api_performance", priority=Priority.CRITICAL)
injector.add_context("lang", "node.js", priority=Priority.MEDIUM)
injector.add_context("urgency", "high", priority=Priority.HIGH)

injector.update_conversation_state(
    turn_number=5,
    topics=["db", "queries", "indexing", "caching"],
    user_mood="concerned"
)

user_message = "The query is still taking 3 seconds"

result = injector.inject_context(user_message, compact=True)
print(result)
```

def demo_temporal_context():
“”“Demonstrate temporal context with expiration”””
print(”\n” + “=”*70)
print(“DEMO 3: Temporal Context with Expiration”)
print(”=”*70 + “\n”)

```
injector = ContextInjector()

# Add persistent context
injector.add_context(
    "user_expertise",
    ["Python", "SQL", "REST APIs"],
    priority=Priority.MEDIUM,
    scope="session"
)

# Add temporary context (expires after 1 turn)
injector.add_context(
    "immediate_issue",
    "500 error on POST /api/users",
    priority=Priority.CRITICAL,
    scope="turn",
    expires_after=1
)

print("--- TURN 1 ---")
injector.update_conversation_state(topics=["error", "500"])
result1 = injector.inject_context("What's causing this error?")
print(result1)

print("\n--- TURN 2 (Temporary context expired) ---")
injector.update_conversation_state(topics=["error", "500", "logging"])
result2 = injector.inject_context("How can I debug this?")
print(result2)
```

def demo_context_summary():
“”“Demonstrate context summary view”””
print(”\n” + “=”*70)
print(“DEMO 4: Context Summary”)
print(”=”*70 + “\n”)

```
injector = ContextInjector()

# Add various contexts
injector.add_context("critical_bug", "Production down", priority=Priority.CRITICAL)
injector.add_context("user_type", "DevOps Engineer", priority=Priority.HIGH)
injector.add_context("system_status", "Degraded", priority=Priority.HIGH)
injector.add_context("tech_stack", "K8s + Docker", priority=Priority.MEDIUM)
injector.add_context("region", "us-east-1", priority=Priority.LOW)
injector.add_context("logs_link", "https://logs.example.com", priority=Priority.BACKGROUND)

summary = injector.get_context_summary()
print(summary)
```

if **name** == “**main**”:
# Run all demos
demo_basic_usage()
demo_compact_mode()
demo_temporal_context()
demo_context_summary()

```
print("\n" + "="*70)
print("All demos completed!")
print("="*70)
```
