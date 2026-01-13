#!/usr/bin/env python3
“””
Advanced Context Injection Examples
Real-world scenarios showing how Markdown context injection improves LLM interactions
“””

from context_injector import ContextInjector, Priority
from typing import Dict, List
import json

class LLMContextualizer:
“””
A wrapper that adds context-awareness to LLM interactions
This would integrate with actual LLM APIs (OpenAI, Anthropic, etc.)
“””

```
def __init__(self):
    self.injector = ContextInjector()
    self.user_profile = {}
    self.session_metadata = {}
    
def set_user_profile(self, profile: Dict):
    """Set user profile information"""
    self.user_profile = profile
    
    # Add core user contexts
    self.injector.add_context(
        "user_role",
        profile.get("role", "User"),
        priority=Priority.HIGH,
        scope="session"
    )
    
    if "expertise" in profile:
        self.injector.add_context(
            "expertise_areas",
            profile["expertise"],
            priority=Priority.MEDIUM,
            scope="session"
        )
    
    if "preferences" in profile:
        self.injector.add_context(
            "user_preferences",
            profile["preferences"],
            priority=Priority.MEDIUM,
            scope="session"
        )

def send_message(self, message: str, metadata: Dict = None) -> Dict:
    """
    Send a message with full context injection
    In production, this would call an actual LLM API
    """
    
    # Add any message-specific metadata
    if metadata:
        for key, value in metadata.items():
            self.injector.add_context(
                key,
                value,
                priority=Priority.HIGH,
                scope="turn",
                expires_after=1
            )
    
    # Build the contextualized prompt
    contextualized_prompt = self.injector.inject_context(message)
    
    # In production, you would call the LLM API here:
    # response = llm_api.complete(contextualized_prompt)
    
    # For demo purposes, return the contextualized prompt
    return {
        "contextualized_prompt": contextualized_prompt,
        "context_summary": self.injector.get_context_summary()
    }
```

# ============================================================================

# USE CASE 1: Code Review Assistant

# ============================================================================

def demo_code_review_assistant():
“””
Demonstrates context injection for a code review assistant
Context adapts based on code quality, reviewer expertise, and project standards
“””
print(”\n” + “=”*70)
print(“USE CASE 1: AI Code Review Assistant”)
print(”=”*70 + “\n”)

```
assistant = LLMContextualizer()

# Set up user profile
assistant.set_user_profile({
    "role": "Senior Software Engineer",
    "expertise": ["Python", "System Design", "Security"],
    "preferences": {
        "code_style": "Pythonic",
        "detail_level": "thorough",
        "focus_areas": ["security", "performance", "maintainability"]
    }
})

# Add project-specific context
assistant.injector.add_context(
    "project_standards",
    {
        "Testing": "pytest, 80% coverage minimum",
        "Security": "OWASP compliance required",
        "Documentation": "Google-style docstrings",
        "Type Hints": "Required for all public APIs"
    },
    priority=Priority.HIGH,
    scope="project"
)

assistant.injector.add_context(
    "current_pr",
    {
        "Number": "#347",
        "Author": "@junior_dev",
        "Title": "Add user authentication endpoint",
        "Files Changed": 5,
        "Lines Added": 234
    },
    priority=Priority.CRITICAL,
    scope="conversation"
)

# Update conversation state
assistant.injector.update_conversation_state(
    topics=["authentication", "security", "input validation"],
    user_mood="focused",
    last_intent="review_code"
)

# Simulate a code review question
user_message = """
```

Can you review this authentication function?

```python
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username='{username}'"
    user = db.execute(query).fetchone()
    if user and user['password'] == password:
        return create_token(user)
    return None
```

“””

```
result = assistant.send_message(
    user_message,
    metadata={
        "file": "auth.py",
        "lines": "45-52",
        "severity_detected": "critical",
        "issues_found": ["SQL injection", "plaintext password comparison"]
    }
)

print(result["contextualized_prompt"])
print("\n" + "-"*70 + "\n")
print("CONTEXT SUMMARY:")
print(result["context_summary"])
```

# ============================================================================

# USE CASE 2: Technical Support Chatbot

# ============================================================================

def demo_technical_support():
“””
Demonstrates context injection for technical support
Context includes user frustration level, issue severity, and past tickets
“””
print(”\n” + “=”*70)
print(“USE CASE 2: Technical Support Chatbot”)
print(”=”*70 + “\n”)

```
support_bot = LLMContextualizer()

# Set up customer profile
support_bot.set_user_profile({
    "role": "Premium Customer",
    "expertise": ["Basic technical knowledge"],
    "preferences": {
        "communication_style": "simple explanations",
        "patience_level": "low",
        "previous_satisfaction": "mixed"
    }
})

# Add support ticket context
support_bot.injector.add_context(
    "ticket_info",
    {
        "ID": "TICK-8821",
        "Priority": "High",
        "Category": "Login Issues",
        "Time Open": "2 hours",
        "Previous Attempts": 3,
        "Status": "Escalated"
    },
    priority=Priority.CRITICAL,
    scope="conversation"
)

support_bot.injector.add_context(
    "user_history",
    [
        "Ticket #7734: Billing issue (Resolved, 5-star rating)",
        "Ticket #8201: Feature question (Resolved, 3-star rating)",
        "Ticket #8821: Current - Login issues (In Progress)"
    ],
    priority=Priority.MEDIUM,
    scope="session"
)

support_bot.injector.add_context(
    "customer_sentiment",
    "Frustrated - multiple failed login attempts, urgent business need",
    priority=Priority.HIGH,
    scope="turn",
    expires_after=2
)

# Update conversation state
support_bot.injector.update_conversation_state(
    topics=["login", "password reset", "2FA issues"],
    user_mood="frustrated",
    last_intent="escalate_issue"
)

user_message = "I still can't log in! This is the third time I'm explaining this. I need access NOW for a client meeting!"

result = support_bot.send_message(user_message)

print(result["contextualized_prompt"])
```

# ============================================================================

# USE CASE 3: Educational Tutor

# ============================================================================

def demo_educational_tutor():
“””
Demonstrates context injection for an AI tutor
Adapts to student learning level, pace, and recent struggles
“””
print(”\n” + “=”*70)
print(“USE CASE 3: AI Educational Tutor”)
print(”=”*70 + “\n”)

```
tutor = LLMContextualizer()

# Set up student profile
tutor.set_user_profile({
    "role": "College Student - Computer Science",
    "expertise": ["Basic programming", "Some algebra"],
    "preferences": {
        "learning_style": "visual examples",
        "pace": "moderate",
        "needs_encouragement": True
    }
})

# Add learning context
tutor.injector.add_context(
    "current_topic",
    "Binary Search Trees",
    priority=Priority.CRITICAL,
    scope="lesson"
)

tutor.injector.add_context(
    "learning_progress",
    {
        "Arrays": "Mastered ✅",
        "Linked Lists": "Comfortable ✅",
        "Recursion": "Struggling ⚠️",
        "Trees": "Just started 📚",
        "BST Operations": "Current topic"
    },
    priority=Priority.HIGH,
    scope="course"
)

tutor.injector.add_context(
    "recent_struggles",
    [
        "Difficulty understanding recursive traversal",
        "Confused about base cases",
        "Made 3 attempts at practice problem, all incorrect"
    ],
    priority=Priority.HIGH,
    scope="recent"
)

tutor.injector.add_context(
    "student_state",
    "Slightly discouraged but engaged. Last 3 answers were incorrect. Needs encouragement + clearer explanation.",
    priority=Priority.CRITICAL,
    scope="turn",
    expires_after=1
)

# Update conversation state
tutor.injector.update_conversation_state(
    topics=["BST", "insertion", "recursion", "traversal"],
    user_mood="discouraged",
    last_intent="understand_concept"
)

user_message = "I don't get it. Why does the recursive insert function work? Can you explain it differently?"

result = tutor.send_message(user_message)

print(result["contextualized_prompt"])
```

# ============================================================================

# USE CASE 4: Multi-turn Debugging Session

# ============================================================================

def demo_debugging_session():
“””
Demonstrates context accumulation over multiple turns
Shows how context builds up and expires naturally
“””
print(”\n” + “=”*70)
print(“USE CASE 4: Multi-turn Debugging Session”)
print(”=”*70 + “\n”)

```
debugger = LLMContextualizer()

# Set up developer profile
debugger.set_user_profile({
    "role": "Full Stack Developer",
    "expertise": ["JavaScript", "React", "Node.js", "PostgreSQL"],
    "preferences": {
        "debugging_style": "systematic",
        "code_examples": "required"
    }
})

# Add persistent context
debugger.injector.add_context(
    "application_context",
    {
        "Type": "E-commerce Platform",
        "Stack": "React + Node.js + PostgreSQL",
        "Environment": "Production",
        "Users Affected": "~500"
    },
    priority=Priority.HIGH,
    scope="session"
)

print("=== TURN 1: Initial Problem Report ===\n")

debugger.injector.add_context(
    "error_occurred",
    "500 Internal Server Error on /api/checkout",
    priority=Priority.CRITICAL,
    expires_after=3
)

debugger.injector.update_conversation_state(
    topics=["checkout", "500 error"],
    user_mood="concerned"
)

turn1 = debugger.send_message(
    "Users are reporting 500 errors when trying to check out. What should I check first?",
    metadata={
        "error_rate": "~10% of checkout attempts",
        "started": "30 minutes ago"
    }
)
print(turn1["contextualized_prompt"])

print("\n" + "="*70)
print("=== TURN 2: Following Up on Logs ===\n")

debugger.injector.add_context(
    "investigation_findings",
    "Server logs show: 'Cannot read property price of undefined'",
    priority=Priority.CRITICAL,
    expires_after=2
)

debugger.injector.update_conversation_state(
    topics=["checkout", "500 error", "logs", "undefined property"],
    user_mood="investigating"
)

turn2 = debugger.send_message(
    "I checked the logs. It says 'Cannot read property price of undefined'. Here's the code: cart.items.forEach(item => total += item.price)",
    metadata={
        "code_file": "checkout-service.js",
        "line_number": 67
    }
)
print(turn2["contextualized_prompt"])

print("\n" + "="*70)
print("=== TURN 3: Root Cause Identified ===\n")

debugger.injector.add_context(
    "root_cause",
    "Some cart items have null/undefined price when product is out of stock",
    priority=Priority.CRITICAL,
    expires_after=1
)

debugger.injector.update_conversation_state(
    topics=["checkout", "500 error", "logs", "undefined property", "out of stock"],
    user_mood="understanding"
)

turn3 = debugger.send_message(
    "Oh! I see the issue. Some products are out of stock and their price field is null. What's the best way to fix this?"
)
print(turn3["contextualized_prompt"])
```

# ============================================================================

# USE CASE 5: Context-Aware API Documentation Assistant

# ============================================================================

def demo_api_documentation_assistant():
“””
Demonstrates context injection for API documentation help
Context includes API version, user’s tech stack, and usage patterns
“””
print(”\n” + “=”*70)
print(“USE CASE 5: API Documentation Assistant”)
print(”=”*70 + “\n”)

```
api_helper = LLMContextualizer()

# Set up API user profile
api_helper.set_user_profile({
    "role": "Backend Developer",
    "expertise": ["Python", "REST APIs", "Authentication"],
    "preferences": {
        "example_language": "Python",
        "detail_level": "intermediate"
    }
})

# Add API context
api_helper.injector.add_context(
    "api_context",
    {
        "API": "Payment Gateway API",
        "Version": "v2.1",
        "User's Plan": "Professional",
        "Rate Limit": "1000 req/min",
        "Environment": "Production"
    },
    priority=Priority.HIGH,
    scope="session"
)

api_helper.injector.add_context(
    "recent_api_calls",
    [
        "POST /payments/create - Success",
        "GET /payments/{id} - Success",
        "POST /refunds/create - 400 Error",
        "POST /refunds/create - 400 Error (retry)"
    ],
    priority=Priority.HIGH,
    scope="recent"
)

api_helper.injector.add_context(
    "error_context",
    {
        "Endpoint": "POST /refunds/create",
        "Status": "400 Bad Request",
        "Error": "invalid_amount",
        "Message": "Refund amount exceeds original payment",
        "Attempts": 2
    },
    priority=Priority.CRITICAL,
    scope="turn"
)

api_helper.injector.update_conversation_state(
    topics=["refunds", "API error", "400 error"],
    user_mood="stuck",
    last_intent="fix_api_call"
)

user_message = "I keep getting 'invalid_amount' when trying to create a refund. The payment was $100 and I'm trying to refund $100. What's wrong?"

result = api_helper.send_message(user_message)

print(result["contextualized_prompt"])
```

# ============================================================================

# USE CASE 6: Token-Optimized Context (Production-Scale)

# ============================================================================

def demo_token_optimized():
“””
Demonstrates highly compressed context for production systems
where token efficiency is critical
“””
print(”\n” + “=”*70)
print(“USE CASE 6: Token-Optimized Context (Production)”)
print(”=”*70 + “\n”)

```
efficient_bot = ContextInjector()

# Ultra-compact user context
efficient_bot.add_context(
    "u",  # user
    "dev:sr|py,js,sql",  # senior dev: python, js, sql
    priority=Priority.HIGH
)

# Ultra-compact state
efficient_bot.add_context(
    "s",  # state
    "t:auth|m:debug|p:high|d:2h",  # topic:auth, mode:debug, priority:high, duration:2h
    priority=Priority.CRITICAL
)

# Ultra-compact tech context
efficient_bot.add_context(
    "tech",
    "node18|pg14|jwt|redis",
    priority=Priority.MEDIUM
)

efficient_bot.update_conversation_state(
    turn_number=7,
    topics=["auth", "jwt", "401", "debug"],
    user_mood="focused"
)

result = efficient_bot.inject_context(
    "Token validation still failing",
    compact=True,
    include_state=True
)

print(result)

# Show token savings
verbose_equivalent = """
```

User: Senior Developer with expertise in Python, JavaScript, and SQL
Current Topic: Authentication
Current Mode: Debugging
Priority: High
Duration: 2 hours
Technology Stack: Node.js 18, PostgreSQL 14, JWT, Redis
Conversation Turn: 7
Topics Discussed: authentication, JWT, 401 errors, debugging
User Mood: Focused
“””

```
print("\n" + "-"*70)
print(f"COMPACT VERSION TOKENS: ~{len(result.split())} words")
print(f"VERBOSE VERSION TOKENS: ~{len(verbose_equivalent.split())} words")
print(f"TOKEN SAVINGS: ~{len(verbose_equivalent.split()) - len(result.split())} words (~{100 - int(100*len(result.split())/len(verbose_equivalent.split()))}%)")
```

if **name** == “**main**”:
# Run all use case demos
demos = [
demo_code_review_assistant,
demo_technical_support,
demo_educational_tutor,
demo_debugging_session,
demo_api_documentation_assistant,
demo_token_optimized
]

```
for demo in demos:
    try:
        demo()
        print("\n")
    except Exception as e:
        print(f"Error in {demo.__name__}: {e}")

print("="*70)
print("All use case demonstrations completed!")
print("="*70)
```
