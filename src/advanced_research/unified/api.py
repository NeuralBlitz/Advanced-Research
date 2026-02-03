"""Unified API system connecting all three integration points."""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..core.integrations import (
    IntegrationsManager,
    LRSIntegration,
    OpencodeIntegration,
    NeuralBlitzIntegration,
)
from ..core.lrs_context import LRSContextInjector
from ..workflow.research import ResearchWorkflow, ResearchStage, DocumentType
from ..neuralblitz.geometric import (
    NeuralBlitzIntegration as GeometricIntegration,
    GeometricFeatures,
)


class SystemMode(Enum):
    RESEARCH = "research"
    LEARNING = "learning"
    COMPUTATION = "computation"
    COLLABORATIVE = "collaborative"


@dataclass
class UserContext:
    user_id: str
    session_id: str
    mode: SystemMode
    current_stage: Optional[ResearchStage] = None
    preferences: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


class UnifiedResearchSystem:
    """Main system that orchestrates all integrations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrations = IntegrationsManager(config)

        # Core components
        self.context_injector: Optional[LRSContextInjector] = None
        self.research_workflow: Optional[ResearchWorkflow] = None
        self.geometric_integration: Optional[GeometricIntegration] = None

        # User sessions
        self.active_sessions: Dict[str, UserContext] = {}

        # System state
        self.initialized = False
        self.system_stats = {
            "total_queries": 0,
            "documents_created": 0,
            "learning_events": 0,
            "geometric_computations": 0,
        }

    async def initialize(self) -> None:
        """Initialize all system components."""
        if self.initialized:
            return

        print("Initializing Unified Research System...")

        # Initialize all integrations
        await self.integrations.initialize_all()

        # Get integration instances
        lrs_integration = self.integrations.get_integration("lrs")
        opencode_integration = self.integrations.get_integration("opencode")
        neuralblitz_integration = self.integrations.get_integration("neuralblitz")

        # Initialize core components
        if lrs_integration and isinstance(lrs_integration, LRSIntegration):
            self.context_injector = LRSContextInjector(lrs_integration)

        if opencode_integration and isinstance(
            opencode_integration, OpencodeIntegration
        ):
            self.research_workflow = ResearchWorkflow(opencode_integration)

        if neuralblitz_integration:
            self.geometric_integration = GeometricIntegration(
                self.config.get("neuralblitz", {})
            )
            await self.geometric_integration.initialize()

        self.initialized = True
        print("System initialization complete!")

    async def shutdown(self) -> None:
        """Shutdown all system components."""
        print("Shutting down Unified Research System...")
        await self.integrations.shutdown_all()

        if self.geometric_integration:
            await self.geometric_integration.shutdown()

        self.initialized = False
        print("System shutdown complete!")

    async def create_session(
        self,
        user_id: str,
        mode: SystemMode = SystemMode.RESEARCH,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new user session."""
        session_id = f"session_{int(datetime.now().timestamp())}_{user_id}"

        user_context = UserContext(
            user_id=user_id,
            session_id=session_id,
            mode=mode,
            preferences=preferences or {},
        )

        self.active_sessions[session_id] = user_context

        # Set user for context injector
        if self.context_injector:
            self.context_injector.set_user(user_id)

        # Record session creation
        await self._record_learning_event(
            user_id,
            "session_created",
            {"session_id": session_id, "mode": mode.value},
        )

        return session_id

    async def process_query(
        self,
        session_id: str,
        query: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a user query through the unified system."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        user_context = self.active_sessions[session_id]
        self.system_stats["total_queries"] += 1

        # Add query to context
        if self.context_injector:
            self.context_injector.add_context(
                f"query_{self.system_stats['total_queries']}",
                query,
                priority=self._get_query_priority(query),
                tags=["query", user_context.mode.value],
            )

        # Process based on mode
        if user_context.mode == SystemMode.RESEARCH:
            return await self._process_research_query(user_context, query, context_data)
        elif user_context.mode == SystemMode.LEARNING:
            return await self._process_learning_query(user_context, query, context_data)
        elif user_context.mode == SystemMode.COMPUTATION:
            return await self._process_computation_query(
                user_context, query, context_data
            )
        else:  # COLLABORATIVE
            return await self._process_collaborative_query(
                user_context, query, context_data
            )

    async def _process_research_query(
        self,
        user_context: UserContext,
        query: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process research-specific queries."""
        response = {"mode": "research", "query": query, "actions": []}

        # Check if query is about creating documents
        if "create" in query.lower() and (
            "document" in query.lower() or "doc" in query.lower()
        ):
            if self.research_workflow:
                # Determine stage and type from query
                stage = self._extract_research_stage(query) or ResearchStage.IDEAS
                doc_type = self._extract_document_type(query) or DocumentType.THEORY

                # Create document
                doc_id = await self.research_workflow.create_document(
                    title=f"Auto-generated: {query[:50]}...",
                    content=query,
                    stage=stage,
                    doc_type=doc_type,
                    tags=["auto-generated", "query-driven"],
                )

                response["actions"].append(
                    {
                        "type": "document_created",
                        "document_id": doc_id,
                        "stage": stage.value,
                        "doc_type": doc_type.value,
                    }
                )

                self.system_stats["documents_created"] += 1

        # Search for relevant documents
        if "search" in query.lower() or "find" in query.lower():
            if self.research_workflow:
                search_term = self._extract_search_term(query)
                docs = await self.research_workflow.search_documents(search_term)

                response["actions"].append(
                    {
                        "type": "documents_found",
                        "documents": [
                            {"id": doc.id, "title": doc.title, "stage": doc.stage.value}
                            for doc in docs
                        ],
                    }
                )

        return response

    async def _process_learning_query(
        self,
        user_context: UserContext,
        query: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process learning-specific queries."""
        response = {"mode": "learning", "query": query, "actions": []}

        # Record learning interaction
        await self._record_learning_event(
            user_context.user_id,
            "query_asked",
            {"query": query, "mode": "learning"},
        )

        # Provide learning context
        if self.context_injector:
            context = self.context_injector.get_context(track_access=True)
            response["context"] = context

        return response

    async def _process_computation_query(
        self,
        user_context: UserContext,
        query: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process computation-specific queries."""
        response = {"mode": "computation", "query": query, "actions": []}

        if self.geometric_integration:
            # Extract computation requests
            if "geometric" in query.lower() and (
                "features" in query.lower() or "compute" in query.lower()
            ):
                # Mock input data for demonstration
                import numpy as np

                input_data = np.random.randn(10, 64)  # 10 samples, 64 dimensions

                features = await self.geometric_integration.compute_geometric_features(
                    input_data
                )
                response["actions"].append(
                    {
                        "type": "geometric_computed",
                        "features": features.to_dict(),
                    }
                )

                self.system_stats["geometric_computations"] += 1

        return response

    async def _process_collaborative_query(
        self,
        user_context: UserContext,
        query: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process collaborative queries that span multiple modes."""
        response = {"mode": "collaborative", "query": query, "actions": []}

        # Process across all systems
        research_result = await self._process_research_query(
            user_context, query, context_data
        )
        learning_result = await self._process_learning_query(
            user_context, query, context_data
        )
        computation_result = await self._process_computation_query(
            user_context, query, context_data
        )

        response["research_actions"] = research_result.get("actions", [])
        response["learning_actions"] = learning_result.get("actions", [])
        response["computation_actions"] = computation_result.get("actions", [])

        return response

    def _get_query_priority(self, query: str):
        """Determine priority based on query content."""
        query_lower = query.lower()
        if any(urgent in query_lower for urgent in ["urgent", "critical", "emergency"]):
            from ..core.context import Priority

            return Priority.CRITICAL
        elif any(important in query_lower for important in ["important", "priority"]):
            from ..core.context import Priority

            return Priority.HIGH
        from ..core.context import Priority

        return Priority.MEDIUM

    def _extract_research_stage(self, query: str) -> Optional[ResearchStage]:
        """Extract research stage from query."""
        query_lower = query.lower()
        if "idea" in query_lower or "theory" in query_lower:
            return ResearchStage.IDEAS
        elif "synthesis" in query_lower or "framework" in query_lower:
            return ResearchStage.SYNTHESIS
        elif "implementation" in query_lower or "experiment" in query_lower:
            return ResearchStage.IMPLEMENTATION
        return None

    def _extract_document_type(self, query: str) -> Optional[DocumentType]:
        """Extract document type from query."""
        query_lower = query.lower()
        if "theory" in query_lower:
            return DocumentType.THEORY
        elif "framework" in query_lower:
            return DocumentType.FRAMEWORK
        elif "experiment" in query_lower:
            return DocumentType.EXPERIMENT
        elif "code" in query_lower:
            return DocumentType.CODE
        return DocumentType.THEORY

    def _extract_search_term(self, query: str) -> str:
        """Extract search term from query."""
        # Simple extraction - in practice this would use NLP
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ["search", "find"] and i + 1 < len(words):
                return " ".join(words[i + 1 :])
        return query

    async def _record_learning_event(
        self,
        user_id: str,
        verb: str,
        object_data: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a learning event."""
        lrs_integration = self.integrations.get_integration("lrs")
        if (
            lrs_integration
            and isinstance(lrs_integration, LRSIntegration)
            and lrs_integration.enabled
        ):
            actor = {"account": {"homePage": "advanced-research", "name": user_id}}

            await lrs_integration.record_learning_event(
                actor=actor,
                verb=verb,
                object_data={
                    "id": f"urn:activity:{verb}",
                    "objectType": "Activity",
                    "definition": object_data,
                },
                result=result,
            )

            self.system_stats["learning_events"] += 1

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        workflow_report = None
        if self.research_workflow:
            workflow_report = await self.research_workflow.generate_workflow_report()

        return {
            "initialized": self.initialized,
            "active_sessions": len(self.active_sessions),
            "system_stats": self.system_stats,
            "integrations": {
                "lrs": getattr(
                    self.integrations.get_integration("lrs"), "enabled", False
                ),
                "opencode": getattr(
                    self.integrations.get_integration("opencode"), "enabled", False
                ),
                "neuralblitz": getattr(
                    self.integrations.get_integration("neuralblitz"), "enabled", False
                ),
            },
            "workflow_report": workflow_report,
            "last_updated": datetime.now().isoformat(),
        }
