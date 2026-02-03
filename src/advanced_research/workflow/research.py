"""Research workflow management system with opencode integration."""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..core.integrations import OpencodeIntegration
from ..core.context import ContextInjector, Priority


class ResearchStage(Enum):
    IDEAS = "ideas"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"


class DocumentType(Enum):
    THEORY = "theory"
    FRAMEWORK = "framework"
    EXPERIMENT = "experiment"
    PROOF = "proof"
    CODE = "code"
    ANALYSIS = "analysis"


@dataclass
class ResearchDocument:
    id: str
    title: str
    content: str
    stage: ResearchStage
    doc_type: DocumentType
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(
        default_factory=list
    )  # Document IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def semantic_tags(self) -> List[str]:
        """Generate semantic tags based on stage and type."""
        tags = [f"#{self.stage.value}", f"#{self.doc_type.value}"]
        tags.extend(self.tags)
        return list(set(tags))  # Remove duplicates


class ResearchWorkflow:
    def __init__(self, opencode_integration: OpencodeIntegration):
        self.opencode = opencode_integration
        self.documents: Dict[str, ResearchDocument] = {}
        self.workflow_rules = self._initialize_workflow_rules()

    def _initialize_workflow_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ideas_to_synthesis": {
                "trigger": "sufficient_theoretical_foundation",
                "conditions": {
                    "min_theory_docs": 3,
                    "required_tags": ["#theory", "#ideas"],
                },
                "actions": ["create_synthesis_framework", "validate_dependencies"],
            },
            "synthesis_to_implementation": {
                "trigger": "framework_complete",
                "conditions": {
                    "min_framework_docs": 2,
                    "required_tags": ["#framework", "#synthesis"],
                },
                "actions": ["create_implementation_plan", "setup_experiments"],
            },
            "implementation_to_validation": {
                "trigger": "experiments_ready",
                "conditions": {
                    "min_experiment_docs": 1,
                    "required_tags": ["#experiment", "#implementation"],
                },
                "actions": ["run_validation", "generate_results"],
            },
        }

    async def create_document(
        self,
        title: str,
        content: str,
        stage: ResearchStage,
        doc_type: DocumentType,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        doc_id = f"doc_{int(datetime.now().timestamp())}"

        document = ResearchDocument(
            id=doc_id,
            title=title,
            content=content,
            stage=stage,
            doc_type=doc_type,
            tags=tags or [],
            dependencies=dependencies or [],
            metadata=metadata or {},
        )

        self.documents[doc_id] = document

        # Create in opencode if enabled
        if self.opencode.enabled:
            await self.opencode.create_research_document(
                title=title,
                content=content,
                research_area=stage.value,
                tags=document.semantic_tags,
                metadata={
                    **(metadata or {}),
                    "doc_type": doc_type.value,
                    "dependencies": dependencies,
                },
            )

        # Check workflow triggers
        await self._check_workflow_triggers(stage)

        return doc_id

    async def _check_workflow_triggers(self, current_stage: ResearchStage) -> None:
        """Check if any workflow transitions should be triggered."""
        if current_stage == ResearchStage.IDEAS:
            await self._check_ideas_to_synthesis()
        elif current_stage == ResearchStage.SYNTHESIS:
            # await self._check_synthesis_to_implementation()
            pass  # TODO: Implement synthesis to implementation check
        elif current_stage == ResearchStage.IMPLEMENTATION:
            # await self._check_implementation_to_validation()
            pass  # TODO: Implement implementation to validation check

    async def _check_ideas_to_synthesis(self) -> None:
        """Check if conditions are met to move from Ideas to Synthesis."""
        rule = self.workflow_rules["ideas_to_synthesis"]
        conditions = rule["conditions"]

        # Count theory documents in Ideas stage
        theory_docs = [
            doc
            for doc in self.documents.values()
            if doc.stage == ResearchStage.IDEAS and doc.doc_type == DocumentType.THEORY
        ]

        if len(theory_docs) >= conditions["min_theory_docs"]:
            # Trigger the workflow transition
            await self._create_synthesis_framework(theory_docs)

    async def _create_synthesis_framework(
        self, theory_docs: List[ResearchDocument]
    ) -> None:
        """Create a synthesis framework based on theory documents."""
        # Generate synthesis framework content
        framework_content = self._generate_synthesis_content(theory_docs)

        await self.create_document(
            title="Integrated Research Framework",
            content=framework_content,
            stage=ResearchStage.SYNTHESIS,
            doc_type=DocumentType.FRAMEWORK,
            tags=["#auto-generated", "#integration"],
            dependencies=[doc.id for doc in theory_docs],
            metadata={"auto_generated": True, "source": "workflow_transition"},
        )

    def _generate_synthesis_content(self, theory_docs: List[ResearchDocument]) -> str:
        """Generate synthesis content from theory documents."""
        content_parts = [
            "# Integrated Research Framework",
            "",
            "## Overview",
            "This framework synthesizes the theoretical foundations established in the Ideas stage.",
            "",
            "## Theoretical Components",
        ]

        for doc in theory_docs:
            content_parts.extend(
                [
                    f"### {doc.title}",
                    doc.content[:200] + "...",  # First 200 chars
                    "",
                ]
            )

        content_parts.extend(
            [
                "## Synthesis Approach",
                "- Integration of geometric principles with learning algorithms",
                "- Formal verification of theoretical claims",
                "- Bridge to implementation stage",
                "",
                "## Next Steps",
                "1. Develop computational models based on this framework",
                "2. Design validation experiments",
                "3. Prepare implementation roadmap",
            ]
        )

        return "\n".join(content_parts)

    async def search_documents(
        self,
        query: str,
        stage: Optional[ResearchStage] = None,
        doc_type: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ResearchDocument]:
        """Search documents based on criteria."""
        results = []

        for doc in self.documents.values():
            # Basic text search
            if (
                query.lower() not in doc.title.lower()
                and query.lower() not in doc.content.lower()
            ):
                continue

            # Stage filter
            if stage and doc.stage != stage:
                continue

            # Type filter
            if doc_type and doc.doc_type != doc_type:
                continue

            # Tags filter
            if tags and not any(tag in doc.tags for tag in tags):
                continue

            results.append(doc)

        # Sort by relevance (simple: by creation date, newest first)
        results.sort(key=lambda d: d.created_at, reverse=True)
        return results

    def get_document_dependencies(
        self, doc_id: str, visited: Optional[set] = None
    ) -> List[str]:
        """Get all dependencies for a document (recursive)."""
        if visited is None:
            visited = set()

        if doc_id in visited or doc_id not in self.documents:
            return []

        visited.add(doc_id)
        doc = self.documents[doc_id]

        all_deps = list(doc.dependencies)
        for dep_id in doc.dependencies:
            all_deps.extend(self.get_document_dependencies(dep_id, visited))

        return list(set(all_deps))  # Remove duplicates

    async def generate_workflow_report(self) -> Dict[str, Any]:
        """Generate a comprehensive workflow status report."""
        stage_counts = {}
        type_counts = {}
        total_documents = len(self.documents)

        for doc in self.documents.values():
            stage_counts[doc.stage.value] = stage_counts.get(doc.stage.value, 0) + 1
            type_counts[doc.doc_type.value] = type_counts.get(doc.doc_type.value, 0) + 1

        # Check workflow progress
        workflow_status = {
            "ideas_complete": stage_counts.get("ideas", 0) >= 3,
            "synthesis_ready": stage_counts.get("synthesis", 0) >= 1,
            "implementation_ready": stage_counts.get("implementation", 0) >= 1,
        }

        return {
            "total_documents": total_documents,
            "stage_distribution": stage_counts,
            "type_distribution": type_counts,
            "workflow_status": workflow_status,
            "last_updated": datetime.now().isoformat(),
        }
