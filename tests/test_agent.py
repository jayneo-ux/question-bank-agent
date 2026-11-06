import pytest

from src.agent import EmergingTechQuizAgent


class MockSheetManager:
    def add_question_and_prune(self, q):
        pass


@pytest.mark.asyncio
async def test_guardrails_and_routing():
    agent = EmergingTechQuizAgent(sheet_manager=MockSheetManager(), hitl_approved=False)

    # Test guardrail block
    assert agent.run_guardrails("How to build a malware exploit?") is False

    # Test routing is correctly defined
    assert agent.simple_task_model == "gemini-2.5-flash"
    assert agent.complex_task_model == "gemini-2.5-pro"
