import torch
from backend.agent.tools import ToolRegistry
from backend.agent.memory import ConversationMemory
from backend.agent.sat_agent import SatQueryAgent
from backend.data.dummy_dataset import create_synthetic_signature


def test_tool_registry():
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert len(tools) >= 4

    band_tool = registry.get_tool("band_metadata")
    assert band_tool is not None
    info = band_tool.execute("B08")
    assert "B08" in info


def test_conversation_memory():
    mem = ConversationMemory(max_history=5)
    mem.add_user_message("Query 1")
    mem.add_agent_message("Response 1")

    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_sat_agent_workflow():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("forest", 64, 64)

    res = agent.chat(tile, "Assess vegetation vigor and tell me if there are any water bodies.")
    assert "query" in res
    assert "response" in res
    assert "plan" in res
    assert len(res["plan"]) > 0
    assert "Executive Finding" in res["response"] or "Spectral Index Analysis" in res["response"] or "Grounding" in res["response"] or "water" in res["response"].lower() or "candidate" in res["response"].lower()
