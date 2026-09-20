import torch
import numpy as np
from backend.agent.tools import ToolRegistry
from backend.agent.memory import ConversationMemory, SessionMemoryStore
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
    mem = ConversationMemory(max_history=5, session_id="test_sess")
    mem.add_user_message("Query 1", tile_id="tile_1")
    mem.add_agent_message("Response 1", tools_used=["spectral"], artifacts={"key": "val"}, tile_id="tile_1")

    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["tile_id"] == "tile_1"
    assert history[1]["role"] == "assistant"
    assert history[1]["tools_used"] == ["spectral"]

    clean = mem.get_history_clean()
    assert len(clean) == 2
    assert "artifacts" not in clean[1]

    last_ast = mem.get_last_assistant_turn("tile_1")
    assert last_ast is not None
    assert last_ast["content"] == "Response 1"

    # Different tile_id returns None
    assert mem.get_last_assistant_turn("tile_other") is None


def test_session_store_lru_and_isolation():
    store = SessionMemoryStore(max_sessions=3, ttl_seconds=3600.0)

    sid1, mem1 = store.get_or_create("sess_1")
    sid2, mem2 = store.get_or_create("sess_2")
    sid3, mem3 = store.get_or_create("sess_3")

    mem1.add_user_message("Hello from 1")
    mem2.add_user_message("Hello from 2")
    mem3.add_user_message("Hello from 3")

    assert len(mem1.get_history()) == 1
    assert len(mem2.get_history()) == 1
    assert mem1.get_history()[0]["content"] == "Hello from 1"
    assert mem2.get_history()[0]["content"] == "Hello from 2"

    # Adding a 4th session should evict sess_1 (least recently used)
    # Touch sess_1 and sess_2 first
    store.get("sess_1")
    store.get("sess_3")
    # Now sess_2 is oldest
    sid4, mem4 = store.get_or_create("sess_4")
    assert store.get("sess_2") is None  # Evicted
    assert store.get("sess_1") is not None
    assert store.get("sess_3") is not None
    assert store.get("sess_4") is not None


def test_same_session_keeps_history():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("forest", 64, 64)
    session_id = "test_keep_history_session"

    res1 = agent.chat(tile, "Assess vegetation vigor and health.", tile_id="tile_forest", session_id=session_id)
    assert res1["session_id"] == session_id
    assert res1["history_length"] == 2

    res2 = agent.chat(tile, "Where are the trees?", tile_id="tile_forest", session_id=session_id)
    assert res2["session_id"] == session_id
    assert res2["history_length"] == 4


def test_two_sessions_do_not_share_history():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("forest", 64, 64)

    res_a = agent.chat(tile, "Query from user A", tile_id="tile_1", session_id="session_A")
    assert res_a["session_id"] == "session_A"
    assert res_a["history_length"] == 2

    res_b = agent.chat(tile, "Query from user B", tile_id="tile_1", session_id="session_B")
    assert res_b["session_id"] == "session_B"
    assert res_b["history_length"] == 2  # Independent, not 4!


def test_follow_up_100_percent_does_not_call_spectral_tool():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("forest", 64, 64)
    session_id = "test_100_pct_session"

    # Turn 1: Classification
    res1 = agent.chat(tile, "Classify the land cover in this scene.", tile_id="tile_forest", session_id=session_id)
    assert res1["selected_task"] == "classification"

    # Turn 2: Follow-up asking why percentages don't add up to 100%
    res2 = agent.chat(tile, "Why don't those add up to 100%?", tile_id="tile_forest", session_id=session_id)
    assert res2["selected_task"] == "follow_up_explanation"
    assert res2["selected_model_or_tool"] == "ConversationMemory"
    assert "multi-label" in res2["answer"].lower() or "independent probability" in res2["answer"].lower()
    # Spectral tool was NOT called
    assert "SpectralService" not in res2["selected_model_or_tool"]


def test_which_tools_did_you_use_answers_from_memory():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("urban", 64, 64)
    session_id = "test_tools_used_session"

    # Turn 1: Grounding query
    res1 = agent.chat(tile, "Where are the buildings?", tile_id="tile_urban", session_id=session_id)
    assert res1["selected_task"] == "grounding"

    # Turn 2: Tool provenance inspection
    res2 = agent.chat(tile, "Which tools did you use?", tile_id="tile_urban", session_id=session_id)
    assert res2["selected_task"] == "tool_inspection"
    assert res2["selected_model_or_tool"] == "ConversationMemory"
    assert "grounding" in res2["answer"].lower() or "grounding_localization" in res2["answer"].lower()


def test_changing_tile_id_starts_fresh_analysis():
    agent = SatQueryAgent()
    tile_a = create_synthetic_signature("forest", 64, 64)
    tile_b = create_synthetic_signature("water", 64, 64)
    session_id = "test_tile_change_session"

    res1 = agent.chat(tile_a, "Classify this image.", tile_id="tile_alpha", session_id=session_id)
    assert res1["selected_task"] == "classification"

    # Turn 2 has a follow-up-sounding phrase but DIFFERENT tile_id -> must run fresh analysis
    res2 = agent.chat(tile_b, "Explain why this image is different.", tile_id="tile_beta", session_id=session_id)
    assert res2["selected_model_or_tool"] != "ConversationMemory"
    assert res2["tile_id"] == "tile_beta"
    assert res2["history_length"] == 4


def test_old_request_without_session_id_still_works():
    agent = SatQueryAgent()
    tile = create_synthetic_signature("forest", 64, 64)

    # Legacy request with no session_id
    res = agent.chat(tile, "Assess vegetation vigor.")
    assert "session_id" in res
    assert res["session_id"].startswith("session_")
    assert res["history_length"] == 2
    assert "answer" in res
    assert len(res["answer"]) > 0


def test_rgb_query_intent_differentiation():
    """
    Verify that on the same 3-channel RGB tile:
    - 'mark buildings' -> grounding intent, returns bounding box info or honest RGB grounding statement (never canned coverage sentence)
    - 'what is the vegetation coverage' -> rgb_coverage intent, returns vegetation percentage
    - 'describe this image' -> rgb_scene_analysis / description intent
    - 'so there is no vegetation' -> rgb_challenge intent, confirms or denies claim based on estimate
    All 4 responses must be distinct, and 'mark buildings' must not return the old coverage sentence.
    """
    agent = SatQueryAgent()
    # Create a 3-channel synthetic RGB tile (shape: (3, 64, 64))
    np.random.seed(42)
    rgb_tile = np.zeros((3, 64, 64), dtype=np.float32)
    # Give some green channel excess for vegetation
    rgb_tile[1, :32, :] = 0.6  # green
    rgb_tile[0, :32, :] = 0.2  # red
    rgb_tile[2, :32, :] = 0.2  # blue
    # Urban in bottom half
    rgb_tile[:, 32:, :] = 0.5

    session_id = "test_rgb_routing_session"
    tile_id = "rgb_test_tile"

    # 1. Grounding query
    res_mark = agent.chat(rgb_tile, "mark buildings", tile_id=tile_id, session_id=session_id)
    # 2. Coverage inquiry
    res_cov = agent.chat(rgb_tile, "what is the vegetation coverage", tile_id=tile_id, session_id=session_id)
    # 3. Scene description
    res_desc = agent.chat(rgb_tile, "describe this image", tile_id=tile_id, session_id=session_id)
    # 4. Challenge query
    res_chall = agent.chat(rgb_tile, "so there is no vegetation", tile_id=tile_id, session_id=session_id)

    # Responses must all be different
    answers = [res_mark["answer"], res_cov["answer"], res_desc["answer"], res_chall["answer"]]
    assert len(set(answers)) == 4, f"Expected 4 distinct responses, got: {answers}"

    # 'mark buildings' must not return the coverage sentence or canned phrase
    canned_coverage_phrase = "Standard 3-band RGB optical analysis reveals a landscape characterized by"
    assert canned_coverage_phrase not in res_mark["answer"]
    assert res_mark["selected_task"] == "grounding"

    # Check grounding artifacts structure
    assert "tool_artifacts" in res_mark
    assert "grounding" in res_mark["tool_artifacts"]
    assert isinstance(res_mark["tool_artifacts"]["grounding"], list)

    # Check that each response honesty note mentions RGB color heuristics and NIR/SWIR
    for r in [res_mark, res_cov, res_desc, res_chall]:
        ans_lower = r["answer"].lower()
        assert "rgb" in ans_lower or "visible" in ans_lower or "optical" in ans_lower
        assert "nir" in ans_lower or "swir" in ans_lower or "spectral" in ans_lower or "sentinel-2" in ans_lower


def test_rgb_answer_sentence_length_and_structure():
    """
    Assert that for coverage questions on one RGB tile:
    1. The main text has 2-3 sentences.
    2. Contains the asked class name and its percentage (e.g. 'Vegetation', '68.7%').
    3. Does not include uncomputed directional words (north, south, east, west, etc.).
    4. A separate '(Note: ...)' or '*(Note: ...)*' paragraph exists at the end.
    5. On a repeated question, the duplicate guard rephrases with a different sentence.
    """
    import re
    agent = SatQueryAgent()
    np.random.seed(42)
    rgb_tile = np.zeros((3, 64, 64), dtype=np.float32)
    rgb_tile[1, :32, :] = 0.6  # green
    rgb_tile[0, :32, :] = 0.2
    rgb_tile[2, :32, :] = 0.2
    rgb_tile[:, 32:, :] = 0.5

    session_id = "test_sentence_structure_session"
    tile_id = "rgb_structure_tile"

    # Turn 1: Vegetation coverage question
    res1 = agent.chat(rgb_tile, "what is the vegetation coverage", tile_id=tile_id, session_id=session_id)
    answer1 = res1["answer"]

    # Split into main text and note paragraph
    paragraphs = [p.strip() for p in answer1.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2, f"Expected main text and note paragraphs, got: {paragraphs}"
    main_text = paragraphs[0]
    note_paragraph = paragraphs[-1]

    # Check note format
    assert note_paragraph.startswith("*(Note:") or note_paragraph.startswith("(Note:"), f"Note should start with Note prefix: {note_paragraph}"

    # Check sentence count in main text (2 to 3 sentences)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', main_text) if s.strip()]
    assert 2 <= len(sentences) <= 3, f"Expected 2-3 sentences, got {len(sentences)}: {sentences}"

    # Check that main text contains asked class name and percentage format
    assert "vegetation" in main_text.lower()
    assert re.search(r'\d+\.\d+%', main_text) is not None

    # Check no directional words
    direction_words = ["north", "south", "east", "west", "northwest", "northeast", "southwest", "southeast", "top-left", "bottom-right"]
    for dir_word in direction_words:
        assert dir_word not in main_text.lower(), f"Unexpected directional word '{dir_word}' in: {main_text}"

    # Turn 2: Repeat the same question on the same tile -> Duplicate guard must rephrase
    res2 = agent.chat(rgb_tile, "what is the vegetation coverage", tile_id=tile_id, session_id=session_id)
    answer2 = res2["answer"]
    assert answer2 != answer1, "Duplicate guard should vary/rephrase response on repeated query"
    paragraphs2 = [p.strip() for p in answer2.split("\n\n") if p.strip()]
    sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraphs2[0]) if s.strip()]
    assert 2 <= len(sentences2) <= 3, f"Expected 2-3 sentences on rephrased reply, got {len(sentences2)}: {sentences2}"



