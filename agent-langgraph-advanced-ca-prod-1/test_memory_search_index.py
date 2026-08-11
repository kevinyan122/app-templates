from agent_server import utils_memory
from agent_server.prompts import SYSTEM_PROMPT


def test_search_returns_only_ranked_index_fields(monkeypatch):
    observed = {}

    class FakeApiClient:
        def do(self, method, path, **kwargs):
            observed.update(method=method, path=path, **kwargs)
            return {
                "results": [
                    {
                        "score": 0.91,
                        "memory_entry": {
                            "path": "/memories/pets.md",
                            "description": "The user has a dog",
                            "contents": "The dog's name is Mochi.",
                        },
                    },
                    {
                        "score": 0.83,
                        "memory_entry": {
                            "path": "/memories/preferences/tea.md",
                            "description": "The user prefers green tea",
                            "contents": "",
                        },
                    },
                ]
            }

    class FakeWorkspaceClient:
        api_client = FakeApiClient()

    monkeypatch.setattr(utils_memory, "_ws", lambda: FakeWorkspaceClient())
    monkeypatch.setenv("DATABRICKS_MEMORY_STORE", "catalog.schema.store")

    raw_output = utils_memory._search("scope-1", "the user's preferences", 5)
    assert raw_output == (
        "2 memory matches (ranked index):\n"
        "- path: /memories/pets.md\n"
        "  description: The user has a dog\n"
        "  has_contents: true\n\n"
        "- path: /memories/preferences/tea.md\n"
        "  description: The user prefers green tea\n"
        "  has_contents: false"
    )
    assert "Mochi" not in raw_output
    assert "0.91" not in raw_output
    assert observed["query"] == {"scope": "scope-1"}
    assert observed["body"] == {"query": "the user's preferences", "top_k": 5}


def test_search_empty_result_is_empty_index(monkeypatch):
    class FakeApiClient:
        def do(self, *_args, **_kwargs):
            return {"results": []}

    class FakeWorkspaceClient:
        api_client = FakeApiClient()

    monkeypatch.setattr(utils_memory, "_ws", lambda: FakeWorkspaceClient())
    monkeypatch.setenv("DATABRICKS_MEMORY_STORE", "catalog.schema.store")

    assert utils_memory._search("scope-1", "missing topic") == (
        "No memories matched 'missing topic'."
    )


def test_prompt_and_tools_require_get_for_relevant_full_memories():
    tools = {tool.name: tool for tool in utils_memory.memory_tools()}
    search_description = " ".join(tools["search_memory"].description.split())
    get_description = " ".join(tools["get_memory"].description.split())
    system_prompt = " ".join(SYSTEM_PROMPT.split())

    assert "readable ranked list" in search_description
    assert "ONLY path, description, and has_contents" in search_description
    assert "scores and contents are not returned" in search_description
    assert "has_contents=true and is worth looking into" in search_description
    assert "call get_memory on its path" in search_description
    assert "has_contents=false" in search_description

    assert (
        "search_memory and list_memories return only path, description, and has_contents"
        in get_description
    )
    assert "has_contents=true when it is worth looking into" in get_description
    assert "has_contents=false is fully captured by its description" in get_description

    assert "On EVERY user turn, call search_memory at least once" in system_prompt
    assert "mandatory even when the request seems general, impersonal" in system_prompt
    assert "semantic query based on the user's latest request" in system_prompt
    assert "compact retrieval rewrite rather than merely repeating" in system_prompt
    assert "add one to three closely related facets" in system_prompt
    assert "never invent specific facts" in system_prompt
    assert "Aim for roughly 6-20 words" in system_prompt
    assert "never repeat the same or an equivalent query" in system_prompt
    assert "Do not repeat terms or keyword-stuff the query" in system_prompt
    assert "mandatory first retrieval step on EVERY user turn" in search_description
    assert "fully answerable from the current conversation" in search_description
    assert "first query must be based on the user's latest request" in search_description
    assert "compact retrieval rewrite rather than merely repeating" in search_description
    assert "add one to three closely related facets" in search_description
    assert "never invent specific facts" in search_description

    assert "does not return scores or full contents" in system_prompt
    assert "has_contents=true that is worth looking into" in system_prompt
    assert "Treat search as the selection step" in system_prompt
