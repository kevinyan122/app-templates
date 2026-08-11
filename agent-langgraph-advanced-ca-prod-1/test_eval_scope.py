from types import SimpleNamespace

import pytest

from agent_server import utils_memory


def _request(eval_scope=None, **custom_inputs):
    if eval_scope is not None:
        custom_inputs["eval_scope"] = eval_scope
    return SimpleNamespace(custom_inputs=custom_inputs)


def test_direct_eval_scope_is_ignored_when_mode_is_disabled(monkeypatch):
    monkeypatch.delenv("DATABRICKS_MEMORY_EVAL_MODE", raising=False)

    assert utils_memory._direct_eval_scope(_request("case-1")) is None


def test_eval_scope_is_used_directly(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")

    scope = "memeval-search:fullpool-sonnet45-sdk900-v1:user_e33c296d1f_memeval.local"

    assert utils_memory._direct_eval_scope(_request(scope)) == scope


def test_eval_mode_without_scope_falls_back_to_normal_resolution(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")

    assert utils_memory._direct_eval_scope(_request()) is None


def test_read_only_tools_require_enabled_eval_mode_and_direct_scope(monkeypatch):
    request = _request("fixture", eval_read_only=True)
    monkeypatch.delenv("DATABRICKS_MEMORY_EVAL_MODE", raising=False)
    assert utils_memory.is_eval_read_only(request) is False

    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")
    assert utils_memory.is_eval_read_only(_request(eval_read_only=True)) is False
    assert utils_memory.is_eval_read_only(request) is True


def test_read_only_flag_must_be_boolean(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")

    with pytest.raises(ValueError, match="must be a boolean"):
        utils_memory.is_eval_read_only(_request("fixture", eval_read_only="true"))


def test_read_only_memory_tools_exclude_mutations():
    assert [tool.name for tool in utils_memory.memory_tools(read_only=True)] == [
        "search_memory",
        "get_memory",
        "list_memories",
    ]
    assert [tool.name for tool in utils_memory.memory_tools()] == [
        "search_memory",
        "save_memory",
        "get_memory",
        "list_memories",
        "update_memory",
        "delete_memory",
    ]


@pytest.mark.parametrize("selector", ["", "contains spaces", ":other-user", "x" * 129])
def test_eval_scope_rejects_unsafe_selectors(monkeypatch, selector):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")

    with pytest.raises(ValueError, match="eval_scope"):
        utils_memory._direct_eval_scope(_request(selector))


def test_resolve_scope_uses_direct_eval_scope_without_obo_lookup(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")
    monkeypatch.setattr(
        utils_memory,
        "get_request_headers",
        lambda: {"x-forwarded-access-token": "forwarded-oauth-token"},
    )
    monkeypatch.setattr(
        utils_memory,
        "get_user_workspace_client",
        lambda: SimpleNamespace(
            current_user=SimpleNamespace(me=lambda: SimpleNamespace(id="verified-42"))
        ),
    )

    assert utils_memory.resolve_scope(_request("scenario-1")) == "scenario-1"


def test_resolve_scope_uses_verified_obo_user_without_eval_scope(monkeypatch):
    monkeypatch.setenv("DATABRICKS_MEMORY_EVAL_MODE", "true")
    monkeypatch.setattr(
        utils_memory,
        "get_request_headers",
        lambda: {"x-forwarded-access-token": "forwarded-oauth-token"},
    )
    monkeypatch.setattr(
        utils_memory,
        "get_user_workspace_client",
        lambda: SimpleNamespace(
            current_user=SimpleNamespace(me=lambda: SimpleNamespace(id="verified-42"))
        ),
    )

    assert utils_memory.resolve_scope(_request()) == "verified-42"
