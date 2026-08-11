from types import SimpleNamespace

from shared_memory_eval import memory_eval


class _FakeConfig:
    host = "https://workspace.example"

    @staticmethod
    def authenticate():
        return {"Authorization": "Bearer oauth-token"}


class _FakeCurrentUser:
    @staticmethod
    def me():
        return SimpleNamespace(id="12345")


class _FakeWorkspaceClient:
    config = _FakeConfig()
    current_user = _FakeCurrentUser()


def test_configure_uses_direct_eval_scope():
    target = memory_eval.configure(
        app_url="https://app.example/invocations",
        memory_store="catalog.schema.store",
        workspace_client=_FakeWorkspaceClient(),
    )

    assert target["app_url"] == "https://app.example"
    assert memory_eval._full_eval_scope("run-case:user-42") == "run-case:user-42"


def test_store_io_map_preserves_input_order(monkeypatch):
    monkeypatch.setattr(memory_eval, "STORE_IO_MAX_WORKERS", 3)

    result = memory_eval._store_io_map(lambda value: value * 2, [3, 1, 2])

    assert result == [6, 2, 4]


def test_remote_invocation_sends_eval_scope_and_thread(monkeypatch):
    memory_eval.configure(
        app_url="https://app.example",
        memory_store="catalog.schema.store",
        workspace_client=_FakeWorkspaceClient(),
    )
    observed = {}

    class _Response:
        status_code = 200
        ok = True
        text = '{"output": []}'

        @staticmethod
        def json():
            return {"output": []}

    def fake_post(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return _Response()

    monkeypatch.setattr(memory_eval.requests, "post", fake_post)

    response = memory_eval._invoke_turn_sync("hello", "run-case", "thread-2")

    assert response == {"output": []}
    assert observed["url"] == "https://app.example/invocations"
    assert observed["headers"]["Authorization"] == "Bearer oauth-token"
    assert observed["json"]["custom_inputs"] == {
        "eval_scope": "run-case",
        "thread_id": "thread-2",
    }


def test_remote_invocation_can_request_eval_read_only(monkeypatch):
    memory_eval.configure(
        app_url="https://app.example",
        memory_store="catalog.schema.store",
        workspace_client=_FakeWorkspaceClient(),
    )
    observed = {}

    class _Response:
        status_code = 200
        ok = True
        text = '{"output": []}'

        @staticmethod
        def json():
            return {"output": []}

    monkeypatch.setattr(
        memory_eval.requests,
        "post",
        lambda url, **kwargs: observed.update(url=url, **kwargs) or _Response(),
    )

    memory_eval._invoke_turn_sync(
        "hello",
        "run-case",
        "thread-2",
        eval_read_only=True,
    )

    assert observed["json"]["custom_inputs"] == {
        "eval_scope": "run-case",
        "thread_id": "thread-2",
        "eval_read_only": True,
    }


def test_remote_runner_waits_for_delayed_update_visibility(monkeypatch):
    path = "/memories/project.md"
    before = {
        path: {"path": path, "description": "Old date", "contents": "October 10"}
    }
    after = {
        path: {"path": path, "description": "New date", "contents": "October 24"}
    }
    responses = [before, after]
    tool_calls = [
        {
            "name": "update_memory",
            "arguments": {"path": path},
            "output": f"Updated {path}.",
        }
    ]

    monkeypatch.setattr(memory_eval, "snapshot_scope", lambda scope: responses.pop(0))
    monkeypatch.setattr(memory_eval.time, "sleep", lambda seconds: None)

    observed = memory_eval.wait_for_turn_snapshot(
        "scope", before, tool_calls, timeout_s=1, poll_interval_s=0
    )

    assert observed == after
    assert responses == []


def test_remote_runner_ignores_stale_read_on_non_mutating_turn(monkeypatch):
    path = "/memories/work/current-role.md"
    confirmed = {
        path: {
            "path": path,
            "description": "Works as shift supervisor",
            "contents": "Role: shift supervisor",
        }
    }
    stale = {
        path: {
            "path": path,
            "description": "Works as a barista",
            "contents": "Role: barista",
        }
    }
    responses = [stale, confirmed]

    monkeypatch.setattr(memory_eval, "snapshot_scope", lambda scope: responses.pop(0))
    monkeypatch.setattr(memory_eval.time, "sleep", lambda seconds: None)

    observed = memory_eval.wait_for_turn_snapshot(
        "scope",
        confirmed,
        [{"name": "search_memory", "arguments": {}, "output": "shift supervisor"}],
        timeout_s=1,
        poll_interval_s=0,
    )

    assert observed == confirmed
    assert responses == []


def test_remote_judge_retries_transient_failures(monkeypatch):
    attempts = []

    def flaky_judge(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("temporary endpoint failure")
        return "passed"

    monkeypatch.setattr(memory_eval.time, "sleep", lambda seconds: None)

    result = memory_eval._invoke_judge_with_retry(
        flaky_judge,
        inputs={"case": "x"},
        outputs={"answer": "y"},
        expectations={"required": "z"},
    )

    assert result == "passed"
    assert len(attempts) == 2
