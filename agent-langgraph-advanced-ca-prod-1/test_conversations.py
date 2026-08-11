from types import SimpleNamespace

import pytest
from mlflow.types.responses import ResponsesAgentRequest

from agent_server import utils_conversations as conversations


class FakeApiClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def do(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected API call: {method} {path}")
        return self.responses.pop(0)


class FakeWorkspaceClient:
    def __init__(self, responses=(), user_id="user-42"):
        self.api_client = FakeApiClient(responses)
        self.current_user = SimpleNamespace(me=lambda: SimpleNamespace(id=user_id))


def _request(messages, *, conversation_id=None, custom_inputs=None):
    context = {"conversation_id": conversation_id} if conversation_id else None
    return ResponsesAgentRequest(
        input=messages,
        context=context,
        custom_inputs=custom_inputs,
    )


def _enable(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONVERSATIONS_ENABLED", "true")
    monkeypatch.setenv("DATABRICKS_MEMORY_STORE", "catalog.schema.store")


def test_prepare_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DATABRICKS_CONVERSATIONS_ENABLED", raising=False)

    assert conversations.prepare_conversation_turn(_request([])) is None


def test_create_conversation_is_bound_to_authenticated_user(monkeypatch):
    _enable(monkeypatch)
    client = FakeWorkspaceClient([{"id": "conv_created"}], user_id="verified-user")
    monkeypatch.setattr(conversations, "_caller", lambda: (client, "verified-user"))

    result = conversations.prepare_conversation_turn(
        _request([{"role": "user", "content": "My session color is amber."}])
    )

    assert result is not None
    assert result.conversation_id == "conv_created"
    assert result.created is True
    method, path, kwargs = client.api_client.calls[0]
    assert (method, path) == ("POST", conversations._CONVERSATIONS_PATH)
    assert kwargs["body"]["memory_store"] == {"name": "catalog.schema.store"}
    assert kwargs["body"]["scope"] == {
        "kind": "user",
        "value": "verified-user",
    }
    assert kwargs["body"]["items"] == [
        {"type": "message", "role": "user", "content": "My session color is amber."}
    ]


def test_existing_conversation_deduplicates_replayed_history(monkeypatch):
    _enable(monkeypatch)
    history = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": "My session color is amber.",
        },
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Understood.", "annotations": []}
            ],
        },
    ]
    appended = {
        "id": "msg_3",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": "What color did I choose?",
    }
    client = FakeWorkspaceClient(
        [
            {"data": history, "has_more": False, "last_id": "msg_2"},
            {"data": [appended], "has_more": False},
        ]
    )
    monkeypatch.setattr(conversations, "_caller", lambda: (client, "user-42"))

    result = conversations.prepare_conversation_turn(
        _request(
            [
                {"role": "user", "content": "My session color is amber."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "What color did I choose?"},
            ],
            conversation_id="conv_existing",
        )
    )

    assert result is not None
    assert result.items == [*history, appended]
    assert len(client.api_client.calls) == 2
    method, path, kwargs = client.api_client.calls[1]
    assert method == "POST"
    assert path.endswith("/conv_existing/items")
    assert kwargs["body"]["items"] == [
        {"type": "message", "role": "user", "content": "What color did I choose?"}
    ]


def test_latest_only_input_is_appended(monkeypatch):
    _enable(monkeypatch)
    history = [
        {"id": "msg_1", "type": "message", "role": "user", "content": "One"},
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": "Acknowledged",
        },
    ]
    appended = {
        "id": "msg_3",
        "type": "message",
        "role": "user",
        "content": "Two",
    }
    client = FakeWorkspaceClient(
        [
            {"data": history, "has_more": False},
            {"data": [appended], "has_more": False},
        ]
    )
    monkeypatch.setattr(conversations, "_caller", lambda: (client, "user-42"))

    result = conversations.prepare_conversation_turn(
        _request(
            [{"role": "user", "content": "Two"}],
            custom_inputs={"managed_conversation_id": "conv_existing"},
        )
    )

    assert result is not None
    assert result.items[-1] == appended
    assert len(client.api_client.calls) == 2


def test_retry_does_not_duplicate_unanswered_user_item(monkeypatch):
    _enable(monkeypatch)
    history = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "user",
            "content": "Please answer this.",
        }
    ]
    client = FakeWorkspaceClient([{"data": history, "has_more": False}])
    monkeypatch.setattr(conversations, "_caller", lambda: (client, "user-42"))

    result = conversations.prepare_conversation_turn(
        _request(
            [{"role": "user", "content": "Please answer this."}],
            conversation_id="conv_existing",
        )
    )

    assert result is not None
    assert result.items == history
    assert len(client.api_client.calls) == 1


def test_list_items_pages_in_chronological_order():
    first = [{"id": "msg_1", "role": "user", "content": "One"}]
    second = [{"id": "msg_2", "role": "assistant", "content": "Two"}]
    client = FakeWorkspaceClient(
        [
            {"data": first, "has_more": True, "last_id": "msg_1"},
            {"data": second, "has_more": False, "last_id": "msg_2"},
        ]
    )

    assert conversations._list_items(client, "conv_paged") == [*first, *second]
    assert client.api_client.calls[0][2]["query"] == {
        "limit": conversations._LIST_PAGE_SIZE,
        "order": "asc",
    }
    assert client.api_client.calls[1][2]["query"]["after"] == "msg_1"


def test_append_output_strips_server_fields_and_batches(monkeypatch):
    _enable(monkeypatch)
    client = FakeWorkspaceClient([{"data": []}, {"data": []}])
    monkeypatch.setattr(conversations, "_caller", lambda: (client, "user-42"))
    output = [
        {
            "id": f"generated-{index}",
            "status": "completed",
            "type": "message",
            "role": "assistant",
            "content": f"Item {index}",
        }
        for index in range(21)
    ]

    conversations.append_conversation_output("conv_existing", output)

    assert len(client.api_client.calls) == 2
    first_items = client.api_client.calls[0][2]["body"]["items"]
    second_items = client.api_client.calls[1][2]["body"]["items"]
    assert len(first_items) == 20
    assert len(second_items) == 1
    assert all(
        "id" not in item and "status" not in item
        for item in [*first_items, *second_items]
    )


def test_deployed_caller_requires_forwarded_user_token(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-v1")
    monkeypatch.setattr(conversations, "get_request_headers", lambda: {})

    with pytest.raises(RuntimeError, match="forwarded end-user token"):
        conversations._caller()
