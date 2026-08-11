"""Managed short-term conversation state backed by the UC Conversations API.

This is deliberately handler-owned state, not an agent tool. The model should always
receive the current session history without having to decide whether to retrieve it.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import get_user_workspace_client

CONVERSATION_EVENT_TYPE = "response.conversation.ready"
CONVERSATION_CUSTOM_OUTPUT_KEY = "managed_conversation"

_CONVERSATIONS_PATH = "/api/2.1/unity-catalog/conversations"
_MAX_ITEMS_PER_WRITE = 20
_LIST_PAGE_SIZE = 100
_local_client: Optional[WorkspaceClient] = None


@dataclass(frozen=True)
class ManagedConversationTurn:
    conversation_id: str
    items: list[dict[str, Any]]
    created: bool

    def custom_output(self) -> dict[str, Any]:
        return {"id": self.conversation_id, "created": self.created}


def conversations_enabled() -> bool:
    return os.getenv("DATABRICKS_CONVERSATIONS_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _store_name() -> str:
    store = os.getenv("DATABRICKS_MEMORY_STORE")
    if not store:
        raise RuntimeError(
            "DATABRICKS_MEMORY_STORE is required when managed conversations are enabled."
        )
    return store


def _local_workspace_client() -> WorkspaceClient:
    global _local_client
    if _local_client is None:
        _local_client = WorkspaceClient()
    return _local_client


def _caller() -> tuple[WorkspaceClient, str]:
    """Return the authenticated caller and its verified workspace user ID.

    The beta API currently supports only user-scoped conversations and rejects a
    scope that does not match the caller. Deployed requests therefore use the OBO
    user client. Local requests use the developer's configured WorkspaceClient.
    """

    headers = get_request_headers() or {}
    if headers.get("x-forwarded-access-token"):
        client = get_user_workspace_client()
    elif os.getenv("DATABRICKS_APP_NAME"):
        raise RuntimeError(
            "Managed conversations require a forwarded end-user token in a Databricks App."
        )
    else:
        client = _local_workspace_client()

    user = client.current_user.me()
    user_id = str(getattr(user, "id", "") or "")
    if not user_id:
        raise RuntimeError("Could not resolve the authenticated workspace user ID.")
    return client, user_id


def _conversation_id(request: ResponsesAgentRequest) -> Optional[str]:
    custom_inputs = dict(request.custom_inputs or {})
    explicit = custom_inputs.get("managed_conversation_id")
    if explicit is not None:
        value = str(explicit)
        if not value.startswith("conv_"):
            raise ValueError("custom_inputs.managed_conversation_id must start with 'conv_'.")
        return value

    context_id = None
    if request.context:
        context_id = getattr(request.context, "conversation_id", None)
    if context_id and str(context_id).startswith("conv_"):
        return str(context_id)
    return None


def _request_items(request: ResponsesAgentRequest) -> list[dict[str, Any]]:
    return [_item_for_write(item.model_dump(exclude_none=True)) for item in request.input]


def _item_for_write(item: dict[str, Any]) -> dict[str, Any]:
    """Remove fields assigned by the Conversations API itself."""

    cleaned = dict(item)
    cleaned.pop("id", None)
    cleaned.pop("status", None)
    return cleaned


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _canonical_item(item: dict[str, Any]) -> tuple[Any, ...]:
    item_type = item.get("type") or "message"
    if item_type == "message":
        return ("message", item.get("role"), _message_text(item.get("content")))
    if item_type == "function_call":
        return (
            "function_call",
            item.get("call_id"),
            item.get("name"),
            item.get("arguments"),
        )
    if item_type == "function_call_output":
        return ("function_call_output", item.get("call_id"), item.get("output"))
    return (item_type, repr(_item_for_write(item)))


def _new_items(
    history: list[dict[str, Any]], request_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Accept both latest-turn-only requests and clients that replay full history."""

    if not request_items:
        return []
    if not history:
        return request_items

    # A failed/retried turn can leave the just-appended user message as the final
    # item. Do not append that same single user item twice.
    if (
        len(request_items) == 1
        and history[-1].get("role") == "user"
        and _canonical_item(request_items[0]) == _canonical_item(history[-1])
    ):
        return []

    if len(request_items) == 1:
        return request_items

    prefix_length = 0
    for persisted, supplied in zip(history, request_items):
        if _canonical_item(persisted) != _canonical_item(supplied):
            break
        prefix_length += 1

    # A replaying client supplied the stored history followed by this turn's delta.
    if prefix_length:
        return request_items[prefix_length:]
    return request_items


def _append_items(
    client: WorkspaceClient, conversation_id: str, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []
    path = f"{_CONVERSATIONS_PATH}/{conversation_id}/items"
    for start in range(0, len(items), _MAX_ITEMS_PER_WRITE):
        batch = [
            _item_for_write(item)
            for item in items[start : start + _MAX_ITEMS_PER_WRITE]
        ]
        response = client.api_client.do("POST", path, body={"items": batch})
        created.extend(response.get("data", []))
    return created


def _list_items(
    client: WorkspaceClient, conversation_id: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    after: Optional[str] = None
    path = f"{_CONVERSATIONS_PATH}/{conversation_id}/items"

    while True:
        query: dict[str, Any] = {"limit": _LIST_PAGE_SIZE, "order": "asc"}
        if after:
            query["after"] = after
        response = client.api_client.do("GET", path, query=query)
        page = response.get("data", [])
        items.extend(page)
        if not response.get("has_more"):
            return items
        after = response.get("last_id")
        if not after:
            raise RuntimeError(
                "Conversation items response had has_more=true without last_id."
            )


def prepare_conversation_turn(
    request: ResponsesAgentRequest,
) -> Optional[ManagedConversationTurn]:
    if not conversations_enabled():
        return None

    client, user_id = _caller()
    request_items = _request_items(request)
    conversation_id = _conversation_id(request)

    if conversation_id:
        history = _list_items(client, conversation_id)
        delta = _new_items(history, request_items)
        created_items = _append_items(client, conversation_id, delta) if delta else []
        return ManagedConversationTurn(
            conversation_id=conversation_id,
            items=[*history, *created_items],
            created=False,
        )

    initial_items = request_items[:_MAX_ITEMS_PER_WRITE]
    body: dict[str, Any] = {
        "memory_store": {"name": _store_name()},
        "scope": {"kind": "user", "value": user_id},
        "metadata": {"source": "agent-langgraph-advanced"},
    }
    if initial_items:
        body["items"] = initial_items
    response = client.api_client.do("POST", _CONVERSATIONS_PATH, body=body)
    conversation_id = str(response["id"])

    # Create accepts at most 20 initial items. Preserve legacy clients that send a
    # longer transcript by appending the remainder in API-sized chunks.
    remaining = request_items[_MAX_ITEMS_PER_WRITE:]
    if remaining:
        _append_items(client, conversation_id, remaining)

    return ManagedConversationTurn(
        conversation_id=conversation_id,
        items=request_items,
        created=True,
    )


def append_conversation_output(
    conversation_id: str, output_items: list[dict[str, Any]]
) -> None:
    if not conversations_enabled() or not output_items:
        return
    client, _ = _caller()
    _append_items(client, conversation_id, output_items)
