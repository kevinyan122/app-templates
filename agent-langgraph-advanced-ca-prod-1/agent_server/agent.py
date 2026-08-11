import asyncio
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.prompts import SYSTEM_PROMPT
from agent_server.utils_conversations import (
    CONVERSATION_CUSTOM_OUTPUT_KEY,
    CONVERSATION_EVENT_TYPE,
    append_conversation_output,
    prepare_conversation_turn,
)
from agent_server.utils import (
    _get_or_create_thread_id,
    get_user_workspace_client,
    init_mcp_client,
    process_agent_astream_events,
)
from agent_server.utils_memory import (
    acquire_lakebase_resources,
    get_lakebase_access_error_message,
    init_lakebase_config,
    is_eval_read_only,
    memory_tools,
    resolve_scope,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
sp_workspace_client = WorkspaceClient()

LLM_ENDPOINT_NAME = "databricks-gpt-5-2"
LAKEBASE_CONFIG = init_lakebase_config()


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


class StatefulAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


async def init_agent(
    workspace_client: Optional[WorkspaceClient] = None,
    checkpointer: Optional[Any] = None,
    memory_read_only: bool = False,
):
    tools = [get_current_time] + memory_tools(read_only=memory_read_only)
    # To use MCP server tools instead, uncomment the below lines:
    # mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    # try:
    #     tools.extend(await mcp_client.get_tools())
    # except Exception:
    #     logger.warning("Failed to fetch MCP tools. Continuing without MCP tools.", exc_info=True)

    model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        state_schema=StatefulAgentState,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = []
    custom_outputs: dict[str, Any] = {}
    async for event in stream_handler(request):
        if event.type == "response.output_item.done":
            outputs.append(event.item)
        elif event.type == CONVERSATION_EVENT_TYPE and event.custom_outputs:
            custom_outputs.update(event.custom_outputs)
    return ResponsesAgentResponse(
        output=outputs,
        custom_outputs=custom_outputs or None,
    )


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    # Resolve the end-user scope for long-term memory and fail closed — never an empty
    # scope or the app's own identity (the SP can read every scope).
    scope = resolve_scope(request)
    if not scope:
        raise HTTPException(
            status_code=401,
            detail="No end-user identity — refusing a shared memory scope.",
        )
    memory_read_only = is_eval_read_only(request)

    # Conversations are deterministic handler-owned session state. The model sees
    # the stored history automatically; this is intentionally not an agent tool.
    conversation_turn = await asyncio.to_thread(prepare_conversation_turn, request)
    thread_id = (
        conversation_turn.conversation_id
        if conversation_turn
        else _get_or_create_thread_id(request)
    )
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "memory_scope": scope}
    }

    conversation_items = (
        conversation_turn.items
        if conversation_turn
        else [i.model_dump() for i in request.input]
    )
    if conversation_turn:
        yield ResponsesAgentStreamEvent(
            type=CONVERSATION_EVENT_TYPE,
            custom_outputs={
                CONVERSATION_CUSTOM_OUTPUT_KEY: conversation_turn.custom_output()
            },
        )

    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input(conversation_items),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    try:
        turn_output_items: list[dict[str, Any]] = []
        # A managed conversation and a LangGraph checkpointer are alternative
        # owners of the same session history. Never stack both for one request.
        checkpointer_context = (
            nullcontext(None)
            if conversation_turn
            else acquire_lakebase_resources(LAKEBASE_CONFIG)
        )
        async with checkpointer_context as checkpointer:
            # By default, uses service principal credentials.
            # For on-behalf-of user authentication, pass get_user_workspace_client() to init_agent.
            agent = await init_agent(
                checkpointer=checkpointer,
                memory_read_only=memory_read_only,
            )

            async for event in process_agent_astream_events(
                agent.astream(input_state, config, stream_mode=["updates", "messages"])
            ):
                if event.type == "response.output_item.done":
                    turn_output_items.append(event.item)
                yield event

        if conversation_turn and turn_output_items:
            await asyncio.to_thread(
                append_conversation_output,
                conversation_turn.conversation_id,
                turn_output_items,
            )
    except Exception as e:
        error_msg = str(e).lower()
        # Check for Lakebase access/connection errors (only when Lakebase is configured)
        if LAKEBASE_CONFIG and any(
            keyword in error_msg
            for keyword in ["lakebase", "pg_hba", "postgres", "database instance"]
        ):
            logger.error("Lakebase access error: %s", e)
            raise HTTPException(
                status_code=503,
                detail=get_lakebase_access_error_message(LAKEBASE_CONFIG.description),
            ) from e
        raise
