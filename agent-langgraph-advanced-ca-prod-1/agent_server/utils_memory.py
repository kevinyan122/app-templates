import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks_langchain import AsyncCheckpointSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from mlflow.genai.agent_server import get_request_headers

from agent_server.utils import _is_databricks_app_env, get_user_workspace_client

logger = logging.getLogger(__name__)

# Long-lived Lakebase checkpointer for SHORT-TERM (session) memory, opened once at app
# startup in start_server.py's lifespan and reused across all requests. LONG-TERM memory
# is the UC managed memory store (see the memory tools below) — a governed REST securable
# with no Lakebase resource of its own.
_lakebase_checkpointer: Optional[AsyncCheckpointSaver] = None


def set_lakebase_resources(checkpointer: AsyncCheckpointSaver) -> None:
    global _lakebase_checkpointer
    _lakebase_checkpointer = checkpointer


@dataclass(frozen=True)
class LakebaseConfig:
    instance_name: Optional[str]
    autoscaling_endpoint: Optional[str]
    autoscaling_project: Optional[str]
    autoscaling_branch: Optional[str]
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or self.instance_name or f"{self.autoscaling_project}/{self.autoscaling_branch}"


def init_lakebase_config() -> Optional[LakebaseConfig]:
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    raw_name = os.getenv("LAKEBASE_INSTANCE_NAME") or None
    project = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None

    has_autoscaling = project and branch
    if not endpoint and not raw_name and not has_autoscaling:
        # Lakebase is OPTIONAL — it backs SHORT-TERM (session) memory only. Long-term managed
        # memory (the UC memory store) needs no Lakebase. With nothing configured, the agent runs
        # without a checkpointer (no cross-request session persistence; background mode disabled).
        logger.info(
            "No Lakebase configured — short-term (session) memory and background mode are disabled; "
            "long-term managed memory is unaffected."
        )
        return None

    # Priority: endpoint > project+branch > instance_name (mutually exclusive in the library)
    if endpoint:
        instance_name = None
        project = None
        branch = None
    elif has_autoscaling:
        instance_name = None
        endpoint = None
    else:
        instance_name = resolve_lakebase_instance_name(raw_name)
        endpoint = None
        project = None
        branch = None

    memory_schema = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA") or None
    return LakebaseConfig(
        instance_name=instance_name,
        autoscaling_endpoint=endpoint,
        autoscaling_project=project,
        autoscaling_branch=branch,
        memory_schema=memory_schema,
    )


def _is_lakebase_hostname(value: str) -> bool:
    """Check if the value looks like a Lakebase hostname rather than an instance name."""
    # Hostname pattern: instance-{uuid}.database.{env}.cloud.databricks.com
    return ".database." in value and value.endswith(".com")


def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """
    Resolve a Lakebase instance name from a hostname if needed.

    If the input is a hostname (e.g., from Databricks Apps value_from resolution),
    this will resolve it to the actual instance name by listing database instances.

    Args:
        instance_name: Either an instance name or a hostname
        workspace_client: Optional WorkspaceClient to use for resolution

    Returns:
        The resolved instance name

    Raises:
        ValueError: If the hostname cannot be resolved to an instance name
    """
    if not _is_lakebase_hostname(instance_name):
        # Input is already an instance name
        return instance_name

    # Input is a hostname - resolve to instance name
    client = workspace_client or WorkspaceClient()
    hostname = instance_name

    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    # Find the instance that matches this hostname
    for instance in instances:
        rw_dns = getattr(instance, "read_write_dns", None)
        ro_dns = getattr(instance, "read_only_dns", None)

        if hostname in (rw_dns, ro_dns):
            resolved_name = getattr(instance, "name", None)
            if not resolved_name:
                raise ValueError(
                    f"Found matching instance for hostname '{hostname}' "
                    "but instance name is not available."
                )
            logging.info(f"Resolved Lakebase hostname '{hostname}' to instance name '{resolved_name}'")
            return resolved_name

    raise ValueError(
        f"Unable to find database instance matching hostname '{hostname}'. "
        "Ensure the hostname is correct and the instance exists."
    )


def get_lakebase_access_error_message(lakebase_instance_name: str) -> str:
    """Generate a helpful error message for Lakebase access issues."""
    if _is_databricks_app_env():
        app_name = os.getenv("DATABRICKS_APP_NAME")
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            f"The App Service Principal for '{app_name}' may not have access.\n\n"
            "To fix this:\n"
            "1. Go to the Databricks UI and navigate to your app\n"
            "2. Click 'Edit' → 'App resources' → 'Add resource'\n"
            "3. Add your Lakebase instance as a resource\n"
            "4. Grant the necessary permissions on your Lakebase instance. "
            "See the README section 'Grant Lakebase permissions to your App's Service Principal' for the SQL commands."
        )
    else:
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            "Please verify:\n"
            "1. The instance name is correct\n"
            "2. You have the necessary permissions to access the instance\n"
            "3. Your Databricks authentication is configured correctly"
        )


@asynccontextmanager
async def lakebase_context(config: LakebaseConfig):
    """Yield the checkpointer for short-term (session) memory."""
    async with AsyncCheckpointSaver(
        instance_name=config.instance_name,
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        schema=config.memory_schema,
    ) as checkpointer:
        yield checkpointer


@asynccontextmanager
async def acquire_lakebase_resources(config: LakebaseConfig):
    """Yield the short-term checkpointer for use in a request handler.

    If start_server.py's lifespan populated the long-lived checkpointer, yield it without
    closing on exit. Otherwise (e.g. evaluate_agent.py running outside the FastAPI server)
    fall back to opening a fresh per-call lakebase_context. When no Lakebase is configured
    (config is None), yield None — the agent runs without a short-term checkpointer.
    """
    if _lakebase_checkpointer is not None:
        yield _lakebase_checkpointer
    elif config is not None:
        async with lakebase_context(config) as checkpointer:
            yield checkpointer
    else:
        yield None


# ---------------------------------------------------------------------------
# Long-term memory: Databricks MANAGED memory (UC memory-store REST API).
# Six tools (search/save/get/list/update/delete) are thin REST calls. No Lakebase, no
# customer-managed embedding endpoint, no extra dependency — just the databricks-sdk already in the template.
# ---------------------------------------------------------------------------

# API: BASE = /api/2.1/unity-catalog/memory-stores/{DATABRICKS_MEMORY_STORE}
#   create  POST {BASE}/entries?scope=…   {path,contents,description,creation_reason,creation_source}  (flat body; scope is a query param)
#   search  POST {BASE}/entries:search    ?scope  {query,top_k} -> {results:[{memory_entry:{path,description,contents,…}, score}]}  (semantic retrieval with BM25 keyword boosting)
#   get     GET  {BASE}/entries:get       ?scope,path        -> {contents, description, ...}
#   list    GET  {BASE}/entries           ?scope             -> {entries:[{path,description,has_contents}]}  (key omitted entirely when empty)
#   update  PATCH{BASE}/entries          {scope, path, [description], [one contents edit op]}  (>=1 of the two)
#   delete  DELETE {BASE}/entries         ?scope,path

_client: Optional[WorkspaceClient] = None

_EVAL_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _eval_scope_enabled() -> bool:
    return os.getenv("DATABRICKS_MEMORY_EVAL_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _direct_eval_scope(request=None) -> Optional[str]:
    """Return the exact caller-supplied scope on a test-only eval deployment.

    This deliberately bypasses per-user OBO namespacing. Keep evaluation mode disabled
    on customer-facing deployments: the App service principal can access every scope in
    its configured memory store.
    """

    if not _eval_scope_enabled():
        return None
    scope = dict(getattr(request, "custom_inputs", None) or {}).get("eval_scope")
    if scope is None:
        return None
    scope = str(scope)
    if not _EVAL_SCOPE_PATTERN.fullmatch(scope):
        raise ValueError(
            "custom_inputs.eval_scope must be 1-128 characters using only letters, "
            "numbers, '.', '_', ':', or '-'."
        )
    return scope


def is_eval_read_only(request=None) -> bool:
    """Allow a frozen-corpus App evaluation to expose retrieval tools only.

    The flag is honored only alongside a valid direct eval scope on a deployment
    with ``DATABRICKS_MEMORY_EVAL_MODE`` enabled. It therefore cannot change the
    tool set for normal OBO-scoped production requests.
    """

    if _direct_eval_scope(request) is None:
        return False
    value = dict(getattr(request, "custom_inputs", None) or {}).get(
        "eval_read_only", False
    )
    if not isinstance(value, bool):
        raise ValueError("custom_inputs.eval_read_only must be a boolean.")
    return value


def _headers() -> Optional[dict]:
    """Extra headers on every memory-store call. DATABRICKS_MEMORY_TRAFFIC_ID routes requests to a
    test environment (e.g. a liteswap) — leave it unset outside endpoint testing."""
    traffic_id = os.getenv("DATABRICKS_MEMORY_TRAFFIC_ID")
    return {"x-databricks-traffic-id": traffic_id} if traffic_id else None


def _ws() -> WorkspaceClient:
    """The memory caller — the app SP when deployed, the developer when local.

    The SP can see every scope. Normal requests therefore derive scope from OBO identity;
    the explicit eval-mode exception is safe only with a dedicated test store.
    """
    global _client
    if _client is None:
        _client = WorkspaceClient()
    return _client


def _entries(suffix: str = "") -> str:
    store = os.getenv("DATABRICKS_MEMORY_STORE")
    if not store:
        raise RuntimeError("DATABRICKS_MEMORY_STORE is not set — it must be the full catalog.schema.name.")
    return f"/api/2.1/unity-catalog/memory-stores/{store}/entries{suffix}"


def resolve_scope(request=None) -> Optional[str]:
    """Resolve the long-term-memory scope and fail closed when it is unavailable.

    Test-only exception: with DATABRICKS_MEMORY_EVAL_MODE enabled, custom_inputs.eval_scope
    is the complete scope. Otherwise deployed requests use only the verified OBO user id;
    client-supplied user identities remain local-development fallbacks.
    """
    eval_scope = _direct_eval_scope(request)
    if eval_scope is not None:
        return eval_scope
    headers = get_request_headers() or {}
    if headers.get("x-forwarded-access-token"):
        return str(get_user_workspace_client().current_user.me().id)
    # Deployed -> the verified OBO token above is the only trusted source. DATABRICKS_APP_NAME is set by
    # the Apps runtime when deployed, so the client-supplied fallbacks below are LOCAL-DEV ONLY:
    if os.getenv("DATABRICKS_APP_NAME"):
        return None
    ci = dict(getattr(request, "custom_inputs", None) or {})
    return headers.get("x-forwarded-user") or ci.get("user_id") or os.getenv("DATABRICKS_MEMORY_SCOPE")


# The six operations. `scope` is passed in (never model-supplied). Each returns a short string.
def _save(scope, path, description, contents=""):
    try:
        _ws().api_client.do("POST", _entries(), query={"scope": scope}, headers=_headers(), body={
            "path": path, "contents": contents, "description": description,
            "creation_reason": "CREATION_REASON_AGENT_INFERRED",
            "creation_source": "CREATION_SOURCE_ONLINE_AGENT"})
    except DatabricksError as e:
        if e.error_code == "ALREADY_EXISTS":
            return f"A memory already exists at {path}; use update_memory to revise it."
        return f"Could not save {path}: {getattr(e, 'message', str(e))}"
    return f"Saved memory at {path}."


def _get(scope, path):
    try:
        entry = _ws().api_client.do("GET", _entries(":get"), query={"scope": scope, "path": path}, headers=_headers())
    except DatabricksError as e:
        if e.error_code == "NOT_FOUND":
            return f"No memory at {path}."
        return f"Could not read {path}: {getattr(e, 'message', str(e))}"
    # A brief memory may have empty contents — its description is then the memory.
    return entry.get("contents") or entry.get("description") or f"(empty memory at {path})"


def _search(scope, query, top_k=10):
    try:
        resp = _ws().api_client.do("POST", _entries(":search"), query={"scope": scope}, headers=_headers(),
                                   body={"query": query, "top_k": top_k})
    except DatabricksError as e:
        return f"Could not search memories: {getattr(e, 'message', str(e))}"
    results = resp.get("results", [])
    if not results:
        return f"No memories matched '{query}'."
    lines = []
    for r in results:
        entry = r.get("memory_entry", {})
        description = " ".join(str(entry.get("description") or "").split())
        # Search currently returns contents but may not return has_contents. Derive the flag before
        # discarding the body so the model knows when get_memory is useful.
        has_contents = bool(entry.get("has_contents") or entry.get("contents"))
        lines.append(
            f"- path: {str(entry.get('path') or '')}\n"
            f"  description: {description}\n"
            f"  has_contents: {str(has_contents).lower()}"
        )
    # Preserve backend rank through list order, but expose neither scores nor contents. Search is
    # the index/selection step; get_memory is the full-body read.
    match_label = "memory match" if len(lines) == 1 else "memory matches"
    return f"{len(lines)} {match_label} (ranked index):\n" + "\n\n".join(lines)


_LIST_PAGE_SIZE = 200


def _list(scope, page_token=None):
    query = {"scope": scope, "page_size": _LIST_PAGE_SIZE}
    if page_token:
        query["page_token"] = page_token
    try:
        resp = _ws().api_client.do("GET", _entries(), query=query, headers=_headers())
    except DatabricksError as e:
        return f"Could not list memories: {getattr(e, 'message', str(e))}"
    items = resp.get("entries", [])
    if not items:
        return "No more memories." if page_token else "No memories yet."
    # Count header (the model is unreliable at tallying a long list); `[has_contents]` marks entries
    # whose body must be read with get_memory — unmarked entries are captured by their description.
    lines = [
        ("[has_contents] " if e.get("has_contents") else "") + f"- {e['path']}: {e.get('description', '')}"
        for e in items
    ]
    next_token = resp.get("next_page_token")
    header = f"{len(items)} memories" + (" (continued)" if page_token else "")
    if next_token and not page_token:
        header = "first " + header
    out = f"{header}:\n" + "\n".join(lines)
    if next_token:
        out += (
            f"\nMore memories exist — call list_memories again with "
            f"page_token='{next_token}' if you need the rest."
        )
    return out


def _update(scope, path, op=None, description=None):  # op = at most one of str_replace/insert/replace_all
    op = op or {}
    if len(op) > 1:
        return "Pass at most one contents edit (str_replace / insert / replace_all)."
    if not op and description is None:
        return "Provide a new description and/or one contents edit (str_replace / insert / replace_all)."
    body = {"scope": scope, "path": path, **op}
    if description is not None:
        body["description"] = description
    try:
        _ws().api_client.do("PATCH", _entries(), body=body, headers=_headers())
    except DatabricksError as e:
        if e.error_code == "NOT_FOUND":
            return f"No memory at {path} to update — check list_memories or save it first."
        # e.g. str_replace.old_str matched 0 or >1 times -> return it so the model re-reads and retries.
        return f"Could not update {path}: {getattr(e, 'message', str(e))}"
    return f"Updated {path}."


def _delete(scope, path):
    try:
        _ws().api_client.do("DELETE", _entries(), query={"scope": scope, "path": path}, headers=_headers())
    except DatabricksError as e:
        if e.error_code == "NOT_FOUND":
            return f"No memory at {path} (already gone)."
        return f"Could not delete {path}: {getattr(e, 'message', str(e))}"
    return f"Deleted {path}."


def _scope(config: RunnableConfig) -> str:
    s = (config.get("configurable") or {}).get("memory_scope")
    if not s:
        raise RuntimeError("No end-user scope in config — refusing a shared memory bucket.")
    return s


def memory_tools(*, read_only: bool = False):
    @tool
    async def search_memory(query: str, config: RunnableConfig, top_k: int = 10) -> str:
        """Search the user's stored memories (facts, preferences, projects, domain knowledge,
        workflows) and return a ranked index of the most relevant entries.

        This is the mandatory first retrieval step on EVERY user turn. Call it at least once before
        answering, even when the request seems general, impersonal, fully answerable from the current
        conversation, or unlikely to have relevant memory. The first query must be based on the user's
        latest request. Rewrite it as the smallest distinctive topic phrase likely to identify the
        relevant memory. Preserve names, identifiers, product names, dates, and error codes exactly
        once. Omit the requested answer attribute (such as name, date, owner, or status) when the
        topic alone is sufficient. Add at most one grounded facet only when genuinely needed to
        disambiguate the topic; never append generic expansion lists such as "projects, demos, code,
        or presentation identifiers." Include only enough prior context to make references
        self-contained, aim for roughly 3-12 words, and omit conversational filler and the action the
        user wants performed. Search again only for a
        meaningfully narrower, broader, or differently focused information need; never repeat the
        same or an equivalent query.

        Parameters:
        - query (required): Use one concise, self-contained natural-language sentence or phrase
          describing the information needed. Semantic retrieval has the most impact, and handles
          paraphrases, while relevant exact terms receive additional keyword weight. Include
          semantically ambiguous keywords like names, identifiers, product names, dates, and error
          codes once when relevant. Do not repeat terms or keyword-stuff the query.
        - top_k (optional, default 10, max 50): how many results to return.

        Returns a readable ranked list of up to top_k entries. Each entry contains ONLY path,
        description, and has_contents; list order reflects relevance, but scores and contents are
        not returned. If a relevant entry has has_contents=true and is worth looking into, call
        get_memory on its path before relying on its details. If has_contents=false, its description
        is the complete memory.
        A no-match result means no relevant memories were returned for this query, not that the user
        has no stored memories.
        Don't re-search a topic you've already seen this turn."""
        return _search(_scope(config), query, top_k)

    @tool
    async def save_memory(path: str, config: RunnableConfig, description: str, contents: str = "") -> str:
        """Create ONE durable memory — a stable preference, fact, decision, or ongoing project; not one-off
        chatter, secrets, or anything the user scoped to this conversation ("for this chat only" = never
        save). Create-only (an existing path errors), so search_memory the topic first and use
        update_memory to revise a topic. path: a SHORT, STABLE topic bucket (lowercase-hyphenated, starts
        /memories/, ends .md) — keep it broad and reusable (e.g. /memories/preferences/food.md); put the
        specifics in description/contents, NOT the path, so related facts share one path and you update it
        instead of minting near-duplicates (avoid over-specific paths like /memories/preferences/coffee-oat-milk.md).
        description: ONE short, specific line summarizing what's inside (e.g. "Kitchen reno: ~30k CAD,
        galley layout, done end of summer") — not a vague category like "Home projects". A single brief
        fact can be the whole description, with contents empty.
        contents: the memory itself — required once there's a second fact, date, number, or any structure
        (bullets welcome); never echo the description."""
        return _save(_scope(config), path, description, contents)

    @tool
    async def get_memory(path: str, config: RunnableConfig) -> str:
        """Read the FULL contents of ONE memory by its exact path. search_memory and list_memories return
        only path, description, and has_contents. Call get_memory for a relevant result with
        has_contents=true when it is worth looking into, before relying on its details. An entry with
        has_contents=false is fully captured by its description. Also re-read before a contents edit
        (search results can lag recent writes), or when create_memory hits an existing path.
        Not found means it isn't stored, not that the fact is false."""
        return _get(_scope(config), path)

    @tool
    async def list_memories(config: RunnableConfig, page_token: Optional[str] = None) -> str:
        """List EVERY saved memory as (path, description) — the full index; returns NO contents.
        Reserve this for when the complete inventory is the point
        (e.g. the user asks "what do you remember about me?") or a search found nothing.
        An entry prefixed `[has_contents]` has a fuller body — get_memory(path) to read it before stating
        specifics; an entry without that prefix is fully captured by its description. If the result notes
        more memories exist, call again with the given page_token only if you need the rest. Omit
        page_token to start from the beginning."""
        return _list(_scope(config), page_token)

    @tool
    async def update_memory(path: str, config: RunnableConfig, description: Optional[str] = None,
                            str_replace: Optional[dict] = None, insert: Optional[dict] = None,
                            replace_all: Optional[dict] = None) -> str:
        """Revise an EXISTING memory in place (same path; the path can't change). Pass description="..." to
        replace its one-line description (use this to correct a brief, description-only memory), and/or EXACTLY
        ONE contents edit op — str_replace={"old_str": ..., "new_str": ...} (old_str must occur once) ·
        insert={"insert_text": ..., "insert_line": <optional>} · replace_all={"contents": ...}. get_memory first
        so a contents edit matches; at least one of description / an edit op is required. New facts go in
        contents, not a longer description — a description outgrowing one line means details belong in contents.
        After a contents edit, refresh a stale or overlong description (it must stay a current one-line
        summary); if the entry already says it, skip the update entirely and just confirm to the user."""
        op = {k: v for k, v in (("str_replace", str_replace), ("insert", insert), ("replace_all", replace_all)) if v}
        return _update(_scope(config), path, op, description)

    @tool
    async def delete_memory(path: str, config: RunnableConfig) -> str:
        """Permanently remove ONE memory by its exact path. Use for stale/wrong/superseded/duplicate entries
        or when the user asks to forget something. Don't delete to rewrite a valid fact — use update_memory."""
        return _delete(_scope(config), path)

    read_tools = [search_memory, get_memory, list_memories]
    if read_only:
        return read_tools
    return [search_memory, save_memory, get_memory, list_memories, update_memory, delete_memory]
