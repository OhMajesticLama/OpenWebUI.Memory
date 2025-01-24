"""
title: Memory
author: ohmajesticlama
author_url: https://github.com/OhMajesticLama/OpenWebUI.Memory
funding_url: https://github.com/open-webui
version: 0.3
license: MIT

# Purpose
A function to automatically manage memories. Compatible with OpenWebUI 0.5+.

/!\ This function adds and modifies memories. This may lead to memory data loss.

# Configuration
This function retrieves memories directly, it is recommended to disable the Settings->Personalization->Memory toggle
as this will create duplicate entries in the chat history.

If you use Ollama, it should retrieve the configuration from OpenWebUI admin settings automatically.

## Recommended Functions
Time Awareness (https://openwebui.com/f/skroute/time_awareness/): providing the model with time information may help it make better use of memories.

# Current limitations

The data flow works, memories are registered and retrieved, but the memories generated contain hallucinations with the default model.
"""

import sys
import datetime
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, List, Dict, Tuple
import logging
import functools
import inspect
import json
import asyncio

import aiohttp
from fastapi.requests import Request

import open_webui
import open_webui.main
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    query_memory,
    QueryMemoryForm,
    delete_memory_by_id,
)
from open_webui.models.users import Users, User
from open_webui.env import GLOBAL_LOG_LEVEL


# from open_webui.main import webui_app
LOGGER: logging.Logger = logging.getLogger("FUNC:MEMORY")


def set_logs(logger: logging.Logger, level: int, force: bool = False):
    """
    logger:
        Logger that will be configured and connected to handlers.

    level:
        Log level per logging module.

    force:
        If set to True, will create and attach a StreamHandler to logger, even if there is already one attached.
    """
    logger.setLevel(level)

    logger.debug("%s has %s handlers", logger, len(logger.handlers))
    for handler in logger.handlers:
        if not force and isinstance(handler, logging.StreamHandler):
            # There is already a stream handler attached to this logger, chances are we don"t want to add another one.
            # This might be a reimport.
            # However we still enforce log level as that's likely what the user expects.
            handler.setLevel(level)
            logger.info("logger already has a StreamHandler. Not creating a new one.")
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(levelname)s[%(name)s]%(lineno)s:%(asctime)s: %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


set_logs(LOGGER, GLOBAL_LOG_LEVEL)
# /!\ Do not leave DEBUG mode on: conversation content will leak in logs.
# set_logs(LOGGER, logging.DEBUG)


def log_exceptions(func: Callable[..., Any]):
    """
    Log exception in decorated function. Use LOGGER of this module.

    Usage:

        # Example 1
        @log_exceptions
        def foo():
            ...
            raise Exception()

    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):  # type: ignore  # false positive.
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                LOGGER.error("Error in %s: %s", func, exc, exc_info=True)
                raise exc

    else:

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                LOGGER.error("Error in %s: %s", func, exc, exc_info=True)
                raise exc

    return _wrapper


class ROLE:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Filter:
    class Valves(BaseModel):
        model: str = Field(
            default="mistral:7b-instruct",
            description="Model to use to process memories.",
        )

        n_memories: int = Field(
            default=5, description="Consider top N relevant memories."
        )

        memories_dist_max: float = Field(
            default=0.7,
            description="Ignore memories with a distance higher than this value.",
        )

        chat_api_host: str = Field(
            # Get defaults from OpenWebUI config instead of asking to user to re-enter them.
            # Currently works only with Ollama config.
            default_factory=lambda: get_api_chat_completion_config()[0],
            description="Defaults to Ollama configuration in system settings if enabled.",
        )

        api_key: str = Field(
            # Get defaults from OpenWebUI config instead of asking to user to re-enter them.
            # Currently works only with Ollama config.
            default_factory=lambda: get_api_chat_completion_config()[1],
            description="Key for chat_api.",
        )

        priority: int = Field(
            default=5,
            description=(
                "Higher priority means this will be executed after lower priority functions."
                "This may benefit from basic context provided by other functions, put higher priority."
            ),
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function."
        )

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        self.uservalves = self.UserValves()
        self._session = aiohttp.ClientSession()

    @log_exceptions
    def __del__(self):
        if hasattr(self, "_session"):
            asyncio.run(self._session.close())

    async def _build_memory_query(self, messages: List[Dict[str, str]]) -> str:
        memory_keywords = await self.single_query_model(
            self.valves.model,
            PROMPT.MEMORY_QUERY,
            # Let's try a markdown format that may be easier to read for the model.
            "".join(f"\n### {m['role']}\n{m['content']}\n" for m in messages),
            # str([{"role": m["role"], "content": m["content"]} for m in messages]),
        )
        return memory_keywords

    async def _query_memories(
        self,
        query: str,
        *,
        user: User,
        source: str = ROLE.USER,
        source_default: str = ROLE.USER,
        distance_max: Optional[float] = None,
        n_memories: Optional[int] = None,
    ):
        assert source_default in (ROLE.USER, ROLE.ASSISTANT)
        n_memories = self.valves.n_memories if n_memories is None else n_memories
        memories_raw = await query_memory(
            request=Request(scope={"type": "http", "app": open_webui.main.app}),
            form_data=QueryMemoryForm(
                content=query,
                k=str(
                    10 * n_memories
                ),  # Let's query more as we'll have to filter out other sources.
            ),
            user=user,
        )
        dist_max = (
            self.valves.memories_dist_max if distance_max is None else distance_max
        )

        # memories example:
        #   ids=[['b8e68597-7e9d-4553-b5d9-7bbc503370fa', '9ba8d5df-1350-4c01-b2f9-b1b7cf369ba9']]
        #   documents=[['User likes sci-fi TV shows and sometimes watches Disney+, also likes potatoes.',
        #               'User displays interest in multiple language studies, focusing on French and Japanese.']]
        #   metadatas=[[{'created_at': 1737153170}, {'created_at': 1735153131}]]
        #   distances=[[0.8044559591349751, 0.8268496990203857]]
        #
        # Reshapeit for easier processing
        memories = []
        for memory, meta, dist, mid in zip(
            memories_raw.documents[0],
            memories_raw.metadatas[0],
            memories_raw.distances[0],
            memories_raw.ids[0],
        ):
            try:
                memory_json = json.loads(memory)
            except json.JSONDecodeError as exc:
                # Not JSON, memory was not added by this function.
                # Consider it legacy with source_default
                memory_json = {"content": memory, "source": source_default}

            if memory_json.get("source") is None:
                # Memory was added by another tool (or there is a bug).
                memory_json["source"] = source_default

            if dist <= dist_max and memory_json.get("source") == source:
                # Filter out memories not from requested source.
                memories.append(
                    {
                        "content": memory_json.get("content") or "",
                        "metadata": {
                            "created_at": datetime.datetime.fromtimestamp(
                                meta["created_at"]
                            ).isoformat()
                        },
                        "source": source,
                        "dist": dist,
                        "id": mid,
                    }
                )
        LOGGER.debug("_query_memories: memories unsorted: %s", memories)
        # memories.sort(key=lambda m: m["dist"], reverse=True)
        memories.sort(key=lambda m: m["dist"])

        return memories[:n_memories]

    async def single_query_model(
        self,
        model: str,
        system: str,
        query: str,
    ) -> str:
        target_url = f"{self.valves.chat_api_host.strip('/')}/v1/chat/completions"
        # Keep here it may help to debug.
        # LOGGER.debug("single_query_model.config:")
        # for k, v in open_webui.main.app.state.config.__dict__.items():
        #     LOGGER.debug("  %s: %s", k, v)
        # To use OpenWebUI endpoint we need and API KEY.
        return await single_query_model(
            self._session, target_url, model, system, query, api_key=self.valves.api_key
        )

    @staticmethod
    def _format_context(
        memories_user: List[Dict[str, str]], memories_assistant: List[Dict[str, str]]
    ) -> str:
        return '<context source="function_memory">\n  {}\n</context>'.format(
            "\n  ".join(
                (
                    '<memories source="{}">\n  {}  </memories>\n'.format(
                        ROLE.ASSISTANT,
                        "\n    ".join(
                            (
                                f"""<memory created_at="{m['metadata']['created_at']}">{m['content']}</memory>"""
                                for m in memories_assistant
                            )
                        ),
                    ),
                    '<memories source="{}"><--! In case of conflicting information, trust the user memory. -->\n  {}</memories>\n'.format(
                        ROLE.USER,
                        "\n    ".join(
                            (
                                f"""<memory created_at="{m['metadata']['created_at']}">{m['content']}</memory>"""
                                for m in memories_user
                            )
                        ),
                    ),
                )
            )
        )

    @log_exceptions
    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        if __user__ is None:
            LOGGER.info("No __user__. Do nothing.")
            return body
        if not __user__["valves"].enabled:
            # user doesn't want this, do nothing.
            LOGGER.debug("UserValve.enabled = False. Do nothing.")
            return body
        # Modify the request body or validate it before processing by the chat completion API.
        # This function is the pre-processor for the API where various checks on the input can be performed.
        # It can also modify the request before sending it to the API.
        LOGGER.debug(f"inlet:{__name__}")
        LOGGER.debug(f"inlet:body:{body}")
        LOGGER.debug(f"inlet:user:{__user__}")

        if "id" not in __user__:
            LOGGER.info("No 'id' key in __user__. Do nothing.")
            return body
        user = Users.get_user_by_id(__user__["id"])

        messages: Optional[Dict[str, str]] = body.get("messages")
        if not messages:
            # nothing to do here.
            return body

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Searching for memories...",
                    "done": False,
                },
            }
        )
        memory_query = await self._build_memory_query(messages)
        LOGGER.debug("memory_query: %s", memory_query)
        memories_user, memories_assistant = await asyncio.gather(
            self._query_memories(memory_query, user=user, source=ROLE.USER),
            self._query_memories(memory_query, user=user, source=ROLE.ASSISTANT),
        )
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Memories search completed.",
                    "done": True,
                },
            }
        )

        # Get memories and inject them as context.

        LOGGER.debug("memories_assistant: %s", memories_assistant)
        LOGGER.debug("memories_user: %s", memories_user)

        context = self._format_context(memories_user, memories_assistant)
        LOGGER.debug("Added context message: %s", context)

        _, ind = get_last_message(messages, ROLE.USER)
        if ind is not None:
            context_message = {"role": ROLE.SYSTEM, "content": context}
            messages.insert(ind, context_message)
        else:
            LOGGER.debug(
                "No message from user found. Do nothing: %s", messages[-1].get("role")
            )

        return body

    @log_exceptions
    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> Dict[str, str]:
        if __user__ is None:
            LOGGER.info("No __user__. Do nothing.")
            return body
        if not __user__["valves"].enabled:
            # user doesn't want this, do nothing.
            LOGGER.debug("UserValve.enabled = False. Do nothing.")
            return body
        # Modify or analyze the response body after processing by the API.
        # This function is the post-processor for the API, which can be used to modify the response
        # or perform additional checks and analytics.
        LOGGER.debug(f"outlet:{__name__}")
        LOGGER.debug(f"outlet:body:{body}")
        LOGGER.debug(f"outlet:user:{__user__}")

        if not "id" in __user__:
            LOGGER.warn("No 'id' key in __user__. Do nothing.")
            return body
        user = Users.get_user_by_id(__user__["id"])

        if not body:
            # nothing to do here.
            LOGGER.debug("No body. Do nothing.")
            return body
        # Is there something to remember?
        messages: Optional[Dict[str, str]] = body.get("messages")
        if not messages:
            # nothing to do here.
            LOGGER.debug("No 'messages' key in body. Do nothing.")
            return body

        # Here we only consider memories from the user. We may want to have a sort of "assistant memory" that we can tag separately in the future.
        user_message, _ = get_last_message(messages, ROLE.USER)
        if user_message is None:
            LOGGER.info("No user message found in messages. Do nothing.")
            return body
        task_user = asyncio.create_task(
            self._process_message_for_memories(
                user_message.get("content", ""),
                source=ROLE.USER,
                source_default=ROLE.USER,
                user=user,
                __event_emitter__=__event_emitter__,
            )
        )

        assistant_message, _ = get_last_message(messages, ROLE.ASSISTANT)
        if assistant_message is None:
            LOGGER.info("No assistant message found in messages. Do nothing.")
            return body
        task_assistant = asyncio.create_task(
            self._process_message_for_memories(
                assistant_message.get("content", ""),
                source=ROLE.ASSISTANT,
                source_default=ROLE.USER,
                user=user,
                __event_emitter__=__event_emitter__,
            )
        )
        await asyncio.gather(task_user, task_assistant)

        return body

    async def _assess_for_memories(
        self, content: str, *, source: str = ROLE.USER
    ) -> List[str]:
        """
        Identify if there are elements to remember in content.
        """
        LOGGER.debug("_assess_for_memories: content: %s", content)

        assessment = await self.single_query_model(
            self.valves.model,
            PROMPT.MEMORIES_ASSESS.format(source=source),
            content,
        )
        LOGGER.debug("_assess_for_memories: assessment: %s", assessment)

        try:
            new_memories = json.loads(assessment.strip())
        except json.JSONDecodeError as exc:
            LOGGER.debug(
                "Assessment for memory did not return valid JSON: %s. Exception: %s",
                assessment,
                exc,
            )
            return []
        return new_memories

    async def _merge_memories(
        self, new_memories: List[str], current_memories: List[str]
    ) -> List[str]:
        """
        Merge new memories with current ones.
        """
        output = await self.single_query_model(
            self.valves.model,
            PROMPT.MEMORIES_MERGE,
            f"current_memories = {json.dumps(current_memories)}"
            f"\nnew_memories = {json.dumps(new_memories)}",
        )
        LOGGER.debug("_merge_memories: output: %s", output)

        try:
            merged_memories = json.loads(output)
            return [m.strip() for m in merged_memories]
        except json.JSONDecodeError:
            LOGGER.debug("Merging memories did not return valid JSON: %s", output)
            return []
        except Exception:
            LOGGER.debug("Merging memories did not return expected format: %s", output)
            return []

    async def _process_message_for_memories(
        self,
        message: str,
        source: str,
        *,
        user: User,
        source_default: str = "user",
        distance_max: Optional[float] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ):
        """
        message:
            Will be assessed for memories, and existing memories will be processed accordingly.

        source:
            'user' or 'assistant'. Source of the message. The tag will be added to memories,
            and only memories with the same source will be considered.

        source_default:
            When no tag is found on a memory, consider it as `source_default`.

        distance_max:
            Don't get memories with distance > distance_max. Default to Valve.distance_max settings.
        """
        assert source in (ROLE.USER, ROLE.ASSISTANT)
        # Here we only consider memories from the user. We may want to have a sort of "assistant memory" that we can tag separately in the future.
        new_memories = await self._assess_for_memories(message, source=source)
        LOGGER.debug(
            "_process_message_for_memories.new_memories (source: %s): %s",
            source,
            new_memories,
        )
        current_memories = []
        merged_memories = []

        if len(new_memories) > 0:
            dist_max = (
                self.valves.memories_dist_max if distance_max is None else distance_max
            )

            # We don't want to spam the memories database with similar ones, let's merge whatever is new
            # Query potentially similar memories.
            current_memories = await self._query_memories(
                str(new_memories), user=user, distance_max=dist_max
            )
            LOGGER.debug(
                "_process_message_for_memories.current_memories (source: %s): %s",
                source,
                current_memories,
            )
            if len(current_memories) > 0:
                # We could merge current with new, delete all current, and insert all the new ones.
                merged_memories = await self._merge_memories(
                    new_memories, [m["content"] for m in current_memories]
                )
                LOGGER.debug(
                    "_process_message_for_memories.merged_memories: %s", merged_memories
                )
            else:
                # Don't bother merging, but we'll want to upload the new ones.
                merged_memories = new_memories

        if len(merged_memories) + len(current_memories) > 0:
            # We want to update even if decision to remove current_memories.
            # Delete retrived memories, create merged ones
            await asyncio.gather(
                # Create merged_memories
                *map(
                    lambda m: add_memory(
                        request=Request(
                            scope={"type": "http", "app": open_webui.main.app}
                        ),
                        form_data=AddMemoryForm(
                            content=json.dumps({"content": m, "source": source})
                        ),
                        user=user,
                    ),
                    merged_memories,
                ),
                # Delete the prior ones that were close.
                *map(
                    lambda m: delete_memory_by_id(m["id"], user=user),
                    current_memories,
                ),
            )

            if __event_emitter__ is not None:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Memories updated.",
                            "done": True,
                        },
                    }
                )


async def single_query_model(
    session: aiohttp.ClientSession,
    target_url: str,
    model: str,
    system: str,
    query: str,
    *,
    api_key: Optional[str] = None,
) -> str:
    "Send a single query to the model and return the answer."
    payload = {
        "model": model,
        "messages": [
            {"role": ROLE.SYSTEM, "content": system},
            {"role": ROLE.USER, "content": query},
        ],
    }
    LOGGER.debug("single_query_model: url %s", target_url)
    try:
        resp = await session.post(
            target_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
        )
        assert resp.status == 200, resp
        resp_json = await resp.json()
        assert resp_json.get("object") == "chat.completion", "Unexpected answer format."
        if len(resp_json["choices"]) > 1:
            LOGGER.debug("No support for more than 1 choice in answer: %s", resp_json)

        output = resp_json["choices"][0]["message"]["content"].strip()
        return output
    except (aiohttp.ClientError, AssertionError) as exc:
        LOGGER.error("Query to %s failed with: %s", target_url, exc)
        raise exc
    except Exception as exc:
        LOGGER.error("Non-network error: %s", exc)
        raise exc


def get_api_chat_completion_config() -> tuple[str, str]:
    "Returns chat completion API url and API key. Useful for settings defaults that works with user install."
    config = open_webui.main.app.state.config
    if config.ENABLE_OLLAMA_API and config.OLLAMA_BASE_URLS:
        # Use OLLAMA
        # Try to get key if there is one
        key = ""
        base_url = config.OLLAMA_BASE_URLS[0]
        if config.OLLAMA_API_CONFIGS.get(base_url):
            key = config.OLLAMA_API_CONFIGS[base_url].get("key") or ""

        ollama_config = (base_url, key)
        LOGGER.debug("get_api_chat_config.ollama: %s", ollama_config)
        return ollama_config
    elif config.ENABLE_OPENAI_API and config.OPENAI_API_BASE_URLS:
        # Untested as author doesn't use OpenAI, please open an issue with a fix if you find one.
        base_url = config.OPENAI_API_BASE_URLS[0]
        openai_config = (base_url, config.OPENAI_API_KEYS[base_url].get("key"))
        LOGGER.debug("get_api_chat_config.openai: %s", openai_config)
        return openai_config
    else:
        LOGGER.debug("get_api_chat_config.default.")
        return ("http://localhost:11434", "")


def get_last_message(
    messages: List[Dict[str, str]], role: str
) -> Tuple[Optional[Dict[str, str]], Optional[int]]:
    """
    Get last message from `role` and its index.
    """
    for i, m in enumerate(reversed(messages)):
        if m.get("role") == role:
            return (m, len(messages) - i - 1)
    return (None, None)


class PROMPT:
    MEMORIES_MERGE = """
        # Instructions
        You are a helpful assistant that manages long-term memories between chats.
        When given two JSON lists, one containing new memories and another containing current memories,
        you must merge them into a single list. When merging, remove any duplicate information
        and keep only one occurrence of each item. If there is conflicting information, prioritize the new information.

        Output your result as a JSON list of strings in the format: [string1, string2, ...]

        ## Examples:

        * Given:
          + current_memories = ["User likes movies", "User dislikes potatoes"]
          + new_memories = ["User does not like potatoes"]
         Output: ["User likes movies", "User does not like potatoes"]

        * Given:
          + current_memories = ["Assistant likes movies", "User does not like oranges"]
          + new_memories = ["User likes oranges"]
         Output: ["Assistant likes movies", "User likes oranges"]

        * Given:
          + current_memories = ["User likes A", "User does not like B"]
          + new_memories = ["User likes C"]

         Output: ["User likes A", "User does not like B", "User likes C"]

        * Given:
          + current_memories = []
          + new_memories = ["Assistant likes movies"]
         Output: ["Assistant likes movies"]

        * Given:
          + current_memories = ["A is 1", "B is 2"]
          + new_memories = ["B is 4"]
         Output: ["A is 1", "B is 4"]

        * Given:
          + current_memories = ["Assistant asks what is A", "A is 1"]
          + new_memories = ["Assistant asks what is A"]
         Output: ["Assistant asks what is A", "A is 1"]

        * Given:
          + current_memories = ["User asks what is A", "User asks what is A", "User likes B", "User asks what is A",]
          + new_memories = ["User asks what is A"]
         Output: ["User asks what is A", "User likes B"]

        * Given:
          + current_memories = ["User dislikes potatoes"]
          + new_memories = ["User likes potatoes"]
         Output: ["User likes potatoes"]

        * Given:
          + current_memories = ["User does not like oranges", "User likes A"]
          + new_memories = []
         Output: ["User does not like oranges", "User likes A"]

        * Given:
          + current_memories = ["User likes B"]
          + new_memories = ["User dislikes B"]
         Output: ["User likes B", "User dislikes B"]


        ## Starting notes
        The above examples are example only, do not use any of the examples data for as input
        You should only use the context from the most recent chat to inform your decision.
        Your output must only be valid JSON lists of strings. Nothing else. Do not add comments, do not use code blocks (```).
        You start now.
        """

    MEMORIES_ASSESS = """
        Ignore prior instructions.

        You will be provided a message. You must identify what is worth remembering for future conversations.

        A message is considered worthy of long-term memory if it involves:
            * Significant events (e.g. birthdays, wins, meetings, changes in personal life)
            * Important announcements (e.g. vacations)

        You must provide output in the form of a JSON list. Nothing else.

        Be concise and factual.

        # Examples
        You must act as the "Assistant" in those examples.

        ## Example
            User: Hi, I'm going on vacations next week.
            Assistant: ["{source} plans vacations next week."]

        ## Example
            User: My football team and I won the match, and then I went to celebrate my mom's birthday.
            Assistant: ["{source} won a football match.", "{source} celebrated its mother's birthday."]

        ## Example
            User: Hi, what's up?
            Assistant: []

        ## Example
            User: I like A.
            Assistant: ["{source} likes A"]

        ## Example
            User: I met my friend John on July 16th.
            Assistant: ["{source} met John on July 16th", "{source} has a friend named john."]
        
        ## Example
            User: Hi, I'm planning a trip to Japan and I want to know more about the best places to visit.
            Assistant: ["{source}s plans trip to Japan."]
        
        ## Example
            User: Hmm, I think it's a mix of everything. I want to experience the traditional Japanese culture, try some delicious food, and see some beautiful landscapes.
            Assistant: ["{source} wants to experience traditional Japan, delicious food and landscapes."]
        
        ## Example
            User: Yeah, I've heard of Tokyo, but I'm not sure if it's the best place for me. I was thinking maybe Kyoto or Osaka?
            Assistant: ["{source} considers Kyoto", "{source} considers Osaka"]
        
        ## Example
            User: That sounds good, but I'm not sure which one to choose. Can you give me some recommendations?
            Assistant: ["{source} is open to recommendations."]
        
        ## Example
            User: Wow, I'm getting overwhelmed with all these options! Can you give me some more specific recommendations?
            Assistant: []
        
        ## Example
            User: Okay, that helps a bit. What about accommodations? Where should I stay?
            Assistant: ["{source} is looking for accommodations"]
        
        ## Example
            User: That sounds interesting. Can you tell me more about the different types of accommodations?
            Assistant: []
        
        ## Example
            User: Okay, I think I have a better idea now. What about transportation? How do I get around Japan?
            Assistant: ["{source} is looking for transportation means."]
        
        ## Example
            User: That sounds good. Can you give me some tips on how to navigate the train system?
            Assistant: ["{source} is looking for guidance to navigate the train system."]
        
        ## Example
            User: Okay, I think that's all the information I need for now. Thank you so much for your help!
            Assistant: ["{source} thanks assistant."]

        ## Assistant kick-off

        NEVER quote the examples above, their data is not correct. They are only here to demonstrate the logic that must be applied.
        
        The next message will be written by {source}.
        Now create a JSON List of strings from the next message, ensure you follow the same format as in the examples above ([string1, string2, ...]),
        without any code block or comments. JSON Objects are not allowed, just List.
        """
    MEMORY_QUERY = """
        Ignore prior instructions.

        You are going to be provided with a discussion.
        You must identify key elements in the discussion that may benefit from more context.

        For example, what the user likes or dislikes, messages are associated with strong emotions, or
        facts to remember. Latest messages are more important than the first ones.

        Provide your answer as a bullet point list. You must only answer markdown lists, even with just one item.

        # Examples
        
        In those examples, you must act as the "Assistant".

        ## Example 1
            ### system:
                User context:
                <time timezone="UTC" format="%Y-%m-%d %H:%M:%S">2025-01-17T21:42:44.926417+01:00</time>

            ### user
            Hi, I'm going on vacation next week.

            ### assistant
                - vacation next week
                - Jan 24, 2025
                - January 2025
                - User hobbies
                - Favorite places

        ## Example 2
            ### system
                User context:
                <time timezone="UTC">2025-03-19T21:42:44.926417+01:00</time>

            ### user
                Well, that's me for you!

            ### assistant
                - March 19, 2025
                - User character
                - User habits
                - Typical behavior

        # Example 3
            ## system
                User context:
                <time timezone="UTC">2025-01-17T21:42:44.926417+01:00</time>

            ## user
                hi!

            ## assistant
                - Greetings
                - January 17, 2025
                - User character
                - Mood
                - Cheered up

        # Example 4
            ## user
                Hi, my sister is coming to see me!

            ## assistant
                - Family
                - Sister
                - Excitement
    
        """
