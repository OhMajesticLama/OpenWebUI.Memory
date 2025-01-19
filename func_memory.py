"""
title: Memory
author: ohmajesticlama
author_url: https://github.com/OhMajesticLama
funding_url: https://github.com/open-webui
version: 0.1
license: MIT

A function to automatically manage memories. Compatible with OpenWebUI 0.5.

/!\ This function adds and modifies memories. This may lead to memory data loss.

This function retrieves memories directly, it is recommended to disable the Settings->Personalization->Memory toggle.
"""

import sys
import datetime
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, List, Dict
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
    LOGGER.setLevel(level)

    LOGGER.debug("%s has %s handlers", LOGGER, len(LOGGER.handlers))
    for handler in LOGGER.handlers:
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

    LOGGER.addHandler(handler)
    return logger


# set_logs(LOGGER, GLOBAL_LOG_LEVEL)
set_logs(LOGGER, logging.DEBUG)


def log_exceptions(func: Callable[Any, Any]):
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
        async def _wrapper(*args, **kwargs):
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


class Filter:
    class Valves(BaseModel):
        model: str = Field(
            default="llama3.2:1b",
            description="Model to use to process memories. Defaults to same model as conversation to save GPU memory / reload time.",
        )

        n_memories: int = Field(
            default=5, description="Consider top N relevant memories."
        )

        memories_dist_min: float = Field(
            default=0.75,
            description="Ignore memories with a distance higher than this value.",
        )

        # We could try and pull from openwebui config
        chat_api_host: str = Field(
            default="http://host.docker.internal:11434",
            description="Defaults to a docker configuration with locally installed ollama on default port.",
        )

        api_key: str = Field(default="", description="Key for chat_api.")

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
        asyncio.run(self._session.close())

    async def _build_memory_query(self, messages: List[Dict[str, str]]) -> str:
        memory_keywords = await self.single_query_model(
            self.valves.model,
            PROMPT.MEMORY_QUERY,
            str([{"role": m["role"], "content": m["content"]} for m in messages]),
        )
        return memory_keywords

    async def _query_memories(
        self,
        query: str,
        *,
        user: User,
        distance_min: Optional[float] = None,
        n_memories: Optional[int] = None,
    ):
        n_memories = self.valves.n_memories if n_memories is None else n_memories
        memories = await query_memory(
            request=Request(scope={"type": "http", "app": open_webui.main.app}),
            form_data=QueryMemoryForm(
                content=query,
                k=str(n_memories),
            ),
            user=user,
        )
        dist_min = (
            self.valves.memories_dist_min if distance_min is None else distance_min
        )

        # memories example:
        #   ids=[['b8e68597-7e9d-4553-b5d9-7bbc503370fa', '9ba8d5df-1350-4c01-b2f9-b1b7cf369ba9']]
        #   documents=[['User likes sci-fi TV shows and sometimes watches Disney+, also likes potatoes.',
        #               'User displays interest in multiple language studies, focusing on French and Japanese.']]
        #   metadatas=[[{'created_at': 1737153170}, {'created_at': 1735153131}]]
        #   distances=[[0.8044559591349751, 0.8268496990203857]]
        #
        # Reshapeit for easier processing
        return [
            {
                "memory": memory,
                "metadata": {
                    "created_at": datetime.datetime.fromtimestamp(
                        meta["created_at"]
                    ).isoformat()
                },
                "dist": dist,
                "id": mid,
            }
            for memory, meta, dist, mid in zip(
                memories.documents[0],
                memories.metadatas[0],
                memories.distances[0],
                memories.ids[0],
            )
            if dist >= dist_min
        ]

    async def single_query_model(
        self,
        model: str,
        system: str,
        query: str,
    ):
        target_url = f"{self.valves.chat_api_host.strip('/')}/v1/chat/completions"
        return await single_query_model(
            self._session, target_url, model, system, query, api_key=self.valves.api_key
        )

    @log_exceptions
    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
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

        if not "id" in __user__:
            LOGGER.warn("No 'id' key in __user__. Do nothing.")
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
                    "description": f"Searching for memories...",
                    "done": False,
                },
            }
        )
        memory_query = await self._build_memory_query(messages)
        LOGGER.debug("memory_query: %s", memory_query)
        memories = await self._query_memories(memory_query, user=user)
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Memories search completed.",
                    "done": True,
                },
            }
        )

        # Get memories and inject them as context.

        LOGGER.debug("memories: %s", memories)

        # Inject context prior to last user message.
        if messages[-1].get("role") == ROLE.USER:
            # Let's insert a system message with context before the user.
            context = "<system_context>{}</system_context>".format(
                "\n  ".join(
                    (
                        f"<time timezone=UTC><!-- Be mindful user may not be in this timezone-->{datetime.datetime.now(datetime.UTC).isoformat()}</time>",
                        "<memories>{}</memories>".format(
                            "\n    ".join(
                                (
                                    f"""<memory created_at="{m['metadata']['created_at']}">{m['memory']}</memory>"""
                                    for m in memories
                                )
                            )
                        ),
                    )
                )
            )
            LOGGER.debug("Added context message: %s", context)
            context_message = {"role": ROLE.SYSTEM, "content": context}
            messages.insert(-1, context_message)
        else:
            LOGGER.debug(
                "Last message not from user, do nothing: %s", messages[-1].get("role")
            )
            # Let's add the time to the latest message, it might be useful.

        # if __user__.get("role", "admin") in ["user", "admin"]:
        #     messages = body.get("messages", [])

        #     max_turns = min(__user__["valves"].max_turns, self.valves.max_turns)
        #     if len(messages) > max_turns:
        #         raise Exception(
        #             f"Conversation turn limit exceeded. Max turns: {max_turns}"
        #         )

        return body

    @log_exceptions
    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> Dict[str, str]:
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

        # Is there something to remember?
        messages: Optional[Dict[str, str]] = body.get("messages")
        if not messages:
            # nothing to do here.
            LOGGER.debug("No 'messages' key in body. Do nothing.")
            return body

        if messages[-2].get("role") == ROLE.USER:
            # Here we only consider memories from the user. We may want to have a sort of "assistant memory" that we can tag separately in the future.
            new_memories = await self._assess_for_memories(messages[-1]["content"])
            current_memories = []
            merged_memories = []

            if len(new_memories) > 0:
                # We don't want to spam the memories database with similar ones, let's merge whatever is new
                # Query potentially similar memories. Be selective: we want to limit duplicates
                current_memories = await self._query_memories(
                    str(new_memories), user=user, distance_min=0.9
                )
                LOGGER.debug("current_memories: %s", current_memories)
                if len(current_memories) > 0:
                    # We could merge current with new, delete all current, and insert all the new ones.
                    merged_memories = await self._merge_memories(
                        new_memories, [m["memory"] for m in current_memories]
                    )
                    LOGGER.debug("merged_memories: %s", merged_memories)

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
                            form_data=AddMemoryForm(content=m),
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

            if len(merged_memories) > 0:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Memories updated.",
                            "done": True,
                        },
                    }
                )

        return body

    async def _assess_for_memories(self, content: str) -> List[str]:
        """
        Identify if there are elements to remember in content.
        """

        assessment = await self.single_query_model(
            self.valves.model,
            PROMPT.MEMORIES_ASSESS,
            content,
        )
        LOGGER.debug("_assess_for_memories: assessment: %s", assessment)

        try:
            new_memories = json.loads(assessment.strip())
        except json.JSONDecodeError as exc:
            LOGGER.debug(
                "Assessment for memory did not return valid JSON: %s", assessment
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
        except json.JSONDecodeError as exc:
            LOGGER.debug("Merging memories did not return valid JSON: %s", output)
            return []
        except Exception as exc:
            LOGGER.debug(
                "Merging memories did not return JSON in expected format: %s", output
            )
            return []


async def single_query_model(
    session: aiohttp.ClientSession,
    target_url: str,
    model: str,
    system: str,
    query: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
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
        # async with aiohttp.ClientSession() as session:
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
          + current_memories = ["User likes movies", "User does not like oranges"]
          + new_memories = ["User likes oranges"]
         Output: ["User likes movies", "User likes oranges"]
        * Given:
          + current_memories = ["User likes A", "User does not like B"]
          + new_memories = ["User likes C"]
         Output: ["User likes A", "User does not like B", "User likes C"]
        * Given:
          + current_memories = []
          + new_memories = ["User likes movies"]
         Output: ["User likes movies"]
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
    MEMORIES_MERGE_V0 = """
        Ignore prior instructions. You are an agent processing JSON data.

        You will be provided with 2 JSON lists of memories. One list is the new memories, the other is the current memories.
        You must merge the 2 lists and output a single list.
        When merging the list, you must remove redundant information and keep only one occurence. If there is conflicting information, keep the new one.

        You must provide output in the form of a JSON list of strings. Nothing else.

        Be concise and factual.
                       
        Examples
        --------
        You must act as the "Assistant" in those examples.

        Example 1: 
            User: 
                current_memories = ["User likes movies", "User dislikes potatoes"]
                new_memories = ["User does not like potatoes"]
                
            Assistant:
                ["User likes movies", "User does not like potatoes"]

        Example 2: 
            User: 
                current_memories = ["User likes movies", "User does not like oranges"]
                new_memories = ["User likes oranges"]
                
            Assistant:
                ["User likes movies", "User likes oranges"]

        Example 3: 
            User: 
                current_memories = ["User likes A", "User does not like B"]
                new_memories = ["User likes C"]
                
            Assistant:
                ["User likes A", "User does not like B", "User likes C."]

        """
    MEMORIES_ASSESS = """
        Ignore prior instructions.

        You will be provided a message. You must identify content that is worth remembering for long-term.
        You must provide output in the form of a JSON list. Nothing else.

        Be concise and factual.
                       
        Examples
        --------
        You must act as the "Assistant" in those examples.

        Example 1: 
            User: Hi, I'm going on vacations next week.     
            Assistant: ["Users plans vacations next week."]

        Example 2:
            User: My football team and I won the match, and then I went to celebrate my mom's birthday.
            Assistant: ["User won a football match.", "User celebrated its mother's birthday."]

        Example 3:
            User: Hi, what's up?
            Assistant: []

        Example 4:
            User: I like A.
            Assistant: ["User likes A"]
        """
    MEMORY_QUERY = """
        Ignore prior instructions.
        
        You are going to be provided with the history of this discussion.
        You must identify key elements in the discussion that may benefit from more context.
        
        For example, mentions of what the user likes or dislikes, that are associated with strong emotions, or
        facts to remember. Pay more attention to the latest messages in the discussion than to the first ones.

        Provide your answer as a bullet point list. You must only answer markdown lists, even with just one item.

        Examples
        --------
        In those examples, you must act as the "Assistant".

        Example 1: 
            System:
                User context:
                <time timezone="UTC">2025-01-17T21:42:44.926417+01:00</time>
                
            User: Hi, I'm going on vacation next week.
            
            Assistant: 
                - vacation
                - next week
                - Jan 24, 2025

        Example 2:
            System:
                User context:
                <time timezone="UTC">2025-03-19T21:42:44.926417+01:00</time>

            User:
                Well, that's me for you!

            Assistant:
                - User
                - March 19, 2025
                - User character
                - Habits

        Example 3:
            System:
                User context:
                <time timezone="UTC">2025-01-17T21:42:44.926417+01:00</time>

            User:
                hi!

            Assistant:
                - Greetings
                - January 17, 2025
                - User character
                - Mood


        Be concise.
        """
