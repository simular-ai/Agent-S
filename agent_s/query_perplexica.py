import asyncio
from typing import NamedTuple, Literal, Union, List, Tuple

import websockets
import json
import os


class Message(NamedTuple):
    type: Literal["message", "messageEnd"]
    data: str


class SourceMetaData(NamedTuple):
    title: str
    url: str


class Source(NamedTuple):
    page_content: str
    meta_data: SourceMetaData


class Sources(NamedTuple):
    type: Literal["sources"]
    datas: List[Source]


def parse_message(response: str) -> Union[Message, Sources]:
    try:
        message = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response")

    if not isinstance(message, dict):
        raise TypeError("Expected a JSON object")

    message_type = message.get('type')
    data = message.get("data", "")
    if message_type in ["message", "messageEnd"]:

        return Message(type=message_type, data=data)

    elif message_type == "sources":
        sources: List[Source] = []
        if not isinstance(data, list):
            raise ValueError("Unable to parse response value")

        for source in data:
            meta_data = SourceMetaData(
                title=source.get("metadata").get("title"),
                url=source.get("metadata").get("url"),
            )
            sources.append(Source(
                page_content=source.get("pageContent"),
                meta_data=meta_data
            ))
        return Sources(
            type=message_type,
            datas=sources
        )

    raise ValueError("Unable to parse response value")


async def websocket_client(query: str) -> Tuple[str, Sources]:
    chat_id = os.urandom(20).hex()
    # WebSocket server URI
    uri = "ws://127.0.0.1:3001/?chatModel=GPT-4+omni+mini&chatModelProvider=openai&embeddingModel=Text+embedding+3+small&embeddingModelProvider=openai"

    async with websockets.connect(uri) as websocket:
        # Send an initial message if needed
        initial_message = {"type": "message", "message": {"chatId": chat_id,
                                                          "content": query},
                           "focusMode": "webSearch", "history": [["human", query]]}
        await websocket.send(json.dumps(initial_message))

        gathered_messages: str = ""
        sources: Sources = Sources(
            type="sources",
            datas=[]
        )

        while True:
            response = await websocket.recv()
            message = json.loads(response)

            # print(f"Received: {message}")

            try:
                message = parse_message(response)

            except (ValueError, TypeError) as e:
                print(f"Error: {e}")

            # Gather the message
            if isinstance(message, Message):
                gathered_messages += message.data
            elif isinstance(message, Sources):
                sources = message

            # Check for the termination message
            if message.type == "messageEnd":
                # print("Received messageEnd. Stopping message gathering.")
                break

        return gathered_messages, sources


def query_to_perplexica(query):
    response, response_sources = asyncio.get_event_loop().run_until_complete(websocket_client(query))

    return response


if __name__ == "__main__":

    query = "How to set Bing as the default search engine in Chrome on Ubuntu"
    
    gathered_messages, sources = asyncio.get_event_loop().run_until_complete(websocket_client(query))
    print("All gathered messages:")
    print(gathered_messages)

    print("Sources")
    print(sources)
