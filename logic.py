"""Shared LangGraph workflows for Groq multimodal and text flows."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
TEXT_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 1024
TOP_P = 1.0

VISION_SYSTEM_PROMPT = """
You are an experienced culinary analyst.

- Look at the provided meal photo and list ONLY the dishes or food items you clearly observe.
- Respond strictly with valid JSON: an array of objects with keys `name`, `serving_size`, `quantity`, and optional `notes`.
- `serving_size` describes the unit (e.g., "1 pc", "120 g", "½ cup") and `quantity` is numeric for downstream calculations.
- DO NOT include macros or micros here.
- If uncertain, include a note such as "I don't know".
- If the photo is not a meal, respond with an empty JSON array `[]`.
""".strip()

NUTRITION_SYSTEM_PROMPT = """
You are an expert dietitian.

- You will receive a JSON array describing dishes (`name`, `serving_size`, `quantity`).
- Return a JSON array with the same number/order of items.
- For each item include: `name`, `serving_size`, `quantity`, `protein`, `carbs`, `fiber`, `fats`, `calories`, `calcium`, `iron`, `potassium`, and any vitamins/minerals you can confidently estimate (vitamin A, B1, B6, B12, C, D, K, etc).
- Use numeric values; if unknown, set the field to null or "I don't know".
- Ensure calories roughly equal 4*protein + 4*carbs + 9*fats.
- Never add dishes that were not in the input list.
""".strip()


def _build_llm(model_id: str) -> ChatGroq:
    """Create the Groq-backed chat model, ensuring credentials exist."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set the GROQ_API_KEY environment variable first.")

    return ChatGroq(
        model=model_id,
        temperature=TEMPERATURE,
        max_tokens=MAX_COMPLETION_TOKENS,
        top_p=TOP_P,
        streaming=True,
    )


def _file_to_data_url(path: str) -> str:
    """Convert a local image file to a base64 data URL."""
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return build_data_url(image_path.name, image_path.read_bytes())


def build_data_url(file_name: str, data: bytes) -> str:
    """Create a base64 data URL from raw bytes."""
    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_image_reference(image_input: str) -> str:
    """Return a URL or data URL that Groq's API can consume."""
    if image_input.startswith(("http://", "https://", "data:")):
        return image_input
    return _file_to_data_url(image_input)


VISION_LLM = _build_llm(VISION_MODEL_ID)
TEXT_LLM = _build_llm(TEXT_MODEL_ID)


class AgentState(TypedDict):
    """Shared LangGraph state for the vision agent."""

    messages: Annotated[List[BaseMessage], add_messages]


def _message_to_text(message: BaseMessage) -> str:
    """Extract printable text from any LangChain message."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(part.get("text", ""))
            elif isinstance(part, str):
                chunks.append(part)
        return "".join(chunks)
    return ""


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_items_from_text(text: str) -> List[Dict]:
    """Attempt to parse structured nutrition data from the LLM output."""
    if not text:
        return []

    candidates = [text]
    candidates.extend(m.group(1) for m in JSON_BLOCK_RE.finditer(text))

    for candidate in candidates:
        snippet = candidate.strip()
        if not snippet:
            continue
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _vision_node(state: AgentState) -> AgentState:
    """Single LangGraph node that calls the multimodal model."""
    response = VISION_LLM.invoke(state["messages"])
    return {"messages": [response]}


def _text_node(state: AgentState) -> AgentState:
    """Text-only node for refining nutrition details."""
    response = TEXT_LLM.invoke(state["messages"])
    return {"messages": [response]}


def _build_graph(node_name: str, node_fn):
    """Wire a single-node graph and compile it for execution."""
    workflow = StateGraph(AgentState)
    workflow.add_node(node_name, node_fn)
    workflow.add_edge(START, node_name)
    workflow.add_edge(node_name, END)
    return workflow.compile()


VISION_APP = _build_graph("vision_model", _vision_node)
TEXT_APP = _build_graph("nutrition_model", _text_node)


def _build_user_message(image_reference: str, prompt: str) -> HumanMessage:
    """Create the multimodal message payload expected by Groq."""
    if not image_reference:
        raise ValueError("image reference is required for the multimodal call.")

    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_reference}},
        ]
    )


def _build_nutrition_message(items: List[Dict], prompt: str) -> HumanMessage:
    """Create the text-only payload for nutrition refinement."""
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    content = (
        f"{prompt}\n\nHere are the confirmed dishes and their quantities:\n"
        f"```json\n{payload}\n```"
    )
    return HumanMessage(content=content)


def stream_multimodal_response(image: str, prompt: str = "") -> None:
    """
    Stream the model response chunk-by-chunk to stdout.

    Example:
        stream_multimodal_response("images/1.jpg", "Describe the dish")
    """
    image_reference = _normalize_image_reference(image)
    user_message = _build_user_message(image_reference=image_reference, prompt=prompt)
    system_message = SystemMessage(content=VISION_SYSTEM_PROMPT)
    initial_state: AgentState = {"messages": [system_message, user_message]}

    for update in VISION_APP.stream(initial_state, stream_mode="messages"):
        message: BaseMessage | None = None

        if isinstance(update, BaseMessage):
            message = update
        elif (
            isinstance(update, tuple)
            and len(update) == 2
            and isinstance(update[0], BaseMessage)
        ):
            message = update[0]
        elif isinstance(update, dict):
            possible = update.get("messages")
            if isinstance(possible, list) and possible and isinstance(
                possible[-1], BaseMessage
            ):
                message = possible[-1]

        if not message:
            continue

        text = _message_to_text(message)
        if not text:
            continue

        print(text, end="", flush=True)

    print()


def detect_food_items(image: str, prompt: str = "") -> Dict[str, Any]:
    """Run the image model and return structured dish detections."""
    image_reference = _normalize_image_reference(image)
    user_message = _build_user_message(image_reference=image_reference, prompt=prompt)
    system_message = SystemMessage(content=VISION_SYSTEM_PROMPT)
    final_state = VISION_APP.invoke({"messages": [system_message, user_message]})

    messages = final_state.get("messages", []) if isinstance(final_state, dict) else []
    response = messages[-1] if messages else None
    text = _message_to_text(response) if response else ""
    items = _parse_items_from_text(text)
    return {"message": response, "text": text, "items": items}


def enrich_food_items(items: List[Dict[str, Any]], prompt: str = "") -> Dict[str, Any]:
    """Run the text model to attach macros/micros to confirmed dishes."""
    if not items:
        return {"message": None, "text": "", "items": []}

    user_message = _build_nutrition_message(items, prompt)
    system_message = SystemMessage(content=NUTRITION_SYSTEM_PROMPT)
    final_state = TEXT_APP.invoke({"messages": [system_message, user_message]})

    messages = final_state.get("messages", []) if isinstance(final_state, dict) else []
    response = messages[-1] if messages else None
    text = _message_to_text(response) if response else ""
    enriched_items = _parse_items_from_text(text)
    return {"message": response, "text": text, "items": enriched_items}


def analyze_image(image: str, prompt: str = "") -> Dict[str, Any]:
    """Backward-compatible alias used elsewhere in the project."""
    return detect_food_items(image=image, prompt=prompt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a Groq multimodal response via LangGraph."
    )
    parser.add_argument(
        "--image",
        help="Path to a local image or a publicly reachable image URL.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional text instructions that accompany the image.",
    )
    return parser.parse_args()


def _resolve_image_input(cli_value: str | None) -> str:
    """Use the CLI value or fall back to interactive input."""
    if cli_value:
        return cli_value

    user_input = input("Enter an image path or URL: ").strip()
    if not user_input:
        raise ValueError("Image path or URL is required.")
    return user_input


if __name__ == "__main__":
    args = _parse_args()
    image_input = _resolve_image_input(args.image)
    stream_multimodal_response(image=image_input, prompt=args.prompt)
