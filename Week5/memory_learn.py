
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)

load_dotenv()


async def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    # 1. Persistent ChromaDB memory
    memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="user_memories",
            persistence_path=str(
                Path(__file__).resolve().parent / "chroma_memory"
            ),
            k=5,
            score_threshold=0.0,
        )
    )

    # 2. OpenAI model
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    # 3. Agent responsible for extracting memories
    memory_agent = AssistantAgent(
        name="memory_extractor",
        model_client=model_client,
        system_message="""
You extract useful long-term memories from user messages.

Remember:
- Explicitly stated names
- Stable preferences
- Learning goals
- Ongoing projects
- Other useful, non-sensitive facts

Do not remember:
- Ordinary questions
- Temporary calculations
- Guesses or inferred facts
- Passwords, API keys, financial or medical details

Return ONLY a valid JSON array of strings.
If there is nothing worth remembering, return [].

Example:
Input: My name is Alex and I prefer Python examples.
Output:
["User's name is Alex", "User prefers Python examples"]
""",
    )

    try:
        print("AutoGen Chatbot with Automatic Memory")
        print("Type 'exit' to quit.\n")

        while True:

            user_input = await asyncio.to_thread(
                input, "You: "
            )
            user_input = user_input.strip()

            if user_input.lower() == "exit":
                break

            if not user_input:
                continue

            # 4. Fresh chat agent:
            # retrieves relevant memories automatically
            chat_agent = AssistantAgent(
                name="chat_assistant",
                model_client=model_client,
                memory=[memory],
                system_message=(
                    "You are a helpful assistant. "
                    "Use retrieved memories when relevant. "
                    "Do not invent user information."
                ),
            )

            # 5. Generate the chatbot response
            response = await chat_agent.run(
                task=user_input
            )

            print(
                "\nAssistant:",
                response.messages[-1].content
            )

            # 6. Extract useful facts automatically
            extraction = await memory_agent.run(
                task=(
                    "Extract durable, non-sensitive user facts "
                    "from this message. Return only JSON.\n\n"
                    f"User message: {json.dumps(user_input)}"
                )
            )

            raw_output = extraction.messages[-1].content

            # 7. Parse extracted facts
            try:
                facts = json.loads(raw_output)

                if not isinstance(facts, list):
                    facts = []

                facts = [
                    fact for fact in facts
                    if isinstance(fact, str) and fact.strip()
                ]

            except (json.JSONDecodeError, TypeError):
                facts = []
                print("Memory extraction returned invalid JSON.")

            # 8. Save facts automatically to ChromaDB
            for fact in facts:
                await memory.add(
                    MemoryContent(
                        content=fact,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "user_fact"}
                    )
                )

                print("[Memory saved]", fact)

            print()

    finally:
        await memory.close()
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
