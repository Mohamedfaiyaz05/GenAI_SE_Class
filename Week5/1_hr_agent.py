
import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


async def main():

    # Step 1: Initialize the OpenAI model
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    try:
        # Step 2: Create the HR Assistant Agent
        hr_agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant."
        )

        # Step 3: Define the task
        task = """
        Write a job description for a Junior Python Developer.

        Requirements:
        - Exactly 5 lines.
        - Each line must be one bullet point.
        - Include responsibilities, Python skills,
          qualifications, and teamwork.
        - Do not include headings or extra explanations.
        """

        # Step 4: Run the agent and display its response
        await Console(
            hr_agent.run_stream(task=task)
        )

    finally:
        # Step 5: Close the model connection
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
