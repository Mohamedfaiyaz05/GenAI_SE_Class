from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, LLM, Task, Crew, Process
from crewai_tools import FileReadTool

# 1. Configure LLM
llm = LLM(model="gpt-4o-mini")

# 2. Create the tool
read_tool = FileReadTool()

# 3. Give the tool to the agent
reader = Agent(
    role="File Reader",
    goal="Read files and answer questions using their contents",
    backstory="""
    You are a careful file-reading assistant.
    You read the requested file before answering.
    You do not guess information that is not present
    in the file.
    """,
    tools=[read_tool],
    llm=llm,
    verbose=True
)

# 4. Create a task
read_task = Task(
    description="""
    Read the file named sample.txt.

    Explain:
    1. What is Python?
    2. What are its main applications mentioned in the file?
    3. Summarize the file in three bullet points.

    Use the FileReadTool to read the file.
    Do not answer from your general knowledge alone.
    """,
    expected_output="""
    A concise answer containing:
    1. Explanation of Python based on the file.
    2. Applications mentioned in the file.
    3. Three-bullet summary.
    """,
    agent=reader
)

# 5. Assemble the crew
crew = Crew(
    agents=[reader],
    tasks=[read_task],
    process=Process.sequential,
    verbose=True
)

# 6. Execute
result = crew.kickoff()

# 7. Print the result
print("\nFINAL ANSWER:")
print(result)