from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM

llm = LLM(model="gpt-4o-mini")

writer = Agent(
    role="Content Writer",
    goal="Create simple, clear content",
    backstory="You write friendly content for beginners.",
    llm=llm,
    verbose=True
)
print("Agent ready!")


# Step A: list ideas
task1 = Task(
    description="List 3 fun facts about the moon.",
    expected_output="3 short fun facts.",
    agent=writer,
)

# Step B: use those facts to write a tweet
task2 = Task(
    description="Use the facts above to write one short tweet.",
    expected_output="One short tweet.",
    agent=writer,
)
print("Two steps ready!")

crew = Crew(
    agents=[writer],
    tasks=[task1, task2]   # they run top to bottom
    
)

# In a Jupyter notebook, use kickoff_async() with "await".
result = crew.kickoff()
print(result)