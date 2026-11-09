from dotenv import load_dotenv
from crewai import Task
from crewai import Crew
from crewai import Agent, LLM

load_dotenv()

llm = LLM(model="gpt-4o-mini")

writer = Agent(
    role="Writer",
    goal="Write short, simple text",
    backstory="You write clearly for beginners.",
    llm=llm,
)
print("Writer ready!")

write_poem = Task(
    description="Write a short 2-line poem about the sun.",
    expected_output="A simple 2-line poem.",
    agent=writer,
)
print("Task ready!")


crew = Crew(agents=[writer], tasks=[write_poem])

# Inside a Jupyter notebook we must use kickoff_async() with "await".
# (A plain .py script uses crew.kickoff() instead.)

# result = await crew.kickoff_async()
result = crew.kickoff() 
print(result)