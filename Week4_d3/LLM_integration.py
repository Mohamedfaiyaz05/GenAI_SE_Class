from crewai import Agent

from dotenv import load_dotenv
load_dotenv()

from crewai import LLM

# Our main AI brain for the whole course
llm = LLM(model="gpt-4o-mini")

print("OpenAI brain is ready!")

helper = Agent(
    role="Helper",
    goal="Answer questions simply",
    backstory="You explain things to beginners.",
    llm=llm,          # <-- the brain we made above
)
print(helper)
print("Agent created with the OpenAI brain!")