import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# =====================================================
# STEP 1: LOAD ENVIRONMENT & TOOLS
# =====================================================
load_dotenv()

# Initialize the Serper Google Search tool
search_tool = SerperDevTool()

# Configure the LLM
llm = LLM(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0.3
)


# =====================================================
# STEP 2: CREATE THE SPECIALIZED AGENTS
# =====================================================

weather_researcher = Agent(
    role="Weather Researcher",
    goal="Monitor and search for severe weather updates including rain, floods, cyclones, heatwaves, and storm alerts.",
    backstory="An expert meteorological research assistant skilled at scanning live web data for weather emergencies and environmental threats.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

child_safety_researcher = Agent(
    role="Child Safety Researcher",
    goal="Search for critical updates regarding child safety, education policy, public health advisories, and school safety alerts.",
    backstory="A vigilant research specialist focused on education, child welfare, and public health bulletins that impact children and schools.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

public_news_researcher = Agent(
    role="Public News Researcher",
    goal="Search for government announcements, public safety updates, major transport disruptions, and civic alerts.",
    backstory="A meticulous public policy and news researcher tracking official government notifications and civic announcements.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

news_analyst = Agent(
    role="News Analyst",
    goal="Verify facts, remove duplicates and irrelevant news, prioritize findings, and compile the final structured daily alert report.",
    backstory="A senior intelligence analyst skilled at synthesizing multi-domain research feeds into clean, prioritized executive briefs.",
    tools=[],  # Pure analysis/synthesis agent
    llm=llm,
    verbose=True
)


# =====================================================
# STEP 3: DEFINE THE TASKS
# =====================================================

weather_task = Task(
    description="Search the web for active weather alerts, heavy rain, floods, cyclones, or heatwaves in {location} for today.",
    expected_output="A structured summary of weather alerts and conditions with exact sources.",
    agent=weather_researcher
)

child_safety_task = Task(
    description="Search the web for any active school closures, child health warnings, or educational safety alerts in {location} for today.",
    expected_output="A summary of child safety, health, and school alerts with exact sources.",
    agent=child_safety_researcher
)

public_news_task = Task(
    description="Search the web for major government announcements, public safety threats, or transport disruptions in {location} today.",
    expected_output="A summary of public safety and government news with exact sources.",
    agent=public_news_researcher
)

analysis_task = Task(
    description="Review the findings from the weather, child safety, and public news researchers. Cross-verify details, eliminate duplicates, prioritize by severity, and compile the Final Daily Alert Report.",
    expected_output="""A final structured report containing:
    - **Priority:** (High / Medium / Low)
    - **Summary:** Concise briefing of verified items
    - **Date:** Current date
    - **Source URL(s):** Direct links from the research
    - **Recommended Action:** Actionable next steps""",
    agent=news_analyst,
    # Context passes the outputs of the research tasks directly into the analyst task
    context=[weather_task, child_safety_task, public_news_task]
)


# =====================================================
# STEP 4: ASSEMBLE AND EXECUTE THE CREW
# =====================================================

daily_monitoring_crew = Crew(
    agents=[
        weather_researcher, 
        child_safety_researcher, 
        public_news_researcher, 
        news_analyst
    ],
    tasks=[
        weather_task, 
        child_safety_task, 
        public_news_task, 
        analysis_task
    ],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    target_location = input("Enter target location to monitor: ").strip()
    
    print(f"\nStarting Daily Monitoring Crew for {target_location}...\n")
    
    result = daily_monitoring_crew.kickoff(
        inputs={"location": target_location}
    )
    
    print("\n" + "="*60)
    print("           FINAL DAILY ALERT REPORT")
    print("="*60)
    print(result)
    print("="*60)