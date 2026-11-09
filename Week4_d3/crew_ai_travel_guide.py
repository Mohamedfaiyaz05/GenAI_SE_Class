import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


# =====================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please add it to your .env file."
    )


# =====================================================
# STEP 2: CONFIGURE THE LLM
# =====================================================

llm = LLM(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    api_key=api_key,
    temperature=0.3
)


# =====================================================
# STEP 3: CREATE THE TRAVEL PLANNER AGENT
# =====================================================

travel_planner = Agent(
    # WHO IS PERFORMING THE WORK?
    role="Experienced Travel Planner",

    # WHAT IS THE AGENT'S OVERALL OBJECTIVE?
    goal=(
        "Recommend exactly three interesting and worthwhile "
        "places to visit in the city provided by the user."
    ),

    # WHAT EXPERTISE AND PERSPECTIVE DOES THE AGENT HAVE?
    backstory="""
    You are an experienced travel planner who specializes
    in helping first-time visitors explore new cities.

    Your audience consists of travellers who want practical,
    interesting and easy-to-understand recommendations.

    You have knowledge of popular tourist attractions,
    historical landmarks, cultural destinations and
    unique local experiences.

    You understand that travellers need useful information,
    not lengthy descriptions.

    You always explain why a place is worth visiting
    and what visitors can experience there.

    You communicate in a clear, concise and friendly manner.
    Avoid generic recommendations and unsupported facts.
    """,

    # LLM USED BY THE AGENT
    llm=llm,

    # SHOW AGENT EXECUTION DETAILS
    verbose=True,

    # LIMIT UNNECESSARY ITERATIONS
    max_iter=5
)


# =====================================================
# STEP 4: CREATE THE TASK
# =====================================================

travel_task = Task(
    description="""Create a practical travel guide for {city} in {month}, tailored to typical climate, seasonal conditions, and a first-time visitor.

Complete these 3 sections:

### SECTION 1: TOP 3 PLACES TO VISIT
- Select 3 distinct, popular/historical/cultural/unique places *strictly* within {city}.
- For each, provide Name, Category, Why visit, and What to experience.

### SECTION 2: WHAT TO CARRY
- **Weather & Clothing:** Tailor to {month} climate (temperature, rainfall, humidity). Specify clothing type (lightweight vs warm), jacket/sweater/raincoat/umbrella needs, and footwear/accessories.
- **Personal Essentials:** Sunglasses, sunscreen, reusable water bottle, power bank, cables, adapters, personal meds/first-aid, documents, and destination-specific items.
- **International Prep (If international only; mark 'NOT APPLICABLE' for domestic):** Passport/visa, Forex card/currency, transaction/ATM fees, travel insurance, eSIM/roaming, and plug adapters. (Do not recommend Forex for domestic).

### SECTION 3: SAVE MONEY, SAVE TIME & STAY SAFE
- **Money-Saving:** 3 practical tips (early booking, transport comparison, avoiding tourist traps/hidden fees, affordable local transit/passes).
- **Time-Saving:** 3 practical tips (optimal visit order for the 3 places, checking hours/advance bookings, beating peak traffic/crowds).
- **Safety:** 3 important precautions (destination-specific risks, doc copies, belongings security, emergency contacts, weather/environmental/health hazards, insurance for international trips).

**Rules:** Tailor strictly to {city} and {month}. Do not invent real-time data (prices, hours, forecasts); advise verifying via official sources. Avoid unnecessary purchases.""",

    expected_output="""Return a structured guide with EXACTLY THREE sections:

### SECTION 1: TOP 3 PLACES TO VISIT
1. Place name:
   Category:
   Why visit:
   What to experience:
2. Place name:
   Category:
   Why visit:
   What to experience:
3. Place name:
   Category:
   Why visit:
   What to experience:

### SECTION 2: WHAT TO CARRY
**Weather and clothing:**
- Recommended clothing:
- Jacket / sweater / raincoat / umbrella:
- Footwear and accessories:

**Travel essentials:**
- Documents:
- Electronics:
- Health and personal essentials:

**International travel essentials:**
- Forex card / currency:
- Passport / visa:
- Travel insurance:
- SIM / eSIM:
- Power adapter:
*(Mark international-only items as NOT APPLICABLE for domestic travel).*

### SECTION 3: SAVE MONEY, SAVE TIME AND STAY SAFE
- **Money-saving tips:** (3 practical tips)
- **Time-saving tips:** (3 practical tips)
- **Safety precautions:** (3 important precautions)
- **Final checklist:** (Key items/verification before departure)""",

    agent=travel_planner
)


# =====================================================
# STEP 5: ASSEMBLE THE CREW
# =====================================================

travel_crew = Crew(
    agents=[travel_planner],
    tasks=[travel_task],
    process=Process.sequential,
    verbose=True
)


# =====================================================
# STEP 6: GET INPUT FROM THE USER
# =====================================================

city = input("\nEnter the city you want to explore: ").strip()
if not city:
    raise ValueError("City name cannot be empty.")

month = input("Enter the travel month (e.g., December): ").strip()
if not month:
    raise ValueError("Travel month cannot be empty.")


# =====================================================
# STEP 7: EXECUTE THE CREW
# =====================================================

print("\nStarting Travel Planner Crew...\n")

result = travel_crew.kickoff(
    inputs={
        "city": city,
        "month": month
    }
)


# =====================================================
# STEP 8: DISPLAY THE FINAL RESULT
# =====================================================

print("\n")
print("=" * 60)
print("            YOUR TRAVEL RECOMMENDATIONS")
print("=" * 60)

print(result)

print("=" * 60)
print("Travel planning completed!")