!!!Pass the pdf file of Wikipedia 2024 and 2025 and prompt below to LLM to generate topics_plan.json, I used Gemini 3.0 Pro.

Role: You are a Lead Data Curator for a high-stakes Event Reasoning Benchmark.

Task:
Analyze the provided "Wikipedia 2024,2025" pdf and generate a `TOPICS_PLAN` JSON list containing exactly **96 distinct, high-impact real-world events**.

**CRITICAL CONSTRAINT: THE "ENTITY ANCHOR" STRATEGY**
You must optimize search queries for the GDELT Keyword Search API.
1. **NO YEARS**: Do NOT include "2024" or "2025" in queries. The API handles time filtering.
2. **MANDATORY ENTITY**: Every query MUST contain the specific **Subject Entity** in double quotes (e.g., `"SpaceX"`, `"Sheikh Hasina"`).
3. **MINIMALIST ACTION**: Use only 1-3 keywords to describe the action. Remove adjectives like "successfully", "major".

**CRITICAL STYLE REQUIREMENT: Headline-Style Topics**
The `topic` field must be a **concise, full-sentence headline** summarizing the event (Subject + Verb + Outcome).

**CRITICAL CONSTRAINT: No Duplicates**
You must NOT generate topics that overlap with the following existing dataset entries:
[
"SpaceX successfully launches first crewed Dragon mission to ISS", "Ever Given container ship blocks Suez Canal", "Biden inaugurated as 46th President", "FIFA World Cup 2022 held in Qatar", "James Webb Space Telescope successfully launched", "Queen Elizabeth II dies", "Apple unveils Vision Pro", "Microsoft acquires Activision Blizzard", "Meta launches Threads", "NASA Artemis I launches", "China Chang'e 5 returns samples", "Blue Origin New Shepard carries Jeff Bezos", "AlphaFold predicts 200 million protein structures", "Taiwan Hualien earthquake", "Iranian President Raisi killed in crash", "Landslide buries villages in Papua New Guinea", "Trump survives assassination attempt", "CrowdStrike update triggers IT outage", "Hezbollah pagers explode", "Israel eliminates Hassan Nasrallah", "Iran launches ballistic missile attack", "Floods devastate Valencia", "Syrian rebels seize Damascus", "South Korean President Yoon faces impeachment", "DeepSeek unveils large language model", "UK PM Liz Truss resigns", "Amazon rainforest wildfires", "UK holds Brexit referendum", "Trump supporters storm US Capitol", "Shinzo Abe assassinated", "Elon Musk acquires Twitter", "Trump re-elected as US President", "7.9 magnitude earthquake in Myanmar", "Elon Musk appointed Efficiency Minister", "US imposes trade tariffs", "Huawei unveils HarmonyOS", "Facebook changes name to Meta", "TikTok US security investigation", "OpenAI releases ChatGPT", "Nvidia stock surges", "California wildfires", "Texas winter storm", "Beirut port explosion", "Myanmar military coup", "Qasem Soleimani killed", "Japan discharges Fukushima wastewater", "Seoul Itaewon crowd crush", "Yellow Vest protests", "Australia bushfires", "Wagner Group mutiny", "George Floyd protests", "Me Too movement", "Taliban regains Afghanistan", "Silicon Valley Bank collapses"
]

**Query Structure (JSON List Format)**
To ensure maximum recall, provide a **list of 3 distinct queries** for the Gold category.
1. **`gold_queries`**: A list of 3 strings.
   - Query 1: `"Entity" Action` (Standard verb)
   - Query 2: `"Entity" Outcome/Result` (Focus on the result)
   - Query 3: `"Entity" Key_Context/Alias` (Alternative keywords or specific project names)
   - *Example*: `["\"SpaceX\" catch booster", "\"Starship\" mechazilla landing", "\"SpaceX\" super heavy test"]`
2. **`hard_negative_queries`**: A list of 2 strings finding the SAME entity in DIFFERENT/ROUTINE events.
   - *Example*: `["\"SpaceX\" Starlink launch", "\"SpaceX\" Falcon 9"]`
3. **`soft_negative_queries`**: A list of 2 strings finding COMPETITOR entities.
   - *Example*: `["\"Blue Origin\" New Glenn", "\"NASA\" SLS"]`

**Target Distribution (Total 96 Topics):**
1. **Politics & Geopolitics**: 32 topics
2. **Space**: 20 topics
3. **Technology**: 20 topics
4. **Disaster & Environment**: 16 topics
5. **Sports**: 4 topics
6. **Economy / World Events**: 4 topics

**Output Format (Strict JSON):**
[
  {
    "id": 101,
    "category": "Space",
    "topic": "SpaceX Starship successfully catches Super Heavy booster on fifth flight test",
    "gold_queries": [
      "\"SpaceX\" catch booster",
      "\"Starship\" mechazilla",
      "\"Super Heavy\" mid-air catch"
    ],
    "hard_negative_queries": [
      "\"SpaceX\" Starlink launch",
      "\"SpaceX\" Falcon 9 mission"
    ],
    "soft_negative_queries": [
      "\"Blue Origin\" launch",
      "\"NASA\" Artemis moon"
    ]
  },
  {
    "id": 102,
    "category": "Politics & Geopolitics",
    "topic": "Nicaragua constitutional reforms establish Co-Presidency for Daniel Ortega and Rosario Murillo",
    "gold_queries": [
      "\"Nicaragua\" constitutional reform",
      "\"Ortega\" Murillo co-president",
      "\"Nicaragua\" elevation co-presidency"
    ],
    "hard_negative_queries": [
      "\"Daniel Ortega\" election victory rally",
      "\"Ortega\" speech 2023"
    ],
    "soft_negative_queries": [
      "\"El Salvador\" Bukele gangs",
      "\"Venezuela\" Maduro referendum"
    ]
  },
  ... (Generate 96 items)
]
