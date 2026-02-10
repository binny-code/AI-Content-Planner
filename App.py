import streamlit as st
from crewai import Agent, Task, Crew
from crewai.llm import LLM
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Content Planner",
    layout="wide"
)

# ---------- LOAD CSS ----------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    """
    <div class="header">
        <h1>AI Content Planner</h1>
        <p>Multi-Agent Content Strategy Generator using CrewAI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.header("Project Settings")

domain = st.sidebar.text_input(
    "Business / Product Domain",
    placeholder="AI-powered data analytics for healthcare startups"
)

run_button = st.sidebar.button("Generate Content Plan")

# ---------- LLM ----------
llm = LLM(
    model="groq/llama3-8b-8192",
    temperature=0.4
)

# ---------- AGENTS ----------
content_strategist = Agent(
    role="Content Strategist",
    goal="Design a high-impact content strategy aligned with business goals",
    backstory="Senior strategist with expertise in SEO, audience research, and content planning.",
    llm=llm
)

seo_analyst = Agent(
    role="SEO Analyst",
    goal="Identify keywords and search intent",
    backstory="SEO specialist experienced in keyword clustering and organic growth.",
    llm=llm
)

content_writer = Agent(
    role="Content Writer",
    goal="Create SEO-optimized blog outlines",
    backstory="Content writer skilled in technical and long-form writing.",
    llm=llm
)

# ---------- TASKS ----------
strategy_task = Task(
    description=(
        "Analyze the domain: {domain}. "
        "Define target audience, content goals, tone, and publishing cadence."
    ),
    expected_output="Audience persona and content strategy",
    agent=content_strategist
)

seo_task = Task(
    description=(
        "Perform keyword research for the domain: {domain}. "
        "List primary and secondary keywords with intent."
    ),
    expected_output="SEO keyword list with intent mapping",
    agent=seo_analyst
)

outline_task = Task(
    description=(
        "Create 5 SEO-optimized blog outlines using the strategy and keywords. "
        "Include title, H2s, and CTA."
    ),
    expected_output="Five structured blog outlines",
    agent=content_writer
)

# ---------- CREW ----------
crew = Crew(
    agents=[content_strategist, seo_analyst, content_writer],
    tasks=[strategy_task, seo_task, outline_task]
)

# ---------- RUN ----------
if run_button:
    if not domain:
        st.warning("Please enter a business domain.")
    else:
        with st.spinner("Generating content plan..."):
            result = crew.kickoff(inputs={"domain": domain})

        st.markdown(
            """
            <div class="output-box">
                <h2>Generated Content Plan</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(result)
