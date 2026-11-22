from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

llm = ChatOpenAI(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)


# 1st agent - search
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
        system_prompt = """

        OUTPUT FORMATTING RULES:

1. Return responses using Markdown only.
2. Never generate HTML tags.
3. Do not use <div>, <span>, <p>, or other HTML elements.
4. Use clear headings and bullet points.
5. Include source titles and complete URLs.
6. Do not invent URLs or sources.
7. Keep the search results relevant to the user's research topic.

"""
    )

# 2nd agent - return agent
def build_return_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt = """
        OUTPUT FORMATTING RULES:

1. Return scraped information using Markdown only.
2. Never generate HTML tags.
3. Extract the important information from the source.
4. Organize the content using headings and bullet points.
5. Include the original source URL.
6. Do not invent missing information.
7. If the URL cannot be accessed, clearly explain why.
8. Do not ask the user for URLs if the search results already contain them.

"""
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
