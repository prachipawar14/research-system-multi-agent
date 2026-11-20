import html
import re
import time

import streamlit as st

from src.agents.agents import (
    build_search_agent,
    build_return_agent,
    writer_chain,
    critic_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_ai_output(content):
    """
    Convert agent output to readable Markdown.

    Removes HTML tags that an agent may accidentally return, while keeping
    Markdown syntax such as **bold**, headings, links and bullet points.
    """
    if content is None:
        return ""

    # Handle LangChain message objects as well as plain strings.
    if hasattr(content, "content"):
        content = content.content

    if not isinstance(content, str):
        content = str(content)

    # Turn common HTML layout tags into line breaks before removing tags.
    content = re.sub(
        r"(?i)<\s*(br|/p|/div|/section|/article|/li|/h[1-6])\s*/?\s*>",
        "\n",
        content,
    )
    content = re.sub(r"(?i)<\s*li\s*>", "- ", content)

    # Remove remaining HTML tags and decode entities such as &amp; and &nbsp;.
    content = re.sub(r"<[^>]*>", "", content)
    content = html.unescape(content)

    # Remove excessive blank lines.
    content = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", content)

    return content.strip()


def get_last_message_content(agent_result):
    """Extract readable text from the agent response."""
    messages = agent_result.get("messages", [])

    # Check messages from newest to oldest
    for message in reversed(messages):
        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            return clean_ai_output(content)

        # Handle content returned as a list of blocks
        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                else:
                    text = getattr(block, "text", "")

                if text:
                    text_parts.append(text)

            if text_parts:
                return clean_ai_output("\n".join(text_parts))

    return "No readable text was returned by the Search Agent."


def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done": ("✓ DONE", "status-done"),
        "error": ("✕ ERROR", "status-error"),
    }

    label, status_class = status_map.get(state, ("WAITING", "status-waiting"))
    card_class = {
        "running": "active",
        "done": "done",
        "error": "error",
    }.get(state, "")

    st.markdown(
        f"""
        <div class="step-card {card_class}">
            <div class="step-header">
                <span class="step-num">{num}</span>
                <span class="step-title">{title}</span>
                <span class="step-status {status_class}">{label}</span>
            </div>
            <div class="step-description">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #07111f;
        background-image:
            radial-gradient(circle at top left, rgba(0,191,255,0.14), transparent 32%),
            radial-gradient(circle at bottom right, rgba(124,58,237,0.12), transparent 30%),
            linear-gradient(180deg, #07111f 0%, #0a1729 100%);
        color: #edf3ff;
    }

    .block-container {
        padding: 2rem 3rem 4rem;
        max-width: 1200px;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 3.5rem 0 2.5rem;
    }

    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #38bdf8;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.8rem, 6vw, 5rem);
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
        color: #f8fbff;
        margin: 0 0 1rem;
    }

    .hero h1 span {
        background: linear-gradient(135deg, #38bdf8, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: 1.05rem;
        font-weight: 300;
        color: #b5c3d9;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.65;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
        margin: 2rem 0;
    }

    .section-heading {
        font-family: 'Syne', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fbff;
        margin: 1.5rem 0 1rem;
    }

    /* Input */
    .input-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 22px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.28);
    }

    .stTextInput > label {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #38bdf8 !important;
        font-weight: 500 !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        border-radius: 12px !important;
        color: #f8fbff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 4px rgba(56,189,248,0.14) !important;
    }

    .stTextInput input::placeholder {
        color: #91a4bd !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.2rem !important;
        box-shadow: 0 8px 30px rgba(56,189,248,0.22) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(56,189,248,0.32) !important;
    }

    .stDownloadButton > button {
        background: rgba(56,189,248,0.12) !important;
        color: #dff5ff !important;
        border: 1px solid rgba(56,189,248,0.35) !important;
        border-radius: 10px !important;
    }

    /* Pipeline cards */
    .step-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }

    .step-card.active {
        border-color: rgba(56,189,248,0.5);
        background: rgba(56,189,248,0.07);
    }

    .step-card.done {
        border-color: rgba(34,197,94,0.3);
        background: rgba(34,197,94,0.05);
    }

    .step-card.error {
        border-color: rgba(248,113,113,0.4);
        background: rgba(248,113,113,0.06);
    }

    .step-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .step-num {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.15em;
        color: #38bdf8;
    }

    .step-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #f8fbff;
    }

    .step-status {
        margin-left: auto;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
    }

    .status-waiting { color: #94a3b8; }
    .status-running { color: #38bdf8; }
    .status-done { color: #22c55e; }
    .status-error { color: #f87171; }

    .step-description {
        color: #aabbd0;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }

    /* Markdown and result readability */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown strong,
    .stMarkdown blockquote {
        color: #edf3ff;
    }

    .stMarkdown a {
        color: #38bdf8 !important;
    }

    .stMarkdown code {
        color: #7dd3fc;
        background: rgba(255,255,255,0.08);
        border-radius: 5px;
        padding: 0.1rem 0.3rem;
    }

    .stMarkdown pre {
        background: #0b1b2e !important;
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 10px;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
    }

    [data-testid="stExpander"] summary {
        color: #d8e2f0 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .stSpinner > div {
        color: #38bdf8 !important;
    }

    .notice {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #7f93ad;
        text-align: center;
        margin-top: 3rem;
        letter-spacing: 0.08em;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 3rem;
        }

        .hero {
            padding: 2rem 0 1.5rem;
        }
    }
    /* FIX: Make search input text visible */
[data-testid="stTextInput"] input,
.stTextInput input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #ffffff !important;
    caret-color: #111827 !important;
    opacity: 1 !important;
}

/* Placeholder text */
[data-testid="stTextInput"] input::placeholder,
.stTextInput input::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    opacity: 1 !important;
}

/* Text area, if used */
[data-testid="stTextArea"] textarea,
.stTextArea textarea {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #ffffff !important;
    caret-color: #111827 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = {}

if "running" not in st.session_state:
    st.session_state.running = False

if "done" not in st.session_state:
    st.session_state.done = False

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None

if "step_states" not in st.session_state:
    st.session_state.step_states = {
        "search": "waiting",
        "reader": "waiting",
        "writer": "waiting",
        "critic": "waiting",
    }


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Multi-Agent AI System</div>
        <h1>Researcher<span>Agent</span></h1>
        <p class="hero-sub">
            Four specialized AI agents collaborate — searching, scraping,
            writing, and critiquing — to deliver a polished research report
            on any topic.
        </p>
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Input and pipeline layout
# ---------------------------------------------------------------------------

col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Roadmap for AGI development in next 5 years",
        key="topic_input",
        label_visibility="visible",
    )

    run_btn = st.button(
        "⚡ Run Research Pipeline",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.5rem 0 1.5rem;">
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#7f93ad;letter-spacing:0.1em;">
                TRY →
            </span>
            <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:0.35rem 0.8rem;font-size:0.75rem;color:#d8e2f0;">
                Future of LLMs in Tech Industry
            </span>
            <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:0.35rem 0.8rem;font-size:0.75rem;color:#d8e2f0;">
                AI Agents in 2026
            </span>
            <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:0.35rem 0.8rem;font-size:0.75rem;color:#d8e2f0;">
                Roadmap for AGI
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_pipeline:
    st.markdown(
        '<div class="section-heading">Pipeline</div>',
        unsafe_allow_html=True,
    )

    states = st.session_state.step_states

    step_card(
        "01",
        "Search Agent",
        states["search"],
        "Gathers recent web information",
    )
    step_card(
        "02",
        "Reader Agent",
        states["reader"],
        "Scrapes and extracts deeper content",
    )
    step_card(
        "03",
        "Writer Chain",
        states["writer"],
        "Drafts the full research report",
    )
    step_card(
        "04",
        "Critic Chain",
        states["critic"],
        "Reviews and scores the report",
    )


# ---------------------------------------------------------------------------
# Start pipeline
# ---------------------------------------------------------------------------

if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.pipeline_error = None
        st.session_state.step_states = {
            "search": "waiting",
            "reader": "waiting",
            "writer": "waiting",
            "critic": "waiting",
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Execute pipeline
# ---------------------------------------------------------------------------

if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_input.strip()

    try:
        # Step 1: Search Agent
        st.session_state.step_states["search"] = "running"
        with st.spinner("🔍 Search Agent is working..."):
            search_agent = build_search_agent()

            search_result = search_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            (
                                "Find recent, reliable and detailed information "
                                f"about: {topic_val}. Return source titles, "
                                "complete URLs and useful summaries. Use Markdown, "
                                "not HTML."
                            ),
                        )
                    ]
                }
            )

            results["search"] = get_last_message_content(search_result)
            st.session_state.results = dict(results)

        st.session_state.step_states["search"] = "done"

        # Step 2: Reader Agent
        st.session_state.step_states["reader"] = "running"
        with st.spinner("📄 Reader Agent is reading the best sources..."):
            reader_agent = build_return_agent()

            reader_result = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            (
                                f"Research topic: {topic_val}\n\n"
                                "Use the URLs included in these search results. "
                                "Choose the most relevant accessible source and "
                                "extract its important information. Do not ask "
                                "the user to provide URLs when URLs are already "
                                "present. If a source cannot be accessed, say so. "
                                "Return Markdown only, never HTML.\n\n"
                                f"Search Results:\n{results['search'][:8000]}"
                            ),
                        )
                    ]
                }
            )

            results["reader"] = get_last_message_content(reader_result)
            st.session_state.results = dict(results)

        st.session_state.step_states["reader"] = "done"

        # Step 3: Writer Chain
        st.session_state.step_states["writer"] = "running"
        with st.spinner("✍️ Writer is drafting the research report..."):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED READER CONTENT:\n{results['reader']}"
            )

            writer_result = writer_chain.invoke(
                {
                    "topic": topic_val,
                    "research": research_combined,
                }
            )

            results["writer"] = clean_ai_output(writer_result)
            st.session_state.results = dict(results)

        st.session_state.step_states["writer"] = "done"

        # Step 4: Critic Chain
        st.session_state.step_states["critic"] = "running"
        with st.spinner("🧐 Critic is reviewing the report..."):
            critic_result = critic_chain.invoke(
                {
                    "report": results["writer"],
                }
            )

            results["critic"] = clean_ai_output(critic_result)
            st.session_state.results = dict(results)

        st.session_state.step_states["critic"] = "done"
        st.session_state.running = False
        st.session_state.done = True

    except Exception as exc:
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.pipeline_error = str(exc)

        # Mark the currently unfinished step as failed.
        for step_name, step_state in st.session_state.step_states.items():
            if step_state == "running":
                st.session_state.step_states[step_name] = "error"
                break

    st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

results = st.session_state.results

if st.session_state.pipeline_error:
    st.error(
        "The research pipeline stopped because an error occurred. "
        "Check the terminal for the full traceback."
    )
    with st.expander("Show error details"):
        st.code(st.session_state.pipeline_error)

if results:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-heading">Research Results</div>',
        unsafe_allow_html=True,
    )

    if "search" in results:
        with st.expander("🔍 Search Results", expanded=True):
            st.subheader("Search Agent Output")

        search_text = clean_ai_output(results["search"])

        st.text_area(
            "Search Engine Response",
            value=search_text,
            height=400,
            disabled=True,
            key="search_output"
        )
    if "reader" in results:
        with st.expander("📄 Scraped Content", expanded=False):
            st.markdown("### Reader Agent Output")
            st.markdown(clean_ai_output(results["reader"]))

    if "writer" in results:
        with st.container(border=True):
            st.markdown("## 📝 Final Research Report")
            st.markdown(clean_ai_output(results["writer"]))

        st.download_button(
            label="⬇ Download Report (.md)",
            data=clean_ai_output(results["writer"]),
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in results:
        with st.container(border=True):
            st.markdown("## 🧐 Critic Feedback")
            st.markdown(clean_ai_output(results["critic"]))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="notice">
        ResearchAgent · Powered by LangChain multi-agent pipeline · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
