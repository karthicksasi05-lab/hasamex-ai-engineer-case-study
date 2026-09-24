import streamlit as st

from src.semantic_retriever import (
    load_embedding_model,
    semantic_search
)
from src.llm_prompt import build_grounded_prompt
from src.llm_engine import generate_answer

from src.document_loader import load_transcripts, load_text_file
from src.transcript_parser import parse_transcript
from src.guide_parser import parse_interview_guide
from src.evidence_retriever import search_evidence
from src.comparison import (
    build_country_evidence,
    find_common_evidence,
    find_country_differences,
    find_contrasting_evidence
)
from src.theme_extractor import (
    build_cross_country_themes
)
from src.qa_engine import (
    QUESTION_KEYWORDS,
    identify_question_type
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hasamex — Expert Interview Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL STYLE — modern SaaS theme (light canvas, dark sidebar,
# indigo accent)
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f7f8fb;
        --surface: #ffffff;
        --surface-hover: #fafbff;
        --border: #e6e8f0;
        --border-strong: #d8dbe6;
        --ink: #111420;
        --ink-soft: #5b6072;
        --ink-faint: #9297a8;
        --accent: #4f46e5;
        --accent-hover: #4338ca;
        --accent-soft: #eeecfd;
        --accent-soft-border: #d9d5fb;
        --sidebar-bg: #0d0f1a;
        --sidebar-border: #1d2032;
        --sidebar-text: #c6c9db;
        --sidebar-text-dim: #74788e;
        --success: #12886b;
        --success-soft: #e4f5ef;
        --warn: #b5750a;
        --shadow-sm: 0 1px 2px rgba(17, 20, 32, 0.04);
        --shadow-md: 0 6px 20px rgba(17, 20, 32, 0.06), 0 1px 3px rgba(17, 20, 32, 0.04);
        --shadow-lg: 0 16px 40px rgba(17, 20, 32, 0.10);
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 18px;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.4px;
        color: var(--ink);
    }

    h2 {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }

    p, span, div, label {
        font-family: 'Inter', sans-serif;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--sidebar-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }

    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--sidebar-border);
    }

    [data-testid="stSidebar"] .stAlert {
        background: #1a1d2e;
        border: 1px solid var(--sidebar-border);
    }

    .sidebar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--sidebar-text-dim) !important;
        margin: 1.3rem 0 0.5rem 0;
    }

    .sidebar-transcript-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.89rem;
        color: var(--sidebar-text) !important;
        padding: 0.32rem 0;
    }

    .sidebar-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.18);
        flex-shrink: 0;
    }

    .sidebar-pipeline {
        font-size: 0.82rem;
        color: var(--sidebar-text) !important;
        line-height: 1.6;
        background: #161829;
        border: 1px solid var(--sidebar-border);
        border-radius: var(--radius-sm);
        padding: 0.7rem 0.85rem;
    }

    /* ---------- Cover header ---------- */

    .cover {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2.4rem;
        margin-bottom: 1.4rem;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
    }

    .cover-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 0.55rem;
    }

    .cover h1 {
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0 0 0.55rem 0;
        line-height: 1.15;
    }

    .cover p {
        font-size: 0.97rem;
        color: var(--ink-soft);
        max-width: 620px;
        margin: 0;
        line-height: 1.55;
    }

    .status-pill {
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: var(--success-soft);
        color: var(--success);
        border: 1px solid #c3e9dc;
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .status-pill .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
    }

    /* ---------- KPI cards ---------- */

    .docket {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.9rem;
    }

    .docket-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
    }

    .docket-item:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--border-strong);
    }

    .docket-icon {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: var(--accent-soft);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        margin-bottom: 0.65rem;
    }

    .docket-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.1;
        letter-spacing: -0.5px;
    }

    .docket-label {
        font-size: 0.79rem;
        font-weight: 500;
        color: var(--ink-faint);
        margin-top: 0.35rem;
    }

    /* ---------- Tabs ---------- */

    [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.3rem;
    }

    [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        padding: 10px 18px;
        color: var(--ink-faint);
        font-weight: 600;
        transition: background-color 0.15s ease, color 0.15s ease;
    }

    [data-baseweb="tab"]:hover {
        background-color: var(--surface-hover);
        color: var(--ink);
    }

    [data-baseweb="tab"] p {
        font-size: 0.9rem;
    }

    [aria-selected="true"] {
        background-color: var(--accent-soft) !important;
        color: var(--accent) !important;
        font-weight: 700;
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* ---------- Source / evidence elements ---------- */

    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.71rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 999px;
        padding: 4px 12px;
        margin-bottom: 0.6rem;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent-soft-border);
    }

    .dossier-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.8rem 1.05rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .dossier-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--border-strong);
    }

    .dossier-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        color: var(--accent);
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .dossier-text {
        font-size: 0.94rem;
        line-height: 1.58;
        color: var(--ink);
    }

    .dossier-keywords {
        margin-top: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-faint);
    }

    /* ---------- Guide questions ---------- */

    .question-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .question-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--accent-soft-border);
    }

    .question-index {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-weight: 700;
        min-width: 1.6rem;
        font-size: 0.85rem;
    }

    .question-text {
        font-weight: 600;
        font-size: 0.98rem;
        color: var(--ink);
    }

    /* ---------- Chips / verdicts ---------- */

    .verdict-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid #c3e9dc;
        background: var(--success-soft);
        color: var(--success);
        border-radius: 999px;
        padding: 4px 12px;
        margin: 0 6px 6px 0;
    }

    /* ---------- Conclusion / AI answer block ---------- */

    .conclusion {
        background: linear-gradient(180deg, #fbfaff 0%, var(--surface) 100%);
        border: 1px solid var(--accent-soft-border);
        border-radius: var(--radius-md);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-md);
        line-height: 1.65;
    }

    /* ---------- Expanders ---------- */

    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.6rem;
    }

    /* ---------- Section headers ---------- */

    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 1.12rem;
        color: var(--ink);
        margin: 0.4rem 0 0.8rem 0;
    }

    .country-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: var(--ink-soft);
        margin: 1rem 0 0.45rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border);
    }

    hr {
        margin: 1.3rem 0;
        border-color: var(--border);
    }

    /* ---------- Inputs ---------- */

    .stTextInput input {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-strong) !important;
        background: var(--surface) !important;
        padding: 0.65rem 0.9rem !important;
    }

    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_evidence_card(item):
    keywords_html = ""

    if item.get("keywords"):
        keywords_html = (
            f'<div class="dossier-keywords">Matched: '
            f'{", ".join(item["keywords"])}</div>'
        )

    card_html = (
        '<div class="dossier-card">'
        f'<div class="dossier-meta">{item["timestamp"]} — {item["speaker"]}</div>'
        f'<div class="dossier-text">{item["text"]}</div>'
        f'{keywords_html}'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def render_source_tag(country_name):
    st.markdown(
        f'<span class="source-tag">📍 {country_name}</span>',
        unsafe_allow_html=True
    )


def country_name_from_filename(filename):
    country = filename.replace("Transcript_", "").replace(".txt", "")

    # Convert filenames such as 1_France -> France
    if "_" in country and country.split("_", 1)[0].isdigit():
        country = country.split("_", 1)[1]

    return country


def prepare_all_semantic_evidence(question, transcripts, model, top_k_per_country=3):
    """
    Retrieve semantically relevant expert statements from every transcript.
    The result is sorted globally by semantic relevance.
    """

    all_results = []

    for filename, content in transcripts.items():
        records = parse_transcript(content)

        results = semantic_search(
            question,
            records,
            model,
            top_k=top_k_per_country
        )

        country = country_name_from_filename(filename)

        for result in results:
            all_results.append(
                {
                    "country": country,
                    "timestamp": result["timestamp"],
                    "speaker": result["speaker"],
                    "text": result["text"],
                    "score": result["score"]
                }
            )

    all_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return all_results


# ============================================================
# LOAD DATA
# ============================================================

transcripts = load_transcripts("data")

guide_path = "data/Interview_Guide.txt"
questions = []
guide_error = None

try:
    interview_guide = load_text_file(guide_path)
    questions = parse_interview_guide(interview_guide)
except FileNotFoundError:
    guide_error = "Interview_Guide.txt was not found inside the data folder."


# ============================================================
# LOAD EMBEDDING MODEL ONCE
# ============================================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


embedding_model = get_embedding_model()


# ============================================================
# HEADER + KPI STRIP
# ============================================================

status_ok = bool(transcripts and questions)
status_value = "All systems ready" if status_ok else "Needs review"
status_dot_color = "" if status_ok else "background:#b5750a;"

cover_html = (
    '<div class="cover"><div>'
    '<div class="cover-eyebrow">Cross-market qualitative research</div>'
    '<h1>Expert Interview Intelligence</h1>'
    '<p>A comparative reading of expert testimony collected in France, '
    'Germany and the United Kingdom, cross-referenced against the '
    'interview guide.</p>'
    '</div>'
    f'<div class="status-pill"><span class="dot" style="{status_dot_color}"></span>'
    f'{status_value}</div>'
    '</div>'
)

st.markdown(cover_html, unsafe_allow_html=True)

transcript_count = len(transcripts) if transcripts else 0

docket_items = [
    ("🗂️", str(transcript_count), "Transcripts on file"),
    ("📋", str(len(questions)), "Guide questions"),
    ("🌍", str(transcript_count), "Countries covered"),
    ("⚡", "OK" if status_ok else "⚠", "File status"),
]

# Built as a single-line string on purpose: st.markdown() renders
# through a standard Markdown parser first, and any HTML line
# indented 4+ spaces gets misread as a fenced code block instead
# of being passed through as HTML.
docket_html = '<div class="docket">' + "".join(
    f'<div class="docket-item"><div class="docket-icon">{icon}</div>'
    f'<div class="docket-value">{value}</div>'
    f'<div class="docket-label">{label}</div></div>'
    for icon, value, label in docket_items
) + '</div>'

st.markdown(docket_html, unsafe_allow_html=True)

if not transcripts:
    st.error("No transcript files found inside the data folder.")

if guide_error:
    st.error(guide_error)


# ============================================================
# SIDEBAR — CASE NAVIGATOR
# ============================================================

with st.sidebar:
    st.markdown("### ◆ Case navigator")

    st.markdown('<div class="sidebar-label">Transcripts</div>', unsafe_allow_html=True)

    if transcripts:
        for filename in transcripts:
            transcript_item_html = (
                '<div class="sidebar-transcript-item">'
                '<span class="sidebar-dot"></span>'
                f'{country_name_from_filename(filename)}'
                '</div>'
            )
            st.markdown(transcript_item_html, unsafe_allow_html=True)
    else:
        st.error("None found")

    st.markdown('<div class="sidebar-label">Interview guide</div>', unsafe_allow_html=True)

    if questions:
        st.markdown(f"{len(questions)} question(s) parsed")
    else:
        st.warning("No questions detected")

    st.markdown('<div class="sidebar-label">AI pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-pipeline">Semantic retrieval → grounded evidence → Gemini synthesis</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption(
        "Work through the tabs to review evidence by question, "
        "compare countries, trace recurring themes, or ask a "
        "free-form question across every transcript."
    )


# ============================================================
# TABS
# ============================================================

tab_guide, tab_evidence, tab_compare, tab_themes, tab_ask, tab_transcripts = st.tabs(
    [
        "📋 Guide",
        "🔎 Evidence",
        "⚖️ Comparison",
        "🧩 Themes",
        "💬 Ask",
        "📄 Transcripts"
    ]
)


# ============================================================
# TAB 1 — INTERVIEW GUIDE QUESTIONS
# ============================================================

with tab_guide:
    st.markdown('<div class="section-heading">Interview guide questions</div>', unsafe_allow_html=True)

    if questions:
        for i, question in enumerate(questions, start=1):
            question_card_html = (
                '<div class="question-card">'
                f'<span class="question-index">{i:02d}</span>'
                f'<span class="question-text">{question}</span>'
                '</div>'
            )
            st.markdown(question_card_html, unsafe_allow_html=True)
    else:
        st.warning(
            "No questions were detected in the Interview Guide."
        )


# ============================================================
# TAB 2 — EVIDENCE BY QUESTION
# ============================================================

with tab_evidence:
    st.markdown('<div class="section-heading">Evidence by question</div>', unsafe_allow_html=True)

    if not questions:
        st.info(
            "No questions available to search evidence for."
        )

    for i, question in enumerate(questions, start=1):

        with st.expander(
            f"{i}. {question}",
            expanded=False
        ):

            question_type = identify_question_type(question)
            keywords = QUESTION_KEYWORDS.get(
                question_type,
                []
            )

            found_any = False

            for filename, content in transcripts.items():

                records = parse_transcript(content)

                evidence = search_evidence(
                    records,
                    keywords
                )

                if evidence:
                    found_any = True

                    render_source_tag(
                        country_name_from_filename(filename)
                    )

                    for item in evidence:
                        render_evidence_card(item)

            if not found_any:
                st.info(
                    "No supporting evidence found in the transcripts."
                )


# ============================================================
# TAB 3 — CROSS-COUNTRY COMPARISON
# ============================================================

with tab_compare:
    st.markdown('<div class="section-heading">Cross-country comparison</div>', unsafe_allow_html=True)

    for i, question in enumerate(questions, start=1):

        with st.expander(
            f"{i}. {question}",
            expanded=False
        ):

            question_type = identify_question_type(question)

            keywords = QUESTION_KEYWORDS.get(
                question_type,
                []
            )

            country_evidence = build_country_evidence(
                transcripts,
                parse_transcript,
                search_evidence,
                keywords
            )

            # ---- Common topics ----

            common_keywords = find_common_evidence(
                country_evidence
            )

            if common_keywords:
                st.markdown(
                    "**Common topics identified across interviews**"
                )

                st.markdown(
                    "".join(
                        f'<span class="verdict-chip">{kw}</span>'
                        for kw in common_keywords
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.info(
                    "No common keyword themes were identified."
                )

            st.markdown("")

            # ---- Country-by-country evidence ----

            st.markdown("**Country perspectives**")

            country_cols = st.columns(
                len(country_evidence) or 1
            )

            for col, (
                country,
                evidence
            ) in zip(
                country_cols,
                country_evidence.items()
            ):

                with col:
                    render_source_tag(country)

                    if evidence:
                        for item in evidence:
                            render_evidence_card(item)
                    else:
                        st.info(
                            "No supporting evidence found."
                        )

            # ---- Direct evidence differences ----

            differences = find_country_differences(
                country_evidence
            )

            has_diffs = any(
                evidence
                for evidence in differences.values()
            )

            st.markdown("")

            if has_diffs:
                st.markdown("**Contrasting evidence**")

                contrasting_evidence = find_contrasting_evidence(
                    country_evidence
                )

                available_countries = [
                    country
                    for country, evidence
                    in contrasting_evidence.items()
                    if evidence
                ]

                if len(available_countries) >= 2:

                    st.info(
                        "The evidence below shows how expert "
                        "perspectives differ across countries. "
                        "Differences are presented directly from "
                        "the transcripts without generating "
                        "unsupported conclusions."
                    )

                    for country, evidence in contrasting_evidence.items():

                        if evidence:

                            with st.expander(
                                f"🌍 {country}"
                            ):

                                for item in evidence:

                                    st.markdown(
                                        f"**{item['timestamp']} — "
                                        f"{item['speaker']}**"
                                    )

                                    st.write(
                                        f"> {item['text']}"
                                    )
                else:
                    st.info(
                        "Not enough country evidence is available "
                        "to compare perspectives."
                    )
            else:
                st.info(
                    "No country-specific evidence was found "
                    "for this question."
                )


# ============================================================
# TAB 4 — CROSS-COUNTRY THEMES
# ============================================================

with tab_themes:
    st.markdown('<div class="section-heading">Cross-country themes</div>', unsafe_allow_html=True)

    st.caption(
        "Recurring themes detected automatically across every transcript "
        "by keyword matching. All evidence shown is verbatim from the "
        "transcripts — no conclusions are generated here."
    )

    cross_country_themes = build_cross_country_themes(
        transcripts,
        parse_transcript
    )

    total_countries = len(transcripts) if transcripts else 0

    if not cross_country_themes:
        st.info("No themes could be detected across the transcripts.")
    else:
        sorted_themes = sorted(
            cross_country_themes.items(),
            key=lambda item: item[1]["country_count"],
            reverse=True
        )

        for theme, data in sorted_themes:

            country_count = data["country_count"]

            with st.expander(
                f"{theme}  ·  present in {country_count}/{total_countries} countries",
                expanded=False
            ):

                if total_countries and country_count == total_countries:
                    st.markdown(
                        '<span class="verdict-chip">✓ Present in every country</span>',
                        unsafe_allow_html=True
                    )

                for country, evidence in data["countries"].items():

                    render_source_tag(country)

                    for item in evidence:
                        render_evidence_card(item)


# ============================================================
# TAB 5 — ASK ACROSS ALL TRANSCRIPTS
# ============================================================

with tab_ask:
    st.markdown('<div class="section-heading">Ask across all transcripts</div>', unsafe_allow_html=True)

    st.caption(
        "Ask a natural-language question. The system first retrieves "
        "relevant transcript evidence using semantic similarity, then "
        "asks Gemini to synthesize an answer using only that evidence."
    )

    user_question = st.text_input(
        "Ask a question about the expert interviews:",
        key="transcript_question",
        placeholder=(
            "e.g. What are the main barriers to adoption in Germany?"
        )
    )

    if user_question:

        st.markdown("---")

        with st.spinner(
            "Retrieving relevant evidence from all transcripts..."
        ):

            all_results = prepare_all_semantic_evidence(
                user_question,
                transcripts,
                embedding_model,
                top_k_per_country=3
            )

        # ------------------------------------------------
        # GROUP RETRIEVED EVIDENCE BY COUNTRY
        # ------------------------------------------------

        country_evidence = {
            "France": [],
            "Germany": [],
            "UK": []
        }

        for result in all_results:
            country = result["country"]

            if country in country_evidence:
                country_evidence[country].append(result)

        # ------------------------------------------------
        # CHECK WHETHER ANY EVIDENCE WAS FOUND
        # ------------------------------------------------

        if not all_results:

            st.warning(
                "No sufficiently relevant evidence was retrieved "
                "from the transcripts."
            )

        else:

            # ------------------------------------------------
            # GROUNDED LLM ANSWER
            # ------------------------------------------------

            st.markdown('<div class="section-heading">🧠 Grounded AI answer</div>', unsafe_allow_html=True)

            # Keep the strongest evidence while preserving
            # country coverage.
            llm_evidence = []

            for country in ["France", "Germany", "UK"]:

                country_results = country_evidence[country]

                # Take up to 3 strongest results for each country
                llm_evidence.extend(
                    country_results[:3]
                )

            grounded_prompt = build_grounded_prompt(
                user_question,
                llm_evidence
            )

            try:

                with st.spinner(
                    "Generating a grounded answer from the evidence..."
                ):

                    answer = generate_answer(
                        grounded_prompt
                    )

                st.markdown(
                    '<div class="conclusion">',
                    unsafe_allow_html=True
                )

                st.markdown(answer)

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            except RuntimeError as error:

                error_message = str(error)

                if "rate limit" in error_message.lower():

                    st.warning(
                        "⚠️ Gemini AI generation is temporarily unavailable "
                        "because the Free Tier request limit has been reached."
                    )

                    st.info(
                        "The semantic retrieval pipeline is still working. "
                        "You can review the retrieved transcript evidence below."
                    )

                else:

                    st.error(
                        "The grounded AI answer could not be generated."
                    )

                    st.code(
                        error_message
                    )

            except Exception as error:

                st.error(
                    "An unexpected error occurred while generating the AI answer."
                )

                st.code(
                    str(error)
                )

                st.info(
                    "Check that GEMINI_API_KEY is available in "
                    "your environment and that the Gemini model "
                    "configured in src/llm_engine.py is accessible."
                )

            # ------------------------------------------------
            # RETRIEVED EVIDENCE
            # ------------------------------------------------

            st.markdown('<div class="section-heading">🔎 Retrieved evidence</div>', unsafe_allow_html=True)

            st.caption(
                "These transcript statements were retrieved "
                "semantically and supplied to the AI as source evidence."
            )

            for country in ["France", "Germany", "UK"]:

                country_results = country_evidence[country]

                st.markdown(
                    f'<div class="country-heading">🌍 {country}</div>',
                    unsafe_allow_html=True
                )

                if not country_results:
                    st.info(
                        "No sufficiently relevant evidence was "
                        "retrieved for this country."
                    )
                    continue

                for index, result in enumerate(
                    country_results,
                    start=1
                ):

                    st.write(
                        f"**{result['timestamp']} — "
                        f"{result['speaker']}**"
                    )

                    st.write(
                        result["text"]
                    )

                    st.caption(
                        f"Semantic relevance: "
                        f"{result['score']:.3f}"
                    )

                    st.divider()


# ============================================================
# TAB 6 — RAW TRANSCRIPTS
# ============================================================

with tab_transcripts:
    st.markdown('<div class="section-heading">Full transcripts</div>', unsafe_allow_html=True)

    st.caption(
        "Browse the complete, parsed transcript for each country. "
        "Use this to read statements in their original context."
    )

    if not transcripts:
        st.info("No transcript files found inside the data folder.")

    for filename, content in transcripts.items():

        country = country_name_from_filename(filename)
        records = parse_transcript(content)

        with st.expander(f"🌍 {country} — {len(records)} statement(s)", expanded=False):

            if not records:
                st.info("No statements could be parsed from this transcript.")
                continue

            for record in records:
                record_html = (
                    '<div class="dossier-card">'
                    f'<div class="dossier-meta">{record["timestamp"]} — {record["speaker"]}</div>'
                    f'<div class="dossier-text">{record["text"]}</div>'
                    '</div>'
                )
                st.markdown(record_html, unsafe_allow_html=True)
