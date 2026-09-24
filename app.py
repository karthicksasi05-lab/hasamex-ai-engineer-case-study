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
    page_title="Hasamex — Expert Interview Dossier",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL STYLE — "case file" dossier theme
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #21242e;
        --ink-soft: #5b5f6e;
        --paper: #efe8d6;
        --paper-side: #e6dcc2;
        --paper-card: #faf7ee;
        --rule: #d8cca6;
        --rust: #9a3b2c;
        --brass: #a3822e;
        --forest: #3f5844;
    }

    .stApp {
        background-color: var(--paper);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.2px;
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background-color: var(--paper-side);
        border-right: 1px solid var(--rule);
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.02rem;
        margin-bottom: 0.4rem;
    }

    .cover {
        position: relative;
        background: var(--paper-card);
        border: 1px solid var(--rule);
        border-radius: 10px;
        padding: 1.9rem 2.2rem 1.6rem 2.2rem;
        margin-bottom: 0;
    }

    .cover-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-bottom: 0.35rem;
    }

    .cover h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }

    .cover p {
        font-size: 1rem;
        color: var(--ink-soft);
        max-width: 640px;
        margin: 0;
        line-height: 1.5;
    }

    .stamp {
        position: absolute;
        top: 1.6rem;
        right: 2rem;
        border: 2px solid var(--rust);
        color: var(--rust);
        border-radius: 6px;
        padding: 0.35rem 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        transform: rotate(6deg);
        opacity: 0.85;
    }

    .perforation {
        border: none;
        border-top: 1.5px dashed var(--rule);
        margin: 0;
    }

    .docket {
        display: flex;
        background: var(--paper-card);
        border: 1px solid var(--rule);
        border-top: none;
        border-radius: 0 0 10px 10px;
        margin-bottom: 1.6rem;
        overflow: hidden;
    }

    .docket-item {
        flex: 1;
        padding: 0.7rem 1.1rem;
        border-right: 1px solid var(--rule);
    }

    .docket-item:last-child {
        border-right: none;
    }

    .docket-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 500;
        color: var(--ink);
        line-height: 1.1;
    }

    .docket-label {
        font-size: 0.78rem;
        color: var(--ink-soft);
        margin-top: 0.2rem;
    }

    [data-baseweb="tab-list"] {
        gap: 3px;
        border-bottom: 2px solid var(--ink);
    }

    [data-baseweb="tab"] {
        background-color: var(--paper-side);
        border: 1px solid var(--rule);
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        color: var(--ink-soft);
        font-weight: 500;
    }

    [data-baseweb="tab"] p {
        font-size: 0.92rem;
    }

    [aria-selected="true"] {
        background-color: var(--paper-card) !important;
        color: var(--ink) !important;
        font-weight: 600;
    }

    .source-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        border: 1px solid var(--ink);
        border-radius: 3px;
        padding: 2px 9px;
        margin-bottom: 0.55rem;
        background: var(--paper);
        color: var(--ink);
    }

    .dossier-card {
        background: var(--paper-card);
        border: 1px solid var(--rule);
        border-left: 3px solid var(--brass);
        border-radius: 4px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.55rem;
    }

    .dossier-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: var(--rust);
        margin-bottom: 0.2rem;
    }

    .dossier-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--ink);
    }

    .dossier-keywords {
        margin-top: 0.35rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-soft);
    }

    .question-card {
        background: var(--paper-card);
        border: 1px solid var(--rule);
        border-left: 3px solid var(--ink);
        border-radius: 4px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
    }

    .question-index {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--brass);
        font-weight: 500;
        min-width: 1.4rem;
    }

    .question-text {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.02rem;
        color: var(--ink);
    }

    .verdict-chip {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        border: 1px solid var(--forest);
        color: var(--forest);
        border-radius: 3px;
        padding: 2px 9px;
        margin: 0 6px 6px 0;
    }

    .conclusion {
        background: var(--paper-card);
        border: 1px solid var(--rule);
        border-left: 3px solid var(--forest);
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }

    [data-testid="stExpander"] {
        background: var(--paper-card);
        border: 1px solid var(--rule) !important;
        border-radius: 6px;
    }

    hr {
        margin: 1.2rem 0;
        border-color: var(--rule);
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

    st.markdown(
        f"""
        <div class="dossier-card">
            <div class="dossier-meta">
                {item['timestamp']} — {item['speaker']}
            </div>
            <div class="dossier-text">
                {item['text']}
            </div>
            {keywords_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_source_tag(country_name):
    st.markdown(
        f'<span class="source-tag">SOURCE · {country_name}</span>',
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
# FILE COVER + KPI STRIP
# ============================================================

st.markdown(
    """
    <div class="cover">
        <div class="stamp">ON FILE</div>
        <div class="cover-eyebrow">Cross-market qualitative research</div>
        <h1>Expert Interview Dossier</h1>
        <p>
            A comparative reading of expert testimony collected in France,
            Germany and the United Kingdom, cross-referenced against the
            interview guide.
        </p>
    </div>
    <hr class="perforation" />
    """,
    unsafe_allow_html=True
)

status_value = "OK" if transcripts and questions else "Needs review"

st.markdown(
    f"""
    <div class="docket">
        <div class="docket-item">
            <div class="docket-value">
                {len(transcripts) if transcripts else 0}
            </div>
            <div class="docket-label">Transcripts on file</div>
        </div>

        <div class="docket-item">
            <div class="docket-value">{len(questions)}</div>
            <div class="docket-label">Guide questions</div>
        </div>

        <div class="docket-item">
            <div class="docket-value">
                {len(transcripts) if transcripts else 0}
            </div>
            <div class="docket-label">Countries covered</div>
        </div>

        <div class="docket-item">
            <div class="docket-value">{status_value}</div>
            <div class="docket-label">File status</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not transcripts:
    st.error("No transcript files found inside the data folder.")

if guide_error:
    st.error(guide_error)


# ============================================================
# SIDEBAR — CASE NAVIGATOR
# ============================================================

with st.sidebar:
    st.markdown("### 🗂️ Case navigator")

    st.markdown("**Transcripts**")

    if transcripts:
        for filename in transcripts:
            st.markdown(
                f"- {country_name_from_filename(filename)}"
            )
    else:
        st.error("None found")

    st.markdown("---")
    st.markdown("**Interview guide**")

    if questions:
        st.markdown(f"{len(questions)} question(s) parsed")
    else:
        st.warning("No questions detected")

    st.markdown("---")
    st.markdown("**AI pipeline**")
    st.markdown(
        "Semantic retrieval → grounded evidence → Gemini synthesis"
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
        "Guide",
        "Evidence",
        "Comparison",
        "Themes",
        "Ask",
        "Transcripts"
    ]
)


# ============================================================
# TAB 1 — INTERVIEW GUIDE QUESTIONS
# ============================================================

with tab_guide:
    st.subheader("Interview guide questions")

    if questions:
        for i, question in enumerate(questions, start=1):
            st.markdown(
                f"""
                <div class="question-card">
                    <span class="question-index">{i:02d}</span>
                    <span class="question-text">{question}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning(
            "No questions were detected in the Interview Guide."
        )


# ============================================================
# TAB 2 — EVIDENCE BY QUESTION
# ============================================================

with tab_evidence:
    st.subheader("Evidence by question")

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
    st.subheader("Cross-country comparison")

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
# TAB 4 — THEMES ACROSS EXPERT INTERVIEWS
# ============================================================

with tab_ask:
    st.subheader("Ask across all transcripts")

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

            st.subheader("🧠 Grounded AI Answer")

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

            st.subheader("🔎 Retrieved Evidence")

            st.caption(
                "These transcript statements were retrieved "
                "semantically and supplied to the AI as source evidence."
            )

            for country in ["France", "Germany", "UK"]:

                country_results = country_evidence[country]

                if not country_results:

                    st.markdown(
                        f"### 🌍 {country}"
                    )

                    st.info(
                        "No sufficiently relevant evidence was "
                        "retrieved for this country."
                    )

                    continue

                for index, result in enumerate(
                    country_results,
                    start=1
                ):

                    st.markdown(
                        f"### 🌍 {country}"
                    )

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
