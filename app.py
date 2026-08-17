import streamlit as st
import textwrap

from rag import (
    create_embeddings,
    get_vector_database,
    create_llm,
    rewrite_question,
    retrieve_documents,
    generate_answer,
)


st.set_page_config(
    page_title="TechNova | NovaAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html):
    st.html(textwrap.dedent(html))


render_html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 55% -15%,
                rgba(87, 72, 255, .13),
                transparent 30%
            ),
            #08090d;
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.15rem;
        padding-bottom: 7rem;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: #0d0f15;
        border-right: 1px solid #20232d;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: .2rem 0 1.35rem;
    }

    .sidebar-logo {
        width: 36px;
        height: 36px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #6846ff,
            #347dff
        );
        color: white;
        font-weight: 800;
        box-shadow:
            0 8px 24px rgba(77, 82, 255, .22);
    }

    .sidebar-name {
        font-size: 1.05rem;
        font-weight: 760;
        color: #f4f6fb;
    }

    .sidebar-tagline {
        color: #70788a;
        font-size: .68rem;
        margin-top: 1px;
    }

    .sidebar-section {
        color: #70798d;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 1.35rem 0 .5rem;
    }

    .topbar {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: .15rem 0 1rem;
        border-bottom: 1px solid #20232d;
        margin-bottom: 1.35rem;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #6846ff,
            #347dff
        );
        color: white;
        font-weight: 800;
        box-shadow:
            0 8px 26px rgba(77, 82, 255, .22);
    }

    .brand-name {
        font-size: 1.08rem;
        font-weight: 760;
        color: #f4f6fb;
    }

    .brand-subtitle {
        color: #70788a;
        font-size: .7rem;
        margin-top: 1px;
    }

    .stButton > button {
        border: 1px solid #292e3b;
        background: #11141b;
        color: #dfe4ee;
        border-radius: 10px;
        min-height: 2.5rem;
        transition: all .16s ease;
    }

    .stButton > button:hover {
        border-color: #5865a5;
        background: #151925;
        color: #ffffff;
    }

    .hero {
        text-align: center;
        padding: 2.8rem 1rem 2.5rem;
    }

    .hero-badge {
        display: inline-block;
        padding: .42rem .78rem;
        border: 1px solid #30354a;
        border-radius: 999px;
        color: #aeb9ff;
        background: rgba(80, 75, 160, .10);
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 1.05;
        letter-spacing: -.045em;
        margin: 1.1rem 0 .8rem;
        font-weight: 830;
        color: #f5f7fb;
    }

    .gradient-text {
        background: linear-gradient(
            90deg,
            #bca7ff,
            #72a5ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-description {
        max-width: 720px;
        margin: 0 auto;
        color: #9299a9;
        font-size: 1rem;
        line-height: 1.7;
    }

    .section-label {
        color: #8793ff;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    .section-title {
        font-size: 1.65rem;
        font-weight: 760;
        margin-top: .35rem;
        color: #f4f6fb;
    }

    .section-description {
        color: #858d9f;
        line-height: 1.65;
        max-width: 720px;
        margin-bottom: 1.1rem;
    }

    .product-card {
        min-height: 210px;
        padding: 1.35rem;
        border: 1px solid #242936;
        border-radius: 18px;
        background:
            linear-gradient(
                145deg,
                rgba(23, 26, 35, .98),
                rgba(14, 16, 22, .98)
            );
        box-shadow:
            0 18px 45px rgba(0, 0, 0, .14);
        margin-bottom: 1rem;
    }

    .product-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #171a24;
        border: 1px solid #2a2f3d;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .product-title {
        font-size: 1.15rem;
        font-weight: 760;
        margin-bottom: .2rem;
        color: #f4f6fb;
    }

    .product-subtitle {
        color: #8793ff;
        font-size: .67rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }

    .product-description {
        color: #8f96a7;
        line-height: 1.65;
        font-size: .9rem;
    }

    .assistant-shell {
        max-width: 920px;
        margin: 0 auto;
    }

    .assistant-header {
        text-align: center;
        padding: 1.2rem 0 1rem;
    }

    .assistant-icon {
        width: 54px;
        height: 54px;
        margin: 0 auto .8rem;
        border-radius: 17px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            linear-gradient(
                135deg,
                #6846ff,
                #347dff
            );
        color: white;
        font-size: 1.35rem;
        box-shadow:
            0 12px 32px rgba(77, 82, 255, .24);
    }

    .assistant-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -.03em;
        color: #f5f7fb;
    }

    .assistant-description {
        color: #8c94a5;
        margin-top: .4rem;
        line-height: 1.55;
    }

    .kb-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: .38rem .72rem;
        margin-top: .8rem;
        border: 1px solid #252a38;
        border-radius: 999px;
        background: #10131a;
        color: #7f8799;
        font-size: .7rem;
    }

    .kb-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #65d49a;
        box-shadow:
            0 0 10px rgba(101, 212, 154, .45);
    }

    .welcome-box {
        margin: 1.2rem auto 1rem;
        padding: 1.35rem;
        border: 1px solid #252a38;
        border-radius: 18px;
        background: rgba(15, 18, 25, .78);
    }

    .welcome-title {
        font-size: 1.05rem;
        font-weight: 720;
        color: #f4f6fb;
    }

    .welcome-text {
        color: #858d9e;
        font-size: .9rem;
        line-height: 1.6;
        margin-top: .35rem;
    }

    .followup-title {
        color: #7e8798;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin: .8rem 0 .45rem;
    }

    div[data-testid="stChatMessage"] {
        border: 0 !important;
        background: transparent !important;
        padding: .55rem 0 !important;
    }

    div[data-testid="stChatMessageContent"] {
        border-radius: 16px;
        padding: .9rem 1rem;
    }

    .source-page {
        padding: .55rem 0;
        border-bottom: 1px solid #222733;
    }

    .source-page:last-child {
        border-bottom: 0;
    }

    .source-page-title {
        color: #dce1ec;
        font-size: .82rem;
        font-weight: 650;
    }

    .source-page-file {
        color: #747d8e;
        font-size: .72rem;
        margin-top: .18rem;
    }

    .app-footer {
        text-align: center;
        color: #555d6d;
        font-size: .7rem;
        padding: 2.2rem 0 .5rem;
    }

    @media (max-width: 760px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding-top: 1.5rem;
        }

        .hero-title {
            font-size: 2.25rem;
        }

    }

    </style>
    """
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


@st.cache_resource
def load_rag_components():

    embeddings = create_embeddings()

    index, chunks = get_vector_database(
        embeddings
    )

    llm = create_llm()

    return embeddings, index, chunks, llm


with st.spinner("Initializing NovaAI..."):

    embeddings, index, chunks, llm = (
        load_rag_components()
    )


def clear_chat():

    st.session_state.messages = []

    st.session_state.pending_question = None


def ask_question(question):

    st.session_state.pending_question = question


def go_to_assistant(question=None):

    if question:
        st.session_state.pending_question = question

    st.query_params["page"] = "assistant"

    st.rerun()


def unique_sources(results):

    sources = []
    seen = set()

    for result in results:

        page = result.metadata.get("page")

        source = result.metadata.get(
            "source",
            "Company knowledge base",
        )

        key = (
            page,
            source,
        )

        if key not in seen:

            seen.add(key)

            sources.append(
                {
                    "page": page,
                    "source": source,
                }
            )

    return sources


def get_followups(answer):

    text = answer.lower()

    if "novaai" in text:

        return [
            "What does NovaAI provide?",
            "What is NovaFlow?",
            "Who founded TechNova?",
        ]

    if "novaflow" in text:

        return [
            "What is NovaAI?",
            "What does NovaFlow provide?",
            "Who founded TechNova?",
        ]

    return [
        "What is NovaAI?",
        "What is NovaFlow?",
        "Where is TechNova headquartered?",
    ]


def process_question(query):

    chat_history = [
        {
            "question": message["question"],
            "answer": message["answer"],
        }
        for message in st.session_state.messages
    ]

    standalone_query = rewrite_question(
        llm,
        query,
        chat_history,
    )

    results = retrieve_documents(
        index,
        chunks,
        embeddings,
        standalone_query,
    )

    answer = generate_answer(
        llm,
        standalone_query,
        results,
    )

    return answer, results


def render_sources(sources):

    if not sources:
        return

    with st.expander(
        f"Sources · {len(sources)}",
        expanded=False,
    ):

        for source in sources:

            render_html(
                f"""
                <div class="source-page">

                    <div class="source-page-title">
                        Page {source["page"]}
                    </div>

                    <div class="source-page-file">
                        {source["source"]}
                    </div>

                </div>
                """
            )


def render_followups(
    followups,
    key_prefix,
):

    if not followups:
        return

    render_html(
        """
        <div class="followup-title">
            You might also ask
        </div>
        """
    )

    cols = st.columns(
        len(followups)
    )

    for i, (
        col,
        suggestion,
    ) in enumerate(
        zip(
            cols,
            followups,
        )
    ):

        with col:

            if st.button(
                suggestion,
                key=f"{key_prefix}_{i}",
                use_container_width=True,
            ):

                ask_question(
                    suggestion
                )

                st.rerun()


with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                T
            </div>

            <div>

                <div class="sidebar-name">
                    TechNova
                </div>

                <div class="sidebar-tagline">
                    Intelligent technology
                </div>

            </div>

        </div>
        """
    )

    if st.button(
        "＋  New conversation",
        use_container_width=True,
    ):

        clear_chat()

        st.query_params["page"] = "assistant"

        st.rerun()

    render_html(
        """
        <div class="sidebar-section">
            Explore
        </div>
        """
    )

    explore_items = [

        (
            "✦  NovaAI",
            "What is NovaAI?",
        ),

        (
            "⚡  NovaFlow",
            "What is NovaFlow?",
        ),

        (
            "◎  Company",
            "Tell me about TechNova.",
        ),

        (
            "♙  Leadership",
            "Who is the founder and CEO of TechNova?",
        ),

    ]

    for label, question in explore_items:

        if st.button(
            label,
            key=f"explore_{label}",
            use_container_width=True,
        ):

            go_to_assistant(
                question
            )

    render_html(
        """
        <div class="sidebar-section">
            Recent questions
        </div>
        """
    )

    recent = (
        st.session_state.messages[-5:]
    )

    if not recent:

        st.caption(
            "Your recent questions will appear here."
        )

    else:

        for i, message in enumerate(
            reversed(recent)
        ):

            label = message["question"]

            if len(label) > 34:

                label = (
                    label[:31]
                    + "..."
                )

            if st.button(
                label,
                key=(
                    f"recent_{i}_"
                    f"{message['question']}"
                ),
                use_container_width=True,
            ):

                go_to_assistant(
                    message["question"]
                )

    render_html(
        """
        <div
            style="
                position: fixed;
                bottom: 18px;
                color: #626a7a;
                font-size: .68rem;
            "
        >
            NovaAI · Company Knowledge Assistant
        </div>
        """
    )


render_html(
    """
    <div class="topbar">

        <div class="brand-mark">
            T
        </div>

        <div>

            <div class="brand-name">
                TechNova
            </div>

            <div class="brand-subtitle">
                Intelligent technology
            </div>

        </div>

    </div>
    """
)


nav1, nav2, nav3, nav4 = st.columns(4)


with nav1:

    if st.button(
        "Home",
        use_container_width=True,
    ):

        st.query_params["page"] = "home"

        st.rerun()


with nav2:

    if st.button(
        "Products",
        use_container_width=True,
    ):

        st.query_params["page"] = "products"

        st.rerun()


with nav3:

    if st.button(
        "About",
        use_container_width=True,
    ):

        st.query_params["page"] = "about"

        st.rerun()


with nav4:

    if st.button(
        "AI Assistant",
        use_container_width=True,
    ):

        st.query_params["page"] = "assistant"

        st.rerun()


page = st.query_params.get(
    "page",
    "assistant",
)


if page == "home":

    render_html(
        """
        <div class="hero">

            <div class="hero-badge">
                Intelligence for modern business
            </div>

            <div class="hero-title">

                Build smarter with

                <span class="gradient-text">
                    TechNova
                </span>

            </div>

            <div class="hero-description">

                Software engineering,
                artificial intelligence,
                cloud computing and
                intelligent automation
                for modern businesses.

            </div>

        </div>
        """
    )

    render_html(
        """
        <div class="section-label">
            Technology
        </div>
        """
    )

    render_html(
        """
        <div class="section-title">
            Explore TechNova
        </div>
        """
    )

    render_html(
        """
        <div class="section-description">

            Discover the platforms and
            technologies behind TechNova's
            intelligent solutions.

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html(
            """
            <div class="product-card">

                <div class="product-icon">
                    ✦
                </div>

                <div class="product-title">
                    NovaAI
                </div>

                <div class="product-subtitle">
                    Artificial intelligence platform
                </div>

                <div class="product-description">

                    Document processing,
                    semantic search,
                    question answering,
                    machine learning APIs
                    and generative AI capabilities.

                </div>

            </div>
            """
        )

    with col2:

        render_html(
            """
            <div class="product-card">

                <div class="product-icon">
                    ⚡
                </div>

                <div class="product-title">
                    NovaFlow
                </div>

                <div class="product-subtitle">
                    Workflow automation
                </div>

                <div class="product-description">

                    A workflow automation platform
                    for approvals, task routing,
                    business process automation,
                    notifications and analytics.

                </div>

            </div>
            """
        )

    if st.button(
        "✦  Ask the TechNova AI",
        use_container_width=True,
    ):

        go_to_assistant()


elif page == "products":

    render_html(
        """
        <div class="hero">

            <div class="hero-badge">
                TechNova platforms
            </div>

            <div class="hero-title">

                Products built for

                <span class="gradient-text">
                    intelligent work
                </span>

            </div>

            <div class="hero-description">

                Explore TechNova's software
                platforms for AI,
                workflow automation
                and business intelligence.

            </div>

        </div>
        """
    )

    products = [

        (
            "✦",
            "NovaAI",
            "ARTIFICIAL INTELLIGENCE",
            "An artificial intelligence platform "
            "providing document processing, semantic "
            "search, question answering, machine "
            "learning APIs and generative AI capabilities.",
        ),

        (
            "⚡",
            "NovaFlow",
            "WORKFLOW AUTOMATION",
            "A workflow automation platform for "
            "approvals, task routing, business process "
            "automation, notifications and workflow analytics.",
        ),

        (
            "▦",
            "NovaInsight",
            "BUSINESS ANALYTICS",
            "A business analytics platform providing "
            "dashboards, scheduled reports, metrics, "
            "data exploration and role-based analytics access.",
        ),

        (
            "☁",
            "NovaCloud",
            "CLOUD MANAGEMENT",
            "A cloud management and deployment platform "
            "for application environments, deployment "
            "automation, monitoring and operations.",
        ),

        (
            "◈",
            "NovaDesk",
            "CUSTOMER SUPPORT",
            "A customer support and ticket management "
            "product for service requests, incidents, "
            "knowledge articles and support analytics.",
        ),

    ]

    for start in range(
        0,
        len(products),
        2,
    ):

        columns = st.columns(2)

        for column, product in zip(
            columns,
            products[start:start + 2],
        ):

            (
                icon,
                title,
                subtitle,
                description,
            ) = product

            with column:

                render_html(
                    f"""
                    <div class="product-card">

                        <div class="product-icon">
                            {icon}
                        </div>

                        <div class="product-title">
                            {title}
                        </div>

                        <div class="product-subtitle">
                            {subtitle}
                        </div>

                        <div class="product-description">
                            {description}
                        </div>

                    </div>
                    """
                )


elif page == "about":

    render_html(
        """
        <div class="hero">

            <div class="hero-badge">
                About TechNova
            </div>

            <div class="hero-title">

                Technology with a

                <span class="gradient-text">
                    practical purpose
                </span>

            </div>

            <div class="hero-description">

                TechNova focuses on software engineering,
                artificial intelligence, cloud computing,
                data analytics and intelligent automation.

            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html(
            """
            <div class="product-card">

                <div class="product-title">
                    Company
                </div>

                <br>

                <div class="product-description">

                    <b>Founded:</b>
                    2019

                    <br><br>

                    <b>Headquarters:</b>
                    Kochi, Kerala, India

                    <br><br>

                    <b>Founder & CEO:</b>
                    Sreehari V S

                    <br><br>

                    <b>Industry:</b>
                    Software, AI, cloud,
                    data analytics and SaaS

                </div>

            </div>
            """
        )

    with col2:

        render_html(
            """
            <div class="product-card">

                <div class="product-title">
                    Mission
                </div>

                <br>

                <div class="product-description">

                    Build practical, reliable technology
                    that helps organizations automate work,
                    understand data and deliver better
                    digital services.

                </div>

            </div>
            """
        )


else:

    render_html(
        """
        <div class="assistant-shell">
        """
    )

    render_html(
        """
        <div class="assistant-header">

            <div class="assistant-icon">
                ✦
            </div>

            <div class="assistant-title">
                NovaAI Assistant
            </div>

            <div class="assistant-description">

                Ask questions about TechNova,
                its leadership, products,
                services and company policies.

            </div>

            <div class="kb-status">

                <span class="kb-dot"></span>

                Connected to TechNova knowledge base

            </div>

        </div>
        """
    )

    if not st.session_state.messages:

        render_html(
            """
            <div class="welcome-box">

                <div class="welcome-title">
                    What would you like to know?
                </div>

                <div class="welcome-text">

                    Ask naturally about TechNova,
                    NovaAI, NovaFlow, leadership,
                    products or company information.

                </div>

            </div>
            """
        )

        render_html(
            """
            <div class="followup-title">
                Try asking
            </div>
            """
        )

        suggestions = [

            "Who founded TechNova?",

            "What is NovaAI?",

            "What does NovaAI provide?",

            "What is NovaFlow?",

            "Where is TechNova headquartered?",

            "What are the typical fresher hiring stages?",

        ]

        for start in range(
            0,
            len(suggestions),
            2,
        ):

            cols = st.columns(2)

            for col, suggestion in zip(
                cols,
                suggestions[start:start + 2],
            ):

                with col:

                    if st.button(
                        suggestion,
                        key=f"welcome_{suggestion}",
                        use_container_width=True,
                    ):

                        ask_question(
                            suggestion
                        )

                        st.rerun()

    for message_index, message in enumerate(
        st.session_state.messages
    ):

        with st.chat_message("user"):

            st.markdown(
                message["question"]
            )

        with st.chat_message("assistant"):

            st.markdown(
                message["answer"]
            )

            render_sources(
                message.get(
                    "sources",
                    [],
                )
            )

            render_followups(
                message.get(
                    "followups",
                    [],
                ),
                f"followup_{message_index}",
            )

    if st.session_state.pending_question:

        query = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

        with st.chat_message("user"):

            st.markdown(
                query
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "NovaAI is thinking..."
            ):

                answer, results = (
                    process_question(
                        query
                    )
                )

            sources = unique_sources(
                results
            )

            followups = get_followups(
                answer
            )

            st.markdown(
                answer
            )

            render_sources(
                sources
            )

            render_followups(
                followups,
                f"pending_{len(st.session_state.messages)}",
            )

        st.session_state.messages.append(
            {
                "question": query,
                "answer": answer,
                "sources": sources,
                "followups": followups,
            }
        )

        st.rerun()

    query = st.chat_input(
        "Ask NovaAI anything about TechNova..."
    )

    if query:

        ask_question(
            query
        )

        st.rerun()

    render_html(
        """
        <div class="app-footer">

            NovaAI answers are grounded in the
            TechNova company knowledge base.

        </div>
        """
    )

    render_html(
        """
        </div>
        """
    )