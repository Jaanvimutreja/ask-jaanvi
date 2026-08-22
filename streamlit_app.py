import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/api/v1/chat"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ask Jaanvi",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PURPLE / BLACK UI THEME
# ============================================================

st.html(
    """
    <style>

    /* =========================
       MAIN APP
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 50% 0%,
                rgba(139, 92, 246, 0.14),
                transparent 38%
            ),
            #08080D;
    }

    .block-container {
        max-width: 900px;
        padding-top: 4rem;
        padding-bottom: 6rem;
    }


    /* =========================
       TITLE
       ========================= */

    h1 {
        color: #F5F3FF !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
        font-size: 3.2rem !important;
    }

    .stCaption {
        color: #9CA3AF !important;
        font-size: 1rem;
    }


    /* =========================
       ALL ALERT BOXES
       ========================= */

    [data-testid="stAlert"] {
        background: rgba(139, 92, 246, 0.07) !important;
        border: 1px solid rgba(168, 85, 247, 0.20) !important;
        border-radius: 14px !important;
    }


    /* =========================
       CHAT MESSAGES
       ========================= */

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.035) !important;
        border: 1px solid rgba(255, 255, 255, 0.055) !important;
        border-radius: 18px !important;
        padding: 1rem 1.15rem !important;
        margin-bottom: 0.8rem !important;
    }


    /* =========================
       CHAT AVATARS
       ========================= */

    [data-testid="stChatMessageAvatar"] {
        background: #7C3AED !important;
        border: 1px solid rgba(192, 132, 252, 0.35) !important;
    }


    [data-testid="chatAvatarIcon-user"] {
        background: #7C3AED !important;
        color: white !important;
    }


    [data-testid="chatAvatarIcon-assistant"] {
        background: #A855F7 !important;
        color: white !important;
    }


    /* =========================
       CHAT INPUT
       ========================= */

    [data-testid="stChatInput"] {
        border-radius: 18px !important;
    }


    [data-testid="stChatInput"] textarea {
        background: #111118 !important;
        color: #F5F3FF !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 16px !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {
        color: #8F8A9D !important;
    }


    [data-testid="stChatInput"] textarea:focus {
        border-color: #A855F7 !important;
        box-shadow: 0 0 0 1px #A855F7 !important;
    }


    /* =========================
       SEND BUTTON
       ========================= */

    [data-testid="stChatInput"] button {
        background: #7C3AED !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
    }


    [data-testid="stChatInput"] button:hover {
        background: #9333EA !important;
    }


    /* =========================
       TEXT
       ========================= */

    p,
    li {
        color: #F3F0F7;
    }


    strong {
        color: #E9D5FF;
    }


    /* =========================
       LINKS
       ========================= */

    a {
        color: #C084FC !important;
    }


    /* =========================
       SCROLLBAR
       ========================= */

    ::-webkit-scrollbar {
        width: 7px;
    }

    ::-webkit-scrollbar-track {
        background: #08080D;
    }

    ::-webkit-scrollbar-thumb {
        background: #4C1D95;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #7C3AED;
    }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown("# ✦ Ask Jaanvi")

st.caption("AI-powered portfolio assistant")

st.markdown(
    "🟣 **Online · Ready to answer**"
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        👋 Ask me about Jaanvi's **projects, skills, experience,
        engineering approach, or technical thinking.**
        """
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        # IMPORTANT:
        # No custom avatar value.
        # Streamlit default avatar is used and CSS styles it.

        with st.chat_message("user"):
            st.markdown(message["content"])

    else:

        # IMPORTANT:
        # No avatar="✦" here.
        # That was causing the StreamlitAPIException.

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask something about Jaanvi..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_message:

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(user_message)


    # --------------------------------------------------------
    # Prepare conversation history
    # --------------------------------------------------------

    conversation_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[:-1]
    ]


    # --------------------------------------------------------
    # Call FastAPI backend
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "message": user_message,
                        "conversation_history": conversation_history,
                    },
                    timeout=60,
                )


                # Raise error for 4xx / 5xx
                response.raise_for_status()


                # Parse JSON
                data = response.json()


                # Extract answer
                answer = data.get("response")


                if not answer:

                    answer = (
                        "Sorry, I couldn't generate a response."
                    )


                # Display answer
                st.markdown(answer)


                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            # ------------------------------------------------
            # CONNECTION ERROR
            # ------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to Ask Jaanvi API. "
                    "Make sure the FastAPI server is running."
                )


            # ------------------------------------------------
            # TIMEOUT
            # ------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "The API request timed out. Please try again."
                )


            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            except requests.exceptions.HTTPError as exc:

                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else "unknown"
                )

                st.error(
                    f"API error: {status_code}"
                )


            # ------------------------------------------------
            # OTHER REQUEST ERROR
            # ------------------------------------------------

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Request failed: {exc}"
                )


            # ------------------------------------------------
            # INVALID JSON
            # ------------------------------------------------

            except ValueError:

                st.error(
                    "The API returned an invalid JSON response."
                )