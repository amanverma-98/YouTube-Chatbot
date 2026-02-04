import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pytube import YouTube

st.markdown("""
<style>
.chat-wrapper {
    height: 70vh;
    display: flex;
    flex-direction: column;
}

.chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    background-color: #fafafa;
}

.chat-input {
    position: sticky;
    bottom: 0;
    background: white;
    padding-top: 10px;
}
.user-msg {
    background: #DCF8C6;
    padding: 8px 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
}
.bot-msg {
    background: #F1F0F0;
    padding: 8px 12px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


# Streamlit config
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)


# Helper functions
def get_video_title(video_id: str) -> str:
    try:
       yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
       return yt.title
    except Exception:
       return "Unable to fetch title"


def load_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()   # 👈 instantiate

    try:
        transcript = api.fetch(
            video_id, languages=["hi"]
        )
        lang = "Hindi"
    except Exception:
        transcript = api.fetch(
            video_id, languages=["en"]
        )
        lang = "English"

    text = " ".join(t.text for t in transcript)
    return text, lang




def create_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=st.secrets["GOOGLE_API_KEY"]
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def build_rag_chain(vectorstore):
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        streaming=True,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
            The provided context may be in Hindi or English.
            ALWAYS answer in clear, simple English.
            If user ask general question you can give answer so user doesn't feel ignorance.
            If the answer is not present, say "I don't know".
            Context:
            {context}
            Question:
            {question}"""
            )
    parser = StrOutputParser()
    rag_chain = {"context": vectorstore.as_retriever(search_kwargs={"k": 4}), "question": RunnablePassthrough() } | prompt | llm| parser

    return rag_chain

def scroll_to_bottom():
    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.querySelector('section.main');
        if (chatContainer) {
            chatContainer.scrollTo(0, chatContainer.scrollHeight);
        }
        </script>
        """,
        unsafe_allow_html=True
    )



# UI
st.title("🎥 YouTube RAG Chatbot")
st.caption("Ask questions from any YouTube video using Gemini + RAG")

left_col, right_col = st.columns([1, 2])



# LEFT COLUMN
with left_col:
        st.subheader("📌 Video Metadata")
        video_url = st.text_input("YouTube URL")
        
        
        if video_url:
           video_id = video_url.split("v=")[-1]
           title = get_video_title(video_id)
           st.markdown(f"**🎬 Title:** {title}")
           st.markdown(f"**🆔 Video ID:** `{video_id}`")

        with st.spinner("Building vector database..."):
            vectorstore = create_vectorstore(transcript_text)
            st.success("Vector DB ready")

        rag_chain = build_rag_chain(vectorstore)
        st.session_state["rag_chain"] = rag_chain


# RIGHT COLUMN
with right_col:
    st.subheader("💬 Chat")

    if "rag_chain" not in st.session_state:
        st.info("👈 Enter a YouTube Video ID to start")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # CHAT HISTORY (SCROLLABLE)

        # INPUT BAR (ALWAYS VISIBLE)
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input(
                "",
                placeholder="Ask something about the video..."
            )
            submitted = st.form_submit_button("Send")

        st.markdown(f"**You:** {query}")
        if submitted and query:
            response_box = st.empty()
            streamed_answer = ""
            with st.spinner("Thinking..."):
                for chunk in st.session_state["rag_chain"].stream(query):
                    streamed_answer += chunk
                    response_box.markdown(f"**Bot:** {streamed_answer}")

            st.session_state.chat_history.append((query, streamed_answer))

        chat_container = st.container(height=380)

        with chat_container:
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
                st.markdown("---")



st.markdown(
    "Built with ❤️ using Streamlit · LangChain · Groq · FAISS"
)


