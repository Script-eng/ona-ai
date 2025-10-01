import os
import io
import sys
import time
import json
import tempfile
import hashlib
from io import BytesIO
from typing import Optional

import streamlit as st
import requests
from gtts import gTTS

# Optional: speech input
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# LLM + Retrieval
try:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import chromadb
    LLM_STACK_AVAILABLE = True
except Exception:
    LLM_STACK_AVAILABLE = False


# ---------------- Config ----------------
SADTALKER_API = os.environ.get("SADTALKER_API", "http://127.0.0.1:7861/generate_video")
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"


# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: #667eea;
        font-size: 1.2rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-offline {
        background-color: #ef4444;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Avatar preview */
    .avatar-preview {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 5px;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Success/Error messages */
    .success-message {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Video player styling */
    video {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- UI Setup ----------------
st.set_page_config(
    page_title="Study Assistant + SadTalker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 ONA AI Study Assistant</h1>
    <p>Ask questions, get answers, and see your AI tutor come to life</p>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # API Status
    st.markdown("#### SadTalker API Status")
    try:
        resp = requests.get(SADTALKER_API.replace("/generate_video", "/health"), timeout=2)
        api_status = "online" if resp.status_code == 200 else "offline"
    except:
        api_status = "offline"
    
    status_class = "status-online" if api_status == "online" else "status-offline"
    st.markdown(f'<div><span class="status-indicator {status_class}"></span><span>{SADTALKER_API}</span></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.markdown("#### 🎬 Video Generation Settings")
    auto_video = st.toggle(
        "Auto-generate talking head",
        value=False,
        help="Automatically creates a talking head video for each assistant response"
    )
    
    if auto_video:
        st.info("💡 Video will be generated automatically after each answer. This may take some time.")
    
    st.divider()
    
    # Knowledge Base Upload
    st.markdown("#### 📚 Knowledge Base")
    st.markdown("Upload PDF documents to expand the assistant's knowledge")
    
    pdf_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload study materials, textbooks, or reference documents",
        label_visibility="collapsed"
    )
    
    if pdf_files:
        st.success(f"✅ {len(pdf_files)} document(s) uploaded")
        for pdf in pdf_files:
            st.text(f"📄 {pdf.name}")
    
    st.divider()
    
    # System Info
    st.markdown("#### 🔧 System Information")
    st.markdown(f"""
    - **Whisper**: {'✅ Available' if WHISPER_AVAILABLE else '❌ Unavailable'}
    - **LLM Stack**: {'✅ Available' if LLM_STACK_AVAILABLE else '❌ Unavailable'}
    - **Embedding Model**: {EMBEDDING_MODEL}
    - **LLM Model**: {LLM_MODEL}
    """)


# ---------------- Session State ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "avatar_image" not in st.session_state:
    st.session_state.avatar_image = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None


# ---------------- Helper Functions ----------------
def cache_tts(text: str) -> bytes:
    """Generate or fetch cached TTS (mp3 bytes)."""
    key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
    if key in st.session_state.audio_cache:
        return st.session_state.audio_cache[key]
    buf = BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)
    audio_bytes = buf.read()
    st.session_state.audio_cache[key] = audio_bytes
    return audio_bytes


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe with Whisper if available."""
    if not WHISPER_AVAILABLE:
        return ""
    if st.session_state.whisper_model is None:
        try:
            st.session_state.whisper_model = whisper.load_model("base")
        except Exception:
            return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        out = st.session_state.whisper_model.transcribe(tmp_path, fp16=False)
        return (out.get("text") or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def call_sadtalker_api(
    image_bytes: bytes,
    audio_bytes: Optional[bytes],
    *,
    text_fallback: Optional[str],
    pose_style: int = 0,
    size_of_image: int = 256,
    preprocess_type: str = "crop",
    still_mode: bool = False,
    enhancer: bool = False,
    batch_size: int = 1,
) -> Optional[bytes]:
    """Send request to SadTalker API and return MP4 bytes."""
    try:
        files = {"image": ("avatar.png", image_bytes, "image/png")}
        data = {
            "pose_style": str(pose_style),
            "size_of_image": str(size_of_image),
            "preprocess_type": preprocess_type,
            "still_mode": "true" if still_mode else "false",
            "enhancer": "true" if enhancer else "false",
            "batch_size": str(batch_size),
        }
        if audio_bytes:
            files["audio"] = ("speech.mp3", audio_bytes, "audio/mpeg")
        elif text_fallback:
            data["text"] = text_fallback
        else:
            st.error("❌ No audio or text provided to SadTalker.")
            return None

        resp = requests.post(SADTALKER_API, files=files, data=data, timeout=None)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.error(f"❌ SadTalker API error [{resp.status_code}]: {detail}")
            return None

        return resp.content
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to reach SadTalker API: {e}")
        return None


def answer_with_llm(question: str) -> str:
    """Answer user query with Groq + Chroma if available."""
    if not LLM_STACK_AVAILABLE:
        return f"LLM stack is not available. Your question: {question}"

    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return f"GROQ_API_KEY is missing. Your question: {question}"

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = Chroma(client=client, collection_name="study_docs", embedding_function=embeddings)
        llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7, groq_api_key=api_key)

        docs = vs.similarity_search(question, k=3)
        ctx = "\n\n".join([d.page_content for d in docs]) if docs else ""
        prompt = f"Answer the user's question clearly and concisely.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        out = llm.invoke(prompt).content
        return out.strip() if out else question
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        return question


# ---------------- Main Layout ----------------
col_chat, col_avatar = st.columns([2.5, 1.5])

# ---------------- Chat Column ----------------
with col_chat:
    st.markdown("### 💬 Conversation")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for idx, turn in enumerate(st.session_state.chat):
            with st.chat_message(turn["role"]):
                st.markdown(turn["text"])
                
                # Audio playback
                if turn.get("audio_bytes"):
                    st.audio(turn["audio_bytes"], format="audio/mp3")
                
                # Video playback
                if turn.get("video_bytes"):
                    st.video(turn["video_bytes"])
    
    # User input area
    st.markdown("---")
    
    # Voice input option
    if WHISPER_AVAILABLE:
        voice_col1, voice_col2 = st.columns([3, 1])
        with voice_col1:
            audio_input = st.file_uploader(
                "🎤 Upload voice recording",
                type=["wav", "mp3", "m4a"],
                help="Upload an audio file to transcribe and ask a question",
                key="voice_upload"
            )
        with voice_col2:
            if audio_input and st.button("🔊 Transcribe"):
                with st.spinner("Transcribing audio..."):
                    audio_bytes = audio_input.read()
                    transcribed = transcribe(audio_bytes)
                    if transcribed:
                        st.success(f"✅ Transcribed: {transcribed}")
                        # Process as user input
                        st.session_state.chat.append({"role": "user", "text": transcribed})
                        st.rerun()
    
    # Text input
    user_text = st.chat_input("💭 Type your question here...")
    
    if user_text:
        # Add user message
        st.session_state.chat.append({"role": "user", "text": user_text})
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = answer_with_llm(user_text)
                st.markdown(answer)
                
                # Generate TTS
                with st.spinner("🔊 Generating speech..."):
                    audio_bytes = cache_tts(answer)
                    st.audio(audio_bytes, format="audio/mp3")
                
                # Save to chat history
                st.session_state.chat.append({
                    "role": "assistant",
                    "text": answer,
                    "audio_bytes": audio_bytes,
                    "video_bytes": None,
                })
                
                # Auto-generate video if enabled
                if auto_video and st.session_state.avatar_image:
                    with st.spinner("🎬 Generating talking head video..."):
                        vid = call_sadtalker_api(
                            image_bytes=st.session_state.avatar_image.getvalue(),
                            audio_bytes=audio_bytes,
                            text_fallback=answer
                        )
                        if vid:
                            st.video(vid)
                            st.session_state.chat[-1]["video_bytes"] = vid
                            st.success("✅ Video generated successfully!")
                elif auto_video:
                    st.warning("⚠️ No avatar uploaded. Please upload an avatar image to generate videos.")
        
        st.rerun()


# ---------------- Avatar & Controls Column ----------------
with col_avatar:
    st.markdown("### 🎭 Avatar Configuration")
    
    # Avatar upload
    avatar_file = st.file_uploader(
        "Upload face image for avatar",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear face image for the talking head animation",
        key="avatar_upload"
    )
    
    if avatar_file:
        buf = io.BytesIO(avatar_file.read())
        buf.seek(0)
        st.session_state.avatar_image = buf
        st.markdown('<div class="avatar-preview">', unsafe_allow_html=True)
        st.image(st.session_state.avatar_image, caption="Current Avatar", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("✅ Avatar loaded successfully!")
    elif st.session_state.avatar_image:
        st.markdown('<div class="avatar-preview">', unsafe_allow_html=True)
        st.image(st.session_state.avatar_image, caption="Current Avatar", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📸 Upload an avatar image to enable talking head generation")
    
    st.markdown("---")
    
    # Video generation controls
    st.markdown("### 🎬 Manual Video Generation")
    
    with st.expander("⚙️ Advanced Settings", expanded=False):
        pose = st.slider("Pose style", 0, 46, 0, help="Choose different head pose styles")
        size = st.radio("Video resolution", [256, 512], index=0, horizontal=True)
        preprocess = st.selectbox(
            "Preprocessing method",
            ["crop", "resize", "full", "extcrop", "extfull"],
            index=0,
            help="How to process the input image"
        )
        still = st.checkbox("Still mode", value=False, help="Minimal head movement")
        enhance = st.checkbox("Use GFPGAN enhancer", value=False, help="Enhance video quality (slower)")
        bsize = st.slider("Batch size", 1, 8, 1, help="Processing batch size")
    
    if st.button("🎥 Generate Video for Last Answer", use_container_width=True):
        if not st.session_state.avatar_image:
            st.error("❌ Please upload an avatar image first!")
        else:
            last = next((t for t in reversed(st.session_state.chat) if t["role"] == "assistant"), None)
            if not last:
                st.error("❌ No assistant answer found. Ask a question first!")
            else:
                ans_text = last["text"]
                audio_bytes = last.get("audio_bytes") or cache_tts(ans_text)
                
                with st.spinner("🎬 Rendering talking head video... This may take a minute."):
                    vid = call_sadtalker_api(
                        image_bytes=st.session_state.avatar_image.getvalue(),
                        audio_bytes=audio_bytes,
                        text_fallback=ans_text,
                        pose_style=pose,
                        size_of_image=size,
                        preprocess_type=preprocess,
                        still_mode=still,
                        enhancer=enhance,
                        batch_size=bsize
                    )
                
                if vid:
                    st.success("✅ Video generated successfully!")
                    st.video(vid)
                    last["video_bytes"] = vid
                    st.rerun()
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🔧 Quick Actions")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Reset Avatar", use_container_width=True):
            st.session_state.avatar_image = None
            st.rerun()


# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with Streamlit • Powered by Groq LLM, SadTalker & Whisper</p>
</div>
""", unsafe_allow_html=True)