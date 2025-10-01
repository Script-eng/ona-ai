import streamlit as st
import os
import fitz  # PyMuPDF
import chromadb
import tempfile
import whisper
import asyncio
import threading
import hashlib
import re
from io import BytesIO
from gtts import gTTS
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import mic_recorder

# --- Configuration ---
@dataclass
class Config:
    CHROMA_PATH: str = "chroma_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONVERSATION_TURNS: int = 20
    MAX_MEMORY_TOKENS: int = 1000
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    WHISPER_MODEL: str = "base"

config = Config()

# --- App Configuration ---
st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("📚 AI Study Assistant: Your All-in-One Learning Partner")
st.markdown("Interact via text or voice. Every assistant response comes with an audio narration option.")

# --- Utility Classes ---
class QueryRouter:
    """Intelligent LLM-based query router with caching and optimization"""
    
    def __init__(self, llm):
        self.llm = llm
        self.cache = {}  # Simple cache for routing decisions
        self.router_prompt = PromptTemplate.from_template("""
You are a query classifier for a study assistant. Classify the user's query into ONE of these categories:

1. "retrieval_query" - User wants information from uploaded documents, asks about specific content, or wants to find/search something in their materials
2. "greeting_or_social" - Simple greetings, thanks, social pleasantries, or casual conversation  
3. "general_knowledge" - General questions that don't require document search

Examples:
- "What does chapter 3 say about..." → retrieval_query
- "Explain the concept mentioned in the document" → retrieval_query
- "Find information about X in my files" → retrieval_query
- "Hi how are you?" → greeting_or_social
- "Thanks for the help!" → greeting_or_social
- "What is photosynthesis?" → general_knowledge
- "How do neural networks work?" → general_knowledge

User Query: "{query}"

Classification (one word only):""")
    
    def route_query(self, query: str) -> str:
        """Intelligent query classification with caching"""
        # Simple cache based on query similarity
        query_key = query.lower().strip()
        if query_key in self.cache:
            return self.cache[query_key]
        
        try:
            chain = self.router_prompt | self.llm
            result = chain.invoke({"query": query})
            classification = result.content.strip().lower()
            
            # Validate and clean the result
            valid_types = ["retrieval_query", "greeting_or_social", "general_knowledge"]
            for valid_type in valid_types:
                if valid_type in classification:
                    classification = valid_type
                    break
            else:
                # Default fallback
                classification = "general_knowledge"
            
            # Cache the result
            self.cache[query_key] = classification
            return classification
            
        except Exception as e:
            # Fallback: if documents exist, try retrieval, otherwise general knowledge
            st.sidebar.warning(f"Router error: {e}")
            if len(st.session_state.processed_docs) > 0:
                return "retrieval_query"
            return "general_knowledge"

class AudioCache:
    """Simple in-memory cache for generated audio"""
    
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, bytes] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[bytes]:
        """Get cached audio or None"""
        key = self._generate_key(text)
        if key in self.cache:
            
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, text: str, audio_bytes: bytes):
        """Cache audio with LRU eviction"""
        key = self._generate_key(text)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = audio_bytes
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

# --- Model Management ---
@st.cache_resource
def load_core_models():
    """Load only essential models upfront"""
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in Streamlit secrets.")
            st.stop()

        llm = ChatGroq(
            model_name=config.LLM_MODEL,
            temperature=0.7,
            groq_api_key=groq_api_key
        )
        
        embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        return llm, embedding_function
    except Exception as e:
        st.error(f"Error loading core models: {e}")
        st.stop()

@st.cache_resource
def load_whisper_model():
    """Lazy load Whisper model only when needed"""
    try:
        return whisper.load_model(config.WHISPER_MODEL)
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

llm, embedding_function = load_core_models()

# --- Initialize Session State with Safety Checks ---
def initialize_session_state():
    """Initialize all session state variables safely"""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationSummaryBufferMemory(
            llm=llm, 
            max_token_limit=config.MAX_MEMORY_TOKENS, 
            memory_key="chat_history", 
            return_messages=True
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = AudioCache()
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = set()
    if "query_router" not in st.session_state:
        st.session_state.query_router = QueryRouter(llm)

# Initialize everything
initialize_session_state()

# --- Core Functionality ---
@st.cache_resource
def load_vector_store():
    """Load and cache vector store with diagnostics"""
    try:
        # Ensure directory exists
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        vector_store = Chroma(
            client=client,
            collection_name="study_docs",
            embedding_function=embedding_function
        )
        
        # Verify vector store is working
        try:
            collection = client.get_collection("study_docs")
            doc_count = collection.count()
            st.sidebar.success(f"Vector store loaded: {doc_count} documents")
        except Exception:
            # Collection doesn't exist yet, create it
            collection = client.create_collection("study_docs")
            st.sidebar.info("Created new vector store collection")
            
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def get_document_hash(pdf_file) -> str:
    """Generate hash for document deduplication"""
    pdf_file.seek(0)
    content = pdf_file.read()
    pdf_file.seek(0)  # Reset for later use
    return hashlib.md5(content).hexdigest()

def process_pdf_streaming(pdf_file, vector_store) -> bool:
    """Process PDF with better memory management and deduplication"""
    try:
        doc_hash = get_document_hash(pdf_file)
        if doc_hash in st.session_state.processed_docs:
            st.warning("This document has already been processed.")
            return False
        
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting
        )
        
        batch_texts = []
        batch_metadatas = []
        batch_ids = []
        batch_size = 10  # Process in batches
        
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            if not text.strip():
                continue
                
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                chunk_id = f"{doc_hash}_page{page_num + 1}_chunk{i}"
                metadata = {
                    "page": page_num + 1, 
                    "source": pdf_file.name,
                    "doc_hash": doc_hash,
                    "chunk_length": len(chunk)
                }
                
                batch_texts.append(chunk)
                batch_metadatas.append(metadata)
                batch_ids.append(chunk_id)
                
                # Process in batches
                if len(batch_texts) >= batch_size:
                    vector_store.add_texts(
                        texts=batch_texts, 
                        metadatas=batch_metadatas, 
                        ids=batch_ids
                    )
                    batch_texts, batch_metadatas, batch_ids = [], [], []
        
        # Process remaining batch
        if batch_texts:
            vector_store.add_texts(
                texts=batch_texts, 
                metadatas=batch_metadatas, 
                ids=batch_ids
            )
        
        st.session_state.processed_docs.add(doc_hash)
        pdf_document.close()
        return True
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def transcribe_audio_safe(audio_bytes) -> Optional[str]:
    """Safe audio transcription with proper cleanup"""
    whisper_model = load_whisper_model()
    if not whisper_model:
        return None
    
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        result = whisper_model.transcribe(tmp_file_path, fp16=False)
        return result['text'].strip()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def generate_audio_sync(text: str) -> Optional[bytes]:
    """Generate audio synchronously with caching"""
    try:
        # Check cache first
        cached_audio = st.session_state.audio_cache.get(text)
        if cached_audio:
            return cached_audio
        
        # Generate new audio
        audio_fp = BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()
        
        # Cache the result
        st.session_state.audio_cache.put(text, audio_bytes)
        return audio_bytes
        
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

def manage_conversation_length():
    """Limit conversation length to prevent memory bloat"""
    if len(st.session_state.messages) > config.MAX_CONVERSATION_TURNS * 2:
        # Keep only recent messages
        st.session_state.messages = st.session_state.messages[-config.MAX_CONVERSATION_TURNS:]
        # Clear and rebuild memory
        st.session_state.memory.clear()

# --- UI Components ---
vector_store = load_vector_store()

with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a document (PDF):", type="pdf")
    
    if uploaded_file and st.button("Add to Knowledge Base"):
        if vector_store:
            with st.spinner("Processing document..."):
                if process_pdf_streaming(uploaded_file, vector_store):
                    # Verify the document was actually added
                    try:
                        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
                        collection = client.get_collection("study_docs")
                        doc_count = collection.count()
                        st.success(f"Document added successfully! Total chunks: {doc_count}")
                        
                        # Test a simple query to verify it works
                        test_docs = vector_store.similarity_search("test", k=1)
                        if test_docs:
                            st.info("✅ Document retrieval test passed")
                        else:
                            st.warning("⚠️ No content found in retrieval test")
                    except Exception as e:
                        st.error(f"Verification failed: {e}")
                else:
                    st.error("Failed to process document")
        else:
            st.error("Vector store not available")
    
    st.write(f"Processed documents: {len(st.session_state.processed_docs)}")
    
    if st.button("Clear Knowledge Base"):
        if os.path.exists(config.CHROMA_PATH):
            import shutil
            shutil.rmtree(config.CHROMA_PATH)
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.processed_docs.clear()
        st.success("Knowledge base cleared. Please refresh the page.")
        st.rerun()

    # Audio toggle
    audio_enabled = st.checkbox("Enable Audio Responses", value=True)

st.header("💬 Chat with your Assistant")

# Display chat history with pagination
max_display_messages = 10
start_idx = max(0, len(st.session_state.messages) - max_display_messages)

for message in st.session_state.messages[start_idx:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (message["role"] == "assistant" and 
            audio_enabled and 
            "audio" in message and 
            message["audio"] is not None):
            try:
                st.audio(message["audio"])
            except Exception as e:
                # Silently handle audio display errors
                pass

def handle_prompt(prompt: str):
    """Optimized prompt handling with better routing and async audio"""
    manage_conversation_length()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Intelligent LLM-based routing with caching
            route = st.session_state.query_router.route_query(prompt)
            
            # Debug information
            st.sidebar.write(f"**Query route:** {route}")
            
            if route == "retrieval_query" and vector_store:
                st.info("🔍 Searching knowledge base...")
                
                # Check if vector store has documents
                try:
                    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
                    collection = client.get_collection("study_docs")
                    doc_count = collection.count()
                    
                    if doc_count == 0:
                        st.warning("No documents found in knowledge base. Please upload a PDF first.")
                        response = "I don't have any documents in my knowledge base yet. Please upload a PDF document first, then ask me questions about it."
                    else:
                        # Try retrieval
                        retriever = vector_store.as_retriever(
                            search_kwargs={'k': 5}
                        )
                        # Test retrieval first
                        docs = retriever.get_relevant_documents(prompt)
                        
                        if not docs:
                            st.warning("No relevant documents found. Trying general knowledge...")
                            response = llm.invoke(f"Answer this question: {prompt}").content
                        else:
                            qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm, 
                                retriever=retriever,
                                memory=st.session_state.memory
                            )
                            result = qa_chain.invoke({"question": prompt})
                            response = result["answer"]
                            
                            # Show source info
                            st.sidebar.write(f"**Found {len(docs)} relevant chunks**")
                            
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    st.info("🧠 Falling back to general knowledge...")
                    response = llm.invoke(prompt).content
                    
            elif route == "greeting_or_social":
                st.info("👋 Responding socially...")
                social_prompt = PromptTemplate.from_template(
                    "You are a friendly AI study assistant. Respond warmly and briefly to: '{prompt}'"
                )
                chain = social_prompt | llm
                response = chain.invoke({"prompt": prompt}).content
            else:
                # For general knowledge, check if we should try retrieval anyway
                if vector_store and len(st.session_state.processed_docs) > 0:
                    st.info("🔍 Checking documents first...")
                    try:
                        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
                        docs = retriever.get_relevant_documents(prompt)
                        
                        if docs and len(docs) > 0:
                            # Found relevant documents, use them
                            qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm, 
                                retriever=retriever,
                                memory=st.session_state.memory
                            )
                            result = qa_chain.invoke({"question": prompt})
                            response = result["answer"]
                            st.sidebar.write(f"**Used {len(docs)} document chunks**")
                        else:
                            st.info("🧠 Using general knowledge...")
                            response = llm.invoke(prompt).content
                    except Exception as e:
                        st.info("🧠 Using general knowledge...")
                        response = llm.invoke(prompt).content
                else:
                    st.info("🧠 Using general knowledge...")
                    response = llm.invoke(prompt).content

            # Display response immediately
            response_placeholder = st.empty()
            response_placeholder.markdown(response)
            
            # Generate audio if enabled
            if audio_enabled:
                audio_placeholder = st.empty()
                
                # Check cache first for instant playback
                cached_audio = st.session_state.audio_cache.get(response)
                if cached_audio:
                    audio_placeholder.audio(cached_audio)
                else:
                    # Generate audio with progress indicator
                    with audio_placeholder.container():
                        with st.spinner("🔊 Generating audio..."):
                            audio_bytes = generate_audio_sync(response)
                            if audio_bytes:
                                audio_placeholder.audio(audio_bytes)
                            else:
                                audio_placeholder.empty()
    
    # Store message with audio if available
    message_data = {"role": "assistant", "content": response}
    if audio_enabled:
        # Get audio from cache if it exists
        audio_bytes = st.session_state.audio_cache.get(response)
        message_data["audio"] = audio_bytes
    else:
        message_data["audio"] = None
    
    st.session_state.messages.append(message_data)

# Input interface
col1, col2 = st.columns([4, 1])

with col1:
    if prompt := st.chat_input("Ask a question..."):
        handle_prompt(prompt)

with col2:
    st.write("&nbsp;")
    audio_data = mic_recorder(
        start_prompt="🎤", 
        stop_prompt="⏹️", 
        key='recorder', 
        format="wav"
    )
    if audio_data:
        with st.spinner("🎙️ Transcribing..."):
            transcribed_text = transcribe_audio_safe(audio_data['bytes'])
            if transcribed_text:
                st.success(f"Transcribed: {transcribed_text[:50]}...")
                handle_prompt(transcribed_text)
            else:
                st.error("Transcription failed. Please try again.")

# Display conversation stats in sidebar
with st.sidebar:
    st.write("---")
    st.write(f"**Conversation turns:** {len(st.session_state.messages) // 2}")
    st.write(f"**Audio cache size:** {len(st.session_state.audio_cache.cache)}")