## 📂 Project Structure

Two separate environments are required.

First, pip install requirements.txt
source venv/bin/activate
streamlit run mainmain.py

cd SadTalker-Implementation
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 7861 --reload

export GROQ_API_KEY=your_groq_api_key
export SADTALKER_API=http://127.0.0.1:7861/generate_video
