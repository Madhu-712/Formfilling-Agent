

import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.qdrant import Qdrant
import pytesseract
from PIL import Image
import os
import tempfile

# Set API keys from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_KEY']
os.environ['QDRANT_API_KEY'] = st.secrets['api_key']
os.environ['QDRANT_URL'] = st.secrets['QDRANT_URL']

# --- Functions ---
def init_knowledge_base(pdf_path):
    kb = PDFKnowledgeBase(
        path=pdf_path,  # Using dynamic PDF path
        vector_db=Qdrant(collection="formfill phidata-qdrant-ipynb"),
        url=os.environ['QDRANT_URL'],
        api_key=os.environ['QDRANT_API_KEY'],
    )
    kb.load(upsert=True)
    return kb

def init_agent(knowledge_base):
    agent = Agent(
        name="Rag Agent",
        role="Do RAG on pdf doc by fetching from knowledge base(kb)",
        model=Gemini(id="gemini-2.0-pro-vision"),
        knowledge=knowledge_base,
        search_knowledge=True,
        read_chat_history=True,
        show_tool_calls=True,
        markdown=True,
        instructions=["Do a RAG on pdf document present in knowledge base(kb)"],
    )
    return agent

def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def generate_final_form(agent, ocr_text, query):
    combined_prompt = f"""
    Use the extracted form data (OCR) and the knowledge base to generate the final bank form:
    - Extracted Form Data (OCR): {ocr_text}
    - Data to be filled (from query): {query}
    """
    output = agent.run(combined_prompt)
    return output.content if output else "Unable to generate form."


# --- Streamlit App UI ---
st.set_page_config(page_title="Form Auto-Filler", layout="wide")
st.title("🏦  Form Auto-Filler with OCR + RAG")

tab1, tab2 = st.tabs(["📄 Upload  Data (PDF)", "🖼️ Upload Form (Image)"])

# TAB 1: PDF Upload for Synthetic Data
with tab1:
    uploaded_pdf = st.file_uploader("Upload Data PDF", type=["pdf"])
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())
            pdf_path = temp_pdf.name

        st.success("PDF uploaded successfully!")

        # Initialize Knowledge Base & Agent
        knowledge_base = init_knowledge_base(pdf_path)
        agent = init_agent(knowledge_base)

        query = st.text_input("Enter your query to auto-fill the form:", placeholder="e.g., Fill the account form for John Doe")
        if st.button("Generate Form from PDF"):
            with st.spinner("Generating form using RAG..."):
                output = agent.run(query)
                st.markdown("### 📝 Auto-Filled Form:")
                st.markdown(output.content)

# TAB 2: Image Upload for Bank Form
with tab2:
    uploaded_image = st.file_uploader("Upload Blank Form Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Bank Form", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:  # Assuming JPEG format
            temp_image.write(uploaded_image.read())
            image_path = temp_image.name
        # Extract text using OCR
        if st.button("Extract Fields from Image (OCR)"):
            with st.spinner("Extracting text from image..."):
                ocr_text = extract_text_from_image(image_path)
                st.text_area("Extracted Text:", ocr_text, height=200)
        
        # Combine OCR + RAG for final form generation
        if st.button("Generate Final Form (OCR + RAG)"):
            if uploaded_pdf and query:
                with st.spinner("Combining OCR and RAG to generate the final form..."):
                    final_output = generate_final_form(agent, ocr_text, query)
                    st.markdown("### ✅ Final Auto-Filled Form:")
                    st.markdown(final_output)
            else:
                st.warning("Please upload PDF, enter a query and extract text using OCR from the uploaded image to proceed.")
