
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.pdf import PDFFileKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
import pytesseract
from PIL import Image
import os

# PostgreSQL connection
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Environment variable for Google API
os.environ['GOOGLE_API_KEY'] = st.secrets.get('GEMINI_KEY', '')

# Initialize Knowledge Base
@st.cache_resource
def init_knowledge_base(pdf_path):
    kb = PDFFileKnowledgeBase(
        files=[pdf_path],
        vector_db=PgVector(table_name="synthetic_customers", db_url=DB_URL, search_type=SearchType.hybrid),
    )
    kb.load(upsert=True)
    return kb

# Initialize Phi Agent
@st.cache_resource
def init_agent(knowledge_base):
    return Agent(
        model=Gemini(id="gemini-2.0-pro-vision"),  # Multimodal model
        knowledge=knowledge_base,
        search_knowledge=True,
        read_chat_history=True,
        show_tool_calls=True,
        markdown=True
    )

# OCR Function for Image Processing
def extract_text_from_image(image):
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text

# Function to generate final form using OCR & RAG
def generate_final_form(agent, ocr_text, rag_query):
    combined_prompt = f"""
    Use the extracted form data and customer knowledge to generate the final bank account opening form:
    - **Extracted Form Data (OCR):** {ocr_text}
    - **Customer Data from Knowledge Base:** {rag_query}
    """
    response = agent.run(combined_prompt)
    return response.content if response else "Unable to generate form."

# Streamlit App UI
st.set_page_config(page_title="Bank Form Auto-Filler", layout="wide")
st.title("🏦 Bank Account Registration Form Auto-Filler with OCR + RAG")

tab1, tab2 = st.tabs(["📄 Upload Synthetic Data (PDF)", "🖼️ Upload Bank Form (Image)"])

# TAB 1: PDF Upload for Synthetic Data
with tab1:
    uploaded_pdf = st.file_uploader("Upload Synthetic Data PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_path = f"/tmp/{uploaded_pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        
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
    uploaded_image = st.file_uploader("Upload Bank Form Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Bank Form", use_column_width=True)

        # Extract text using OCR
        if st.button("Extract Fields from Image (OCR)"):
            with st.spinner("Extracting text from image..."):
                ocr_text = extract_text_from_image(uploaded_image)
            st.text_area("Extracted Text:", ocr_text, height=200)

        # Combine OCR + RAG for final form generation
        if st.button("Generate Final Form (OCR + RAG)"):
            if uploaded_pdf:
                with st.spinner("Combining OCR and RAG to generate the final form..."):
                    final_output = generate_final_form(agent, ocr_text, query)
                st.markdown("### ✅ Final Auto-Filled Form:")
                st.markdown(final_output)
            else:
                st.warning("Please upload the synthetic data PDF in Tab 1 first.")
