
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.pdf import PDFFileKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
import pytesseract
from PIL import Image
import os

# Define system prompt and instructions

SYSTEM_PROMPTS="""*You are a formfilling agent which can help in filling the uploaded blank form by doing RAG on the pdf doc containing customers data and  use your visual capabilities and identify patterns & text(OCR) and fill in the form based on data made available in uploaded pdf ."""
INSTRUCTIONS=""" 
*Do a RAG(Retrival Augmented Generation) on uploaded pdf .
*Identify text,patterns and parse(OCR) information from uploaded blank form.
*Combine RAG and OCR and fill the form accordingly.
*Generate the final filled output form."""

        
# Set API keys from Streamlit secrets

os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_KEY']


# OCR Function for Image Processing
def extract_text_from_image(image):
    img = Image.open(image)
    ocr_text = pytesseract.image_to_string(img)
    return ocr_text

# PostgreSQL connection
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"


# Initialize Knowledge Base
@st.cache_resource

def init_knowledge_base(pdf_path):
    kb = PDFFileKnowledgeBase(
        files=[pdf_path],
        vector_db=PgVector(table_name="customers", db_url=DB_URL, search_type=SearchType.hybrid),
    )
    kb.load(upsert=True)
    return kb

# Initialize Phi Agent
@st.cache_resource
def init_agent(kb):

    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        system_prompt=SYSTEM_PROMPTS,
        instructions=INSTRUCTIONS,
        knowledge=kb,
        search_knowledge=True,
        read_chat_history=True,
        show_tool_calls=True,
        markdown=True
    )


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
if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

# PostgreSQL connection
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"


tab1, tab2 = st.tabs(["📄 Upload Synthetic Data (PDF)", "🖼️ Upload Bank Form (Image)"])

# TAB 1: PDF Upload for Synthetic Data
with tab1:
    uploaded_pdf = st.file_uploader("Upload Customers Data PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_path = f"//tmp/{uploaded_pdf.name}"
        
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
                st.warning("Please upload the customer data PDF in Tab 1 first.")
