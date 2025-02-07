

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
#def knowledge_base(pdf_path):
kb = PDFKnowledgeBase(
     path=pdf_path,
     vector_db=Qdrant(collection="formfill phidata-qdrant-ipynb"),
     url=os.environ['QDRANT_URL'],
     api_key=os.environ['QDRANT_API_KEY'],
     )
kb.load(upsert=True)
   # return kb

def Rag_agent(knowledge_base):
    agent = Agent(
        name="Rag Agent",
        role="Do RAG on pdf doc by fetching from knowledge base(kb)",
        model=Gemini(id="gemini-1.5-flash"),
        knowledge= kb,
        search_knowledge=True,
        read_chat_history=True,
        show_tool_calls=True,
        markdown=True,
        instructions=["Do a RAG on pdf document present in knowledge base(kb)"],
    )
    #return agent

def ocr(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

OCR_agent = Agent(
    name="OCR Agent",
    role="Identify text from an uploaded image having blank form and do an OCR",
    model=Gemini(id="gemini-1.5-flash"),
    tools=([ocr]), 
    instructions=["Do an OCR on uploaded blank form image and extract relevant text"],
    show_tool_calls=True,
    markdown=True,
)

def generate_form(agent, ocr_text, query):
    combined_prompt = f"""
    Use the extracted form data (OCR) and the knowledge base to generate the final bank form:
    - Extracted Form Data (OCR): {ocr_text}
    - Data to be filled (from query): {query}
    """
    output = agent.run(combined_prompt)
    return output.content if output else "Unable to generate form."

# Initialize Formfilling Agent
Formfilling_agent = Agent(
    name="Formfilling Agent",
    role="Agent has to fill form by combining OCR and RAG",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[generate_form],
    instructions=[
        "Generate a filled form by matching fields from the blank form and data from the knowledge base according to RAG query. eg: 'fill bank form for john'"
    ],
    show_tool_calls=True,
    markdown=True,
)




# Initialize Agent Team
agent_team = Agent(
    team=[Rag_agent, OCR_agent, Formfilling_agent],
    instructions=[
        "Facilitate collaboration & orchestration between all the agents and help in filling form accurately and efficiently"
    ],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit App UI with Tabs ---
st.set_page_config(page_title="Form Auto-Filler", layout="wide")
st.title("🏦  Form Auto-Filler with OCR + RAG")

# Create tabs for multi-agent functionality
tab1, tab2, tab3 = st.tabs(["📄 Upload  Data (PDF)", "🖼️ Upload Blank Form (Image)", "🤖 Multi-Agent Form Filling"])

# TAB 1: PDF Upload for Synthetic Data
with tab1:
    uploaded_pdf = st.file_uploader("Upload Synthetic Data PDF", type=["pdf"])
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
    uploaded_image = st.file_uploader("Upload Bank Form Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Bank Form", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image: 
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

# TAB 3: Multi-Agent Form Filling
with tab3:
    query = st.text_input("Enter your query to auto-fill the form (e.g., 'fill the form for customer john'):", key="multi_agent_query") 
    
    if st.button("Fill Form with Multi-Agent", key="multi_agent_button"):
        if query:
            with st.spinner("Processing with Multi-Agent..."):
                response = agent_team.print_response(query, stream=True)
                st.write(response)  
        else:
            st.warning("Please enter a query.")
