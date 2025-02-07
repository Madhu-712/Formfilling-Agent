

import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.qdrant import Qdrant
import pytesseract
from PIL import Image
import os

# Set API keys from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_KEY']
os.environ['QDRANT_API_KEY'] = st.secrets['api_key']
os.environ['QDRANT_URL'] = st.secrets['QDRANT_URL']

# Initialize Knowledge Base
def knowledge_base():
    kb = PDFKnowledgeBase(
        path="/content/customers.pdf",
        vector_db=Qdrant(collection="formfill phidata-qdrant-ipynb"),
        url=os.environ['QDRANT_URL'],
        api_key=os.environ['QDRANT_API_KEY'],
    )
    kb.load(upsert=True)
    return kb

# Initialize Rag Agent
Rag_agent = Agent(
    name="Rag Agent",
    role="Do RAG on pdf doc by fetching from knowledge base(kb)",
    model=Gemini(id="gemini-2.0-pro-vision"),
    knowledge=knowledge_base(),
    search_knowledge=True,
    read_chat_history=True,
    # Remove init_knowledge_base from tools as it's not a tool
    show_tool_calls=True,
    markdown=True,
    instructions=["Do a RAG on pdf document present in knowledge base(kb)"],
)

# OCR Function for Image Processing
def ocr(image):
    image = "/content/Image.jpg"
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text

# Initialize OCR Agent
OCR_agent = Agent(
    name="OCR Agent",
    role="Identify text from an uploaded image having blank form and do an OCR",
    model=Gemini(id="gemini-2.0-pro-vision"),
    tools=[ocr],
    instructions=["Do an OCR on uploaded blank form image and extract relevant text"],
    show_tool_calls=True,
    markdown=True,
)

# Function to generate final form using OCR & RAG
def generate_form(ocr_text, rag_query):
    # Assuming you have an 'agent' defined for this purpose
    combined_prompt = f"""
    Use the extracted form data and knowledge base to generate the final bank form:
    - **Extracted Form Data (OCR):** {ocr_text}
    - ** Data from Knowledge Base:** {rag_query}
    """
    # Assuming 'Formfilling_agent' is the agent to use here
    response = Formfilling_agent.run(combined_prompt) 
    return response.content if response else "Unable to generate form."

# Initialize Formfilling Agent
Formfilling_agent = Agent(
    name="Formfilling Agent",
    role="Agent has to fill form by combining OCR and RAG",
    model=Gemini(id="gemini-2.0-pro-vision"),
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

# Streamlit UI
st.title("Form Filling Agent")

# User input for query
query = st.text_input("Enter your query (e.g., 'fill the form for customer john'):")

# Button to trigger form filling
if st.button("Fill Form"):
    if query:
        with st.spinner("Processing..."):
            response = agent_team.print_response(query, stream=True)
            st.write(response)  # Display the response
    else:
        st.warning("Please enter a query.")
