import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import fitz  # PyMuPDF

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Placeholder for knowledge base embeddings and segments
embeddings = None
segments = None
index = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text())
    return text

# Function to embed the segments
def generate_embeddings(segments):
    global embeddings, index
    embeddings = model.encode(segments)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# Function to retrieve relevant segments
def retrieve_relevant_sections(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=1)
    return segments[indices[0][0]]

# Function to generate answers using Hugging Face model
def generate_answer(prompt):
    generator = pipeline('text-generation', model='gpt2')  # Use 'gpt2' or any other available model
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text'].strip()

# Load the manual and embed the segments
pdf_path = 'knowledge_doc.pdf'  # Path to your PDF file
segments = extract_text_from_pdf(pdf_path)
generate_embeddings(segments)

# Streamlit App Code
def app():
    st.title("RAG System for Device User Manual")
    st.write("Ask any questions about the user manual and get answers based on the manual content.")

    # Input from the user
    user_query = st.text_input("Enter your query here:")

    if st.button("Get Answer"):
        if user_query:
            # Retrieve the most relevant section
            relevant_section = retrieve_relevant_sections(user_query)
            st.write(f"Relevant Section: {relevant_section}")

            # Generate an answer based on the retrieved section
            prompt = f"Based on the following information, answer the question:\n\n{relevant_section}\n\nQuestion: {user_query}\nAnswer:"
            answer = generate_answer(prompt)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please enter a valid query.")

# Run the Streamlit app
if __name__ == '__main__':
    app()