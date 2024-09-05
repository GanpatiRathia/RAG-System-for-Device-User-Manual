# RAG System for Device User Manual

## Overview

This project is a Retrieval-Augmented Generation (RAG) system that answers questions about a device user manual based on the contents of a PDF file. The system uses a combination of Sentence-BERT for text embeddings, FAISS for similarity search, and Hugging Face's `text-generation` pipeline to generate answers.

## Features

- **PDF Text Extraction:** Extracts text from a provided PDF file.
- **Text Embeddings:** Uses Sentence-BERT to create embeddings for text segments.
- **Similarity Search:** Retrieves relevant text segments based on user queries using FAISS.
- **Text Generation:** Generates responses based on the retrieved text using a Hugging Face model.

## Requirements

Make sure to have the following Python packages installed:

- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `transformers`
- `PyMuPDF` (for PDF text extraction)

You can install these packages using pip:

## Setup

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/GanpatiRathia/RAG-System-for-Device-User-Manual.git
   cd <repository_directory>

2. **Install the necessary Python packages:**

```bash
pip install streamlit sentence-transformers faiss-cpu transformers PyMuPDF
```

3. **Add Your PDF**

Place your PDF file (knowledge_doc.pdf) in the same directory as the app.py script.

4. **Update the Script**

Ensure the path to the PDF file is correctly set in the script. Update the extract_text_from_pdf function if necessary to match the file path:

```python
pdf_path = 'knowledge_doc.pdf'  # Path to your PDF file
```

## Usage

1. **Run the Streamlit App**

Start the Streamlit app with the following command:

```bash
streamlit run app.py
```

2. **Interact with the App**

Open the provided local URL (e.g., http://localhost:8501) in your web browser.

Enter your query into the text input field and click "Get Answer".

The app will display the most relevant section from the PDF and generate an answer based on the content.

## Code Explanation

1. **Text Extraction**

The extract_text_from_pdf(pdf_path) function extracts text from each page of the provided PDF file using the PyMuPDF library.

```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text.append(page.get_text())
    return text
```

2. **Text Embedding**

The generate_embeddings(segments) function creates embeddings for the text segments using the Sentence-BERT model.

```python
def generate_embeddings(segments):
    global embeddings, index
    embeddings = model.encode(segments)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
```

3. **Similarity Search**

The retrieve_relevant_sections(query) function retrieves the most relevant text segment for the user's query using FAISS.

```python
def retrieve_relevant_sections(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=1)
    return segments[indices[0][0]]
```

4. **Text Generation**

The generate_answer(prompt) function uses Hugging Face's text-generation pipeline to generate an answer based on the retrieved text.

```python
def generate_answer(prompt):
    generator = pipeline('text-generation', model='gpt2')
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]['generated_text'].strip()
```

## Troubleshooting

1. **PDF Text Extraction Issues**

Ensure the PDF is not encrypted or scanned.

If text extraction is not accurate, preprocess or clean the text data as needed.

2. **Model Errors**

Confirm that the specified models are available and correctly named in the Hugging Face model hub.

Update model names if needed based on the latest available models on Hugging Face.

3. **Library Errors**

Verify that all required Python packages are installed and up-to-date.

If encountering issues with package versions, consider creating a virtual environment and reinstalling dependencies.


## Future Work

Can use OpenAI's API,ngrok,and other pretrained models for better result with further training on user manual. Better UI can be created using Django/Flask for production grade application. 