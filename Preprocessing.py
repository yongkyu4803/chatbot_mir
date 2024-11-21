import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define a function to extract text from a single PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Ensure the page has text
            text += page_text
    return text

# Define a function to clean the extracted text
def clean_text(text):
    # Remove multiple newlines and normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# Define a function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Define a function to process all PDFs in a folder
def preprocess_pdfs(input_folder, output_folder, chunk_size=1000, chunk_overlap=100):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all PDF files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.pdf'):  # Check if the file is a PDF
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")

            # Extract text
            raw_text = extract_text_from_pdf(file_path)

            # Clean the text
            cleaned_text = clean_text(raw_text)

            # Split the text into chunks
            chunks = split_text(cleaned_text, chunk_size, chunk_overlap)

            # Save the chunks to individual files
            base_name = os.path.splitext(file_name)[0]
            for i, chunk in enumerate(chunks):
                chunk_file_path = os.path.join(output_folder, f"{base_name}_chunk_{i+1}.txt")
                with open(chunk_file_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

    print(f"Preprocessing complete. Processed files are saved in '{output_folder}'.")

# Input and output folder paths
input_folder = r'C:\Users...'  # Replace with your input folder path
output_folder = r'C:\Users....'  # Replace with your desired output folder path

# Run the preprocessing
preprocess_pdfs(input_folder, output_folder)
