import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz  # PyMuPDF for PDF extraction
import uvicorn

app = FastAPI(
    title="Mini-Unstructured API",
    description="An API to extract, chunk, and structure PDF data for LLMs.",
    version="1.0.0"
)


def chunk_text(text: str, max_words: int = 500) -> list[str]:
    """
    A basic chunking function that splits text into blocks of a maximum word count.
    In a production app, you would upgrade this to semantic or recursive chunking.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    # Validate that the uploaded file is a PDF
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        # Read the file into memory
        file_bytes = await file.read()

        # Open the PDF using PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted_text = ""

        # Iterate through pages and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text")

        # Clean up the document object
        doc.close()

        # Pass the extracted text to our chunking algorithm
        chunks = chunk_text(extracted_text, max_words=500)

        # Return the structured JSON response
        return {
            "metadata": {
                "filename": file.filename,
                "total_pages": len(doc),
                "total_chunks": len(chunks)
            },
            "documents": [
                {
                    "type": "NarrativeText",
                    "chunk_id": i + 1,
                    "text": chunk
                }
                for i, chunk in enumerate(chunks)
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
