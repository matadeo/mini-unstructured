import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz  # PyMuPDF for PDF extraction
import uvicorn
from mapper import map_text_to_mirakl  # <-- New Import!

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
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        # Read the file into memory
        file_bytes = await file.read()
        extracted_text = ""

        # Use a context manager to safely open and automatically close the PDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            # Save the total pages to a variable BEFORE the document closes
            total_pages = len(doc)

            # Iterate through pages and extract text
            for page in doc:
                extracted_text += page.get_text("text")

        # Pass the extracted text to our chunking algorithm
        chunks = chunk_text(extracted_text, max_words=500)

        # Return the structured JSON response
        return {
            "metadata": {
                "filename": file.filename,
                "total_pages": total_pages,  # Using our saved variable here!
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


# <-- NEW ENDPOINT HERE -->
@app.post("/upload-to-mirakl/")
async def upload_to_mirakl(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        # 1. Read and Extract Text (Our Unstructured Engine)
        file_bytes = await file.read()
        extracted_text = ""

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            # We are limiting to the first 3 pages to avoid overloading the LLM during testing
            limit = min(3, len(doc))
            for page_num in range(limit):
                extracted_text += doc.load_page(page_num).get_text("text")

        # 2. Map the extracted text to the Mirakl Schema (Our GCP Vertex AI Engine)
        mirakl_json = map_text_to_mirakl(extracted_text)

        # 3. Return the final structured data
        return {
            "status": "success",
            "source_file": file.filename,
            "mirakl_listing": mirakl_json
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
