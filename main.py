# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llama-3.1-8b-instant
import os
import json
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
from groq import Groq
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

from dotenv import load_dotenv
load_dotenv()


# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- put your key here
# ----------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize OCR ONCE
ocr = PaddleOCR(use_textline_orientation=True, lang="en")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# ---------- OCR ----------
def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF - first try native text extraction, then OCR if needed
    """
    try:
        print(f"📄 Opening PDF with PyMuPDF...")
        pdf_document = fitz.open(pdf_path)
        extracted_text = ""

        print(f"📄 PDF has {len(pdf_document)} page(s)")

        for page_num in range(len(pdf_document)):
            print(f"📄 Processing page {page_num + 1}/{len(pdf_document)}")
            page = pdf_document[page_num]

            # First, try extracting native text
            native_text = page.get_text()

            if native_text and len(native_text.strip()) > 50:
                # PDF has text layer, use it directly
                print(f"  ✅ Using native text extraction")
                extracted_text += native_text + "\n"
            else:
                # No text layer, use OCR
                print(f"  🔍 No text layer found, using OCR...")

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)

                # Run OCR
                result = ocr.ocr(img_array)

                if result and len(result) > 0 and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                            extracted_text += text + " "
                            print(f"  📝 {text}")

        pdf_document.close()

        final_text = extracted_text.strip()
        print(f"\n✅ Total extracted text length: {len(final_text)} characters")
        print(f"📄 First 500 chars:\n{final_text[:500]}\n")

        return final_text

    except Exception as e:
        print(f"❌ OCR Error: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------- LLM ----------
def extract_invoice_json(text):
    prompt = f"""
Extract invoice data from this text and return ONLY a valid JSON object.

Fields to extract (use null if not found):
- invoiceNumber: Invoice number (e.g., "0022717122400018")
- poNumber: Purchase/Order ID (e.g., "225538462233034")
- supplierName: Restaurant/Supplier name (e.g., "Donne Biriyani House")
- totalAmount: Final invoice total (e.g., "177.66")
- discount: Total discount amount
- taxAmount: Total taxes (e.g., "8.46")
- vat: VAT amount if present
- sgst: SGST/UTGST amount (e.g., "4.23")
- cgst: CGST amount (e.g., "4.23")
- igst: IGST amount (e.g., "0.00")
- invoiceAmount: Same as totalAmount

Important:
- Extract ONLY numbers without currency symbols
- For tax fields, use the actual amount, not rate
- Return numbers as strings (e.g., "177.66" not 177.66)

Invoice text:
{text}

Return ONLY the JSON object:
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You are an invoice data extraction expert. Return only valid JSON without markdown formatting. Extract exact values from the text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        print(f"\n🤖 LLM Raw Response:\n{content}\n")

        # Remove markdown code blocks
        content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        content = content.strip().replace('`', '')

        print(f"🧹 Cleaned content:\n{content}\n")

        parsed_json = json.loads(content)

        # Ensure all expected fields exist
        expected_fields = ["invoiceNumber", "poNumber", "supplierName", "totalAmount",
                           "discount", "taxAmount", "vat", "sgst", "cgst", "igst", "invoiceAmount"]
        for field in expected_fields:
            if field not in parsed_json:
                parsed_json[field] = None

        print(f"✅ Successfully parsed JSON: {json.dumps(parsed_json, indent=2)}")
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw content that failed: {content}")
        raise ValueError(f"Failed to parse JSON from LLM response: {e}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------- PIPELINE ----------
def process_invoice(pdf_path):
    print(f"\n{'=' * 60}")
    print(f"📄 Starting processing: {pdf_path}")
    print(f"{'=' * 60}\n")

    text = extract_text_from_pdf(pdf_path)

    if not text or len(text) < 10:
        raise ValueError(f"Insufficient text extracted from OCR. Got only {len(text)} characters.")

    print(f"\n{'=' * 60}")
    print(f"🤖 Sending to LLM for extraction...")
    print(f"{'=' * 60}\n")

    invoice_json = extract_invoice_json(text)

    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete!")
    print(f"{'=' * 60}\n")

    return invoice_json


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        print(f"✅ File saved: {path}")

        data = process_invoice(path)

        return jsonify({"success": True, "data": data})

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)