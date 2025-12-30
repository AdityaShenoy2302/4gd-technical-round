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
import io

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ----------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize OCR ONCE
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# ---------- OCR ----------
def extract_text_from_pdf(pdf_path):
    """
    Convert PDF to images using PyMuPDF, then use OCR
    No poppler needed!
    """
    try:
        print(f"🔄 Opening PDF with PyMuPDF...")

        # Open PDF
        pdf_document = fitz.open(pdf_path)
        extracted_text = ""

        print(f"📄 PDF has {len(pdf_document)} page(s)")

        for page_num in range(len(pdf_document)):
            print(f"🔄 Processing page {page_num + 1}/{len(pdf_document)}")

            # Get page
            page = pdf_document[page_num]

            # Convert page to image (higher resolution = better OCR)
            mat = fitz.Matrix(3.0, 3.0)  # 3x zoom = 300 DPI equivalent
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to numpy array for PaddleOCR
            img_array = np.array(img)

            # Run OCR (updated API - no cls parameter)
            result = ocr.ocr(img_array)

            if result is None or len(result) == 0:
                print(f"⚠️ No text found on page {page_num + 1}")
                continue

            # Extract text from OCR results
            # Result structure: [[[bbox, (text, confidence)], ...]]
            for line in result[0]:
                if line and len(line) > 1:
                    # line[1] is a tuple of (text, confidence)
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        extracted_text += text + " "
                        print(f"  📝 {text} (confidence: {confidence:.2f})")
                    elif isinstance(line[1], str):
                        # Sometimes it's just a string
                        text = line[1]
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
You are an invoice data extraction system. Extract the following fields from this invoice text and return ONLY a valid JSON object.

Required fields (use null if not found):
- invoiceNumber: The invoice number (look for "Invoice No:", "Invoice Number", etc.)
- poNumber: Purchase order number (look for "PO Number", "Order ID", etc.)
- supplierName: Name of the supplier/restaurant/company (look for "Restaurant Name", "Supplier", company names)
- totalAmount: Total invoice amount (look for "Invoice Total", "Total Amount", final total)
- discount: Any discount amount
- taxAmount: Total tax amount (look for "Total taxes", sum of all taxes)
- vat: VAT amount
- sgst: SGST amount (State GST)
- cgst: CGST amount (Central GST)
- igst: IGST amount (Integrated GST)
- invoiceAmount: Final payable amount (same as totalAmount usually)

Invoice text:
{text}

Return ONLY the JSON object, no explanations, no markdown formatting:
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You are a precise invoice data extractor. Return only valid JSON without any markdown formatting or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        print(f"\n🤖 LLM Raw Response:\n{content}\n")

        # Clean up markdown code blocks more aggressively
        content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        content = content.strip()

        # Remove any remaining backticks
        content = content.replace('`', '')

        print(f"🧹 Cleaned content:\n{content}\n")

        parsed_json = json.loads(content)
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