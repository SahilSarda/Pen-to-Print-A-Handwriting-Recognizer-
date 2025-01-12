from flask import Flask, request, render_template, send_file
import cv2
from paddleocr import PaddleOCR  # Use PaddleOCR for better OCR
from fpdf import FPDF
from docx import Document
import os

app = Flask(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Using English language model with angle classification

@app.route('/landing', methods=['GET'])
def landing():
    return render_template('landing.html')

# --- Image Preprocessing ---
def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy."""
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to improve OCR performance
    new_width = 1000
    height, width = gray.shape
    new_height = int((new_width / width) * height)
    img_resized = cv2.resize(gray, (new_width, new_height))

    # Enhance contrast with adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_equalized = clahe.apply(img_resized)

    # Apply thresholding to binarize the image
    _, img_bin = cv2.threshold(img_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_bin

# --- Text Extraction with PaddleOCR ---
# --- Text Extraction with PaddleOCR (Line-by-Line) ---
def extract_text_with_paddleocr(image_path):
    """Extract text from the image using PaddleOCR, grouped line-by-line from left to right."""
    result = ocr.ocr(image_path, cls=True)  # cls=True for angle classification
    
    # Group lines by their vertical positions
    lines = []
    for line in result[0]:
        box = line[0]  # Get the bounding box coordinates
        text = line[1][0]  # Extracted text
        x_min = min([point[0] for point in box])
        y_min = min([point[1] for point in box])
        lines.append((y_min, x_min, text))  # Append (y_min, x_min, text) for sorting
    
    # Sort lines first by y_min (top to bottom), then by x_min (left to right)
    lines.sort(key=lambda x: (x[0], x[1]))
    
    # Join the sorted text line-by-line
    grouped_text = []
    current_y = None
    current_line = []

    for y_min, x_min, text in lines:
        # If the y_min difference indicates a new line, save the current line and start a new one
        if current_y is None or abs(y_min - current_y) > 10:  # Adjust the threshold if necessary
            if current_line:
                grouped_text.append(' '.join(current_line))
            current_line = [text]
            current_y = y_min
        else:
            current_line.append(text)
    
    # Append the last line
    if current_line:
        grouped_text.append(' '.join(current_line))
    
    # Combine all lines into a final text block with line breaks
    final_text = '\n'.join(grouped_text)
    return final_text.strip()


# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_and_extract():
    extracted_text = None
    if request.method == 'POST':
        uploaded_file = request.files['image_file']
        if uploaded_file:
            image_path = 'uploaded_image.jpg'
            uploaded_file.save(image_path)

            # Preprocess the image
            preprocessed_img = preprocess_image(image_path)

            # Save the preprocessed image for debugging purposes (optional)
            cv2.imwrite("processed_image.jpg", preprocessed_img)

            # Extract text from the preprocessed image using PaddleOCR
            extracted_text = extract_text_with_paddleocr(image_path)

    return render_template('upload_form.html', text=extracted_text)

@app.route('/save', methods=['POST'])
def save_text():
    text = request.form['extracted_text']
    # Save the extracted text to a file
    with open('extracted_text.txt', 'w') as f:
        f.write(text)
    return "Text saved successfully!"

@app.route('/convert', methods=['POST'])
def convert_to_file():
    text = request.form['extracted_text']
    format_type = request.form['format']
    
    if not text:
        return "No text to convert!", 400  # Error if text is empty

    # Define the output path
    output_dir = 'static/output_files/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert text to PDF
    if format_type == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf_output = os.path.join(output_dir, "output.pdf")
        pdf.output(pdf_output)
        return send_file(pdf_output, as_attachment=True, download_name="extracted_text.pdf")

    # Convert text to Word
    if format_type == 'word':
        doc = Document()
        doc.add_paragraph(text)
        doc_output = os.path.join(output_dir, "output.docx")
        doc.save(doc_output)
        return send_file(doc_output, as_attachment=True, download_name="extracted_text.docx")

    return "Invalid format selected.", 400




if __name__ == '__main__':
    app.run(debug=True)
