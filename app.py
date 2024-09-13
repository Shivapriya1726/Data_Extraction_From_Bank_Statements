import os
import io
import sys
import json
import cv2
import numpy as np
from PIL import Image
import fitz  
from docx import Document
import streamlit as st

try:
    from easyocr import Reader
except ImportError:
    print("EasyOCR is not installed. Please install it using 'pip install easyocr'")
    sys.exit(1)
import google.generativeai as genai

genai.configure(api_key="") # Replace your Gemini API key
model = genai.GenerativeModel('gemini-1.5-flash')

generation_config = {
    "temperature": 0.7
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class ImageConversionError(Exception):
    """Custom exception for image conversion errors."""
    pass


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


class OCRProcessingError(Exception):
    """Custom exception for OCR processing errors."""
    pass


class ImageProcessor:

    def __init__(self):
        self.reader = Reader(['fr'])  

    def convert_to_image(self, input_path):
        """
        Converts various input formats (PDF, image, Word) to a single PIL Image object.
        """
        try:
            if input_path.lower().endswith(".pdf"):
                doc = fitz.open(input_path)
                pages = []
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    pixmap = page.get_pixmap()
                    img = Image.open(io.BytesIO(pixmap.tobytes()))
                    pages.append(img)
                doc.close()

                widths, heights = zip(*(i.size for i in pages))
                total_height = sum(heights)
                max_width = max(widths)
                new_im = Image.new('RGB', (max_width, total_height))
                y_offset = 0
                for im in pages:
                    new_im.paste(im, (0, y_offset))
                    y_offset += im.size[1]
                return new_im

            elif input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                return Image.open(input_path)

            elif input_path.lower().endswith(".docx"):
                document = Document(input_path)
                temp_images = []
                for i, paragraph in enumerate(document.paragraphs):
                    if 'image' in paragraph._p.xpath("./w:drawing/wp:inline/a:graphic"):
                        for rId in paragraph._p.xpath("./w:drawing/wp:inline/a:graphic/a:graphicData/pic:pic/pic:blipFill/a:blip/@r:embed"):
                            image_part = document.part.related_parts[rId]
                            image = Image.open(io.BytesIO(image_part.blob))
                            temp_images.append(image)
                widths, heights = zip(*(i.size for i in temp_images))
                total_height = sum(heights)
                max_width = max(widths)
                new_im = Image.new('RGB', (max_width, total_height))
                y_offset = 0
                for im in temp_images:
                    new_im.paste(im, (0, y_offset))
                    y_offset += im.size[1]
                return new_im

            else:
                raise ImageConversionError(f"Unsupported file format: {input_path}")

        except Exception as e:
            raise ImageConversionError(f"Error converting file: {e}")

    def preprocess_image(self, image):
        """
        Performs image preprocessing for scanned images, focusing on adaptive thresholding
        to enhance black text against a white background.
        """
        try:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)

            alpha = 0.7  
            sharpened = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0) 

            return Image.fromarray(sharpened) 

        except Exception as e:
            raise ImageProcessingError(f"Error preprocessing image: {e}")

    def perform_ocr(self, image):
        """
        Performs OCR using EasyOCR, returns JSON with all text, 
        and an annotated image with text above bounding boxes.
        """
        try:
            results = self.reader.readtext(np.array(image))
            ocr_output = []
            annotated_image = np.array(image.copy())
            image = Image.fromarray(annotated_image)
            for (bbox, text, confidence) in results:
                ocr_output.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": [[int(coord) for coord in point] for point in bbox]
                })

                

            all_text = " ".join([item['text'] for item in ocr_output])
            response = model.generate_content([f"""
            I have provided you with a image of a bank statement in french and also here is the ocr extracted text {all_text} of the image, I want you to provide me with a well formatted json with all the information in the image. Remember do not leave any word in the image, and try to preserve the layouts and give a structured json of all the information in the image..
            """, image], safety_settings=safety_settings, generation_config=generation_config)     
            print(response)
            response_text = response.text 
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            response = response_text[start_index:end_index]
            return {"all_text": all_text, "OCR_andLLM": response}, annotated_image

        except Exception as e:
            raise OCRProcessingError(f"Error performing OCR: {e}")


def main():
    st.title("OCR Application")
    st.write("Upload a file (PDF or Image), and extract the text using OCR.")


    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        processor = ImageProcessor()

        try:
            
            input_path = uploaded_file.name
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            image = processor.convert_to_image(input_path)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            preprocessed_image = processor.preprocess_image(image)

            ocr_json, annotated_image = processor.perform_ocr(preprocessed_image)

            st.write("Extracted Text:")
            st.json(ocr_json)

            st.download_button(
                label="Download OCR Result as JSON",
                data=json.dumps(ocr_json, indent=4),
                file_name="ocr_result.json",
                mime="application/json"
            )

        except (ImageConversionError, ImageProcessingError, OCRProcessingError) as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
