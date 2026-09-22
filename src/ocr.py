from PIL import Image
import pytesseract


def extract_text_from_image(image):
    """
    Extract text from a newspaper/news image using Tesseract OCR.
    """
    text = pytesseract.image_to_string(image)

    return text.strip()
