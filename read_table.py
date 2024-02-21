import cv2
from pytesseract import image_to_string

# 1. Read the image file
image = cv2.imread("images/table_image.jpeg")

# Convert the image to grayscale
# Color complexities can hinder the OCR process.
# Converting to grayscale simplifies the image.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarization makes the image black and white, further reducing complexities.
_, binary_image = cv2.threshold(
    gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
# OBS: `THRESH_BINARY_INV` and `THRESH_OTSU` are thresholding techniques to
# automate the selection of the threshold value.

# Save image for verification
cv2.imwrite("images/processed_image.jpg", binary_image)

# 2. Text Extraction with Pytesseract:
extracted_text = image_to_string(binary_image)

print(extracted_text)
