import cv2
import pytesseract 

img = cv2.imread("sample.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray,
    150,
    255,
    cv2.THRESH_BINARY
)

text = pytesseract.image_to_string(
    thresh,
    config="--psm 6"
)


      
