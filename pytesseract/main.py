import cv2
import pytesseract
import sys

print("Starting OCR process...")

try:
    # 画像ファイルのパスを取得
    image_path = sys.argv[1]

    # 画像の読み込み
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        sys.exit(1)

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像のリサイズ（1.5倍）
    height, width = gray.shape[:2]
    resized_image = cv2.resize(gray, (int(1.5 * width), int(1.5 * height)))

    # ガウシアンブラーでノイズを除去
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # ヒストグラム均等化でコントラストを改善（CLAHEを使用）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(blurred_image)

    # 二値化処理
    _, binary = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCRの実行（言語を日本語に設定、縦書き用のPSM 5を使用）
    custom_config = r'--oem 3 --psm 5'
    text = pytesseract.image_to_string(binary, config=custom_config, lang='jpn')

    print("Extracted text:", text)

    if "美容師免許証" in text:
        print("This is a barber license.")
    else:
        print("This is not a barber license.")
except Exception as e:
    print("An error occurred:", e)
