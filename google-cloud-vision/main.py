import io
import os
from google.cloud import vision
from lxml import etree

# Google Cloud Vision APIのクライアントを作成
client = vision.ImageAnnotatorClient()


def detect_text(image_path):
    """Detects text in an image and returns full text and individual words with their bounding boxes."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f'{response.error.message}')

    full_text = response.full_text_annotation.text
    words = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    bounding_box = word.bounding_box
                    words.append({
                        'text': word_text,
                        'bounding_box': bounding_box
                    })

    return full_text, words


def parse_label_xml(xml_path):
    """Parses the label XML and returns structured data."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    labels = []
    for label in root.findall('.//object'):
        name = label.find('name').text
        bndbox = label.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        labels.append({
            'name': name,
            'coordinates': (xmin, ymin, xmax, ymax)
        })

    return labels


def is_within_bounding_box(bounding_box, xmin, ymin, xmax, ymax):
    """Check if a bounding box is within the given coordinates."""
    vertices = bounding_box.vertices
    return (
            xmin <= vertices[0].x <= xmax and ymin <= vertices[0].y <= ymax and
            xmin <= vertices[2].x <= xmax and ymin <= vertices[2].y <= ymax
    )


def process_image(image_path, xml_path):
    # XMLファイルからラベル情報を読み込む
    labels = parse_label_xml(xml_path)

    # OCRの実行
    full_text, words = detect_text(image_path)

    print("Extracted text:", full_text)

    # ラベル情報と照合
    for label in labels:
        print(f"\nChecking for label: {label['name']}")
        relevant_texts = []
        for word in words:
            if is_within_bounding_box(word['bounding_box'], *label['coordinates']):
                relevant_texts.append(word['text'])

        if relevant_texts:
            print(f"Detected {label['name']} within bounding box: {' '.join(relevant_texts)}")
        else:
            print(f"No text found within bounding box for {label['name']}.")


if __name__ == "__main__":
    image_path = "/app/images/sample-image3.png"  # 画像ファイルのパス
    xml_path = "/app/labels/sample-image3.xml"  # ラベル情報のXMLファイルのパス
    process_image(image_path, xml_path)
