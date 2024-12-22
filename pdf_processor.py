import json
from typing import Dict, List, Tuple

import fitz
import yaml

# Load the project config file
with open("configs.yaml", "r", encoding="utf8") as file:
    project_configs = yaml.safe_load(file)

# number of characters that will be highlighted in the pdf
# source chunk will be smaller than the highlighted chunk (700 chars)
CHARS_PER_PASSAGE = project_configs["chars-per-passage"]


def extract_text_with_positions(doc) -> List[Tuple[str, int, fitz.Rect, int]]:
    """
    Extract text and position information from the PDF.
    Returns:
        List of tuples containing (text, page_number, text_rectangle, char_count)
    """
    text_positions = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if text.strip():  # Skip empty text
                            bbox = fitz.Rect(span["bbox"])
                            char_count = len(text)
                            text_positions.append((text, page_num, bbox, char_count))

    return text_positions


def create_passages(
    doc, text_positions: List[Tuple[str, int, fitz.Rect, int]]
) -> List[Dict]:
    """
    Create passages with PDF.js compatible annotations.
    """
    passages = []
    current_passage = ""
    current_positions = []
    char_count = 0
    for text, page_num, bbox, text_char_count in text_positions:
        new_char_count = char_count + text_char_count + 1
        # Get page dimensions for coordinate conversion
        page = doc[page_num]
        page_height = (
            page.rect.height
        )  # number of characters that will be highlighted in the pdf
        # source chunk will be smaller than the highlighted chunk (700 chars)
        if new_char_count > CHARS_PER_PASSAGE and current_passage:
            passage_dict = {
                "page_content": current_passage.strip(),
                "metadata": {
                    "source": doc.name,
                    "pageNum": page_num,
                    "title": "",
                    "pid": str(len(passages)),
                    "annotations": json.dumps(current_positions),
                },
                "type": "Document",
            }
            passages.append(passage_dict)
            # Convert coordinates for PDF.js
            current_passage = text + " "
            current_positions = [
                {
                    "page": page_num,
                    "x": bbox.x0,
                    "y": page_height - bbox.y1,  # Convert y-coordinate
                    "width": bbox.width,
                    "height": bbox.height,
                    "color": "#FFFF00",
                }
            ]
            char_count = text_char_count + 1
        else:
            current_passage += text + " "
            current_positions.append(
                {
                    "page": page_num,
                    "x": bbox.x0,
                    "y": page_height - bbox.y1,  # Convert y-coordinate
                    "width": bbox.width,
                    "height": bbox.height,
                    "color": "#FFFF00",
                }
            )
            char_count = new_char_count

    if current_passage.strip():
        passage_dict = {
            "page_content": current_passage.strip(),
            "metadata": {
                "source": doc.name,
                "pageNum": page_num,
                "title": "",
                "pid": str(len(passages)),
                "annotations": json.dumps(current_positions),
            },
            "type": "Document",
        }
        passages.append(passage_dict)

    return passages
