import fitz  # PyMuPDF
import json
import os
import re
from collections import Counter, defaultdict

def get_font_styles(doc):
    """
    Analyzes the document to get statistics on font sizes and styles.
    This helps create dynamic thresholds instead of fixed ones.

    Args:
        doc: A fitz.Document object.

    Returns:
        A tuple containing:
        - A list of (size, count) for all font sizes found.
        - The most common font size (likely the body text).
    """
    styles = defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        styles[round(s["size"])] += 1
    
    if not styles:
        return [], 0

    sorted_styles = sorted(styles.items(), key=lambda item: item[1], reverse=True)
    body_size = sorted_styles[0][0] if sorted_styles else 0
    
    return sorted_styles, body_size

def is_centered(rect, page_width):
    """Check if a text block is roughly centered on the page."""
    block_center = (rect.x0 + rect.x1) / 2
    page_center = page_width / 2
    tolerance = page_width * 0.15 # Increased tolerance for better matching
    return abs(block_center - page_center) < tolerance

def extract_pdf_structure(pdf_path):
    """
    Extracts a structured outline from a PDF using a more robust two-pass method.
    Pass 1: Classify all text blocks.
    Pass 2: Merge consecutive blocks of the same heading level.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        dict: A dictionary containing the document title and a hierarchical outline.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}")
        return None

    # --- Step 1: Advanced Font & Style Analysis ---
    font_styles, body_size = get_font_styles(doc)
    if body_size == 0:
        print(f"Could not determine font styles for {pdf_path}.")
        return None

    size_hierarchy = sorted([s[0] for s in font_styles if s[0] > body_size], reverse=True)
    
    # Define thresholds dynamically
    H1_SIZE = size_hierarchy[0] if len(size_hierarchy) > 0 else body_size * 1.8
    H2_SIZE = size_hierarchy[1] if len(size_hierarchy) > 1 else body_size * 1.4
    H3_SIZE = size_hierarchy[2] if len(size_hierarchy) > 2 else body_size * 1.2

    # --- Pass 1: Classify all individual blocks ---
    classified_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0: # it's a text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if not text:
                            continue

                        font_size = round(s["size"])
                        is_bold = "bold" in s["font"].lower()
                        rect = fitz.Rect(s['bbox'])
                        
                        heading_level = None
                        # Title Heuristic: Very large, centered text on the first page
                        if page_num == 0 and font_size > H2_SIZE and is_centered(rect, page.rect.width):
                             heading_level = "Title"
                        # Keyword-based headings
                        elif any(re.match(f"^{ch}$", text.strip(), re.IGNORECASE) for ch in ["abstract", "introduction", "method", "discussion", "conclusion", "references"]):
                            heading_level = "H1"
                        # Style-based headings
                        elif font_size >= H1_SIZE:
                            heading_level = "H1"
                        elif font_size >= H2_SIZE:
                            heading_level = "H2"
                        elif font_size > body_size and is_bold:
                            heading_level = "H3"

                        if heading_level:
                            classified_blocks.append({
                                "level": heading_level,
                                "text": text,
                                "page": page_num + 1,
                                "bbox": s['bbox'] # Store bounding box for merging
                            })

    # --- Pass 2: Merge consecutive blocks of the same level ---
    if not classified_blocks:
        return {"title": os.path.basename(pdf_path), "outline": []}

    merged_outline = []
    current_block = classified_blocks[0]

    for i in range(1, len(classified_blocks)):
        next_block = classified_blocks[i]
        # Merge if same level, same page, and vertically close
        if (next_block["level"] == current_block["level"] and
            next_block["page"] == current_block["page"] and
            abs(next_block["bbox"][1] - current_block["bbox"][3]) < 15): # Vertical distance check
            current_block["text"] += " " + next_block["text"]
            # Update bbox to encompass both
            current_block["bbox"] = (
                min(current_block["bbox"][0], next_block["bbox"][0]),
                min(current_block["bbox"][1], next_block["bbox"][1]),
                max(current_block["bbox"][2], next_block["bbox"][2]),
                max(current_block["bbox"][3], next_block["bbox"][3])
            )
        else:
            merged_outline.append(current_block)
            current_block = next_block
    merged_outline.append(current_block) # Add the last block

    # --- Step 3: Assemble Final JSON ---
    final_outline = []
    potential_title = ""
    # Find the best candidate for the title from the merged blocks
    title_candidates = [b for b in merged_outline if b['level'] == 'Title']
    if title_candidates:
        # The one with the largest bbox area is likely the main title
        potential_title = max(title_candidates, key=lambda b: (b['bbox'][2]-b['bbox'][0])*(b['bbox'][3]-b['bbox'][1]))['text']

    if not potential_title:
        potential_title = os.path.basename(pdf_path)

    for block in merged_outline:
        # Don't add titles to the outline itself
        if block['level'] != 'Title' and block['text'].lower() != potential_title.lower():
            final_outline.append({
                "level": block["level"],
                "text": block["text"],
                "page": block["page"]
            })

    result = {
        "title": potential_title,
        "outline": final_outline
    }
    return result

if __name__ == '__main__':
    INPUT_DIR = "/home/necro/Desktop/Adobe/code/input"
    OUTPUT_DIR = "/home/necro/Desktop/Adobe/code/output2"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith(".pdf"):
                print(f"Processing {filename}...")
                pdf_path = os.path.join(INPUT_DIR, filename)
                structure_data = extract_pdf_structure(pdf_path)

                if structure_data:
                    output_filename = os.path.splitext(filename)[0] + ".json"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(structure_data, f, indent=4, ensure_ascii=False)
                    print(f"Successfully created {output_path}")
    else:
        print(f"Input directory '{INPUT_DIR}' not found. Please create it and add PDF files.")

