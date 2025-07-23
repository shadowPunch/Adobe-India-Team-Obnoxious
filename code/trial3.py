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
    tolerance = page_width * 0.20 # Increased tolerance for better matching
    return abs(block_center - page_center) < tolerance

def extract_pdf_structure(pdf_path):
    """
    Extracts a structured outline from a PDF, including the text content under each heading.
    This version uses a multi-pass approach for accuracy.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        dict: A dictionary containing the document title and a hierarchical outline with content.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}")
        return None

    # --- Step 1: Font & Style Analysis ---
    font_styles, body_size = get_font_styles(doc)
    if body_size == 0:
        print(f"Could not determine font styles for {pdf_path}.")
        return None

    size_hierarchy = sorted([s[0] for s in font_styles if s[0] > body_size], reverse=True)
    
    H1_SIZE = size_hierarchy[0] if len(size_hierarchy) > 0 else body_size * 1.5
    H2_SIZE = size_hierarchy[1] if len(size_hierarchy) > 1 else body_size * 1.25

    # --- Pass 1: Classify and Merge Headings ---
    classified_blocks = []
    for page_num, page in enumerate(doc):
        # Removed the incompatible flag from the get_text call
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    # Check if the line itself might be a heading
                    line_text = "".join([s['text'] for s in l['spans']]).strip()
                    if not line_text: continue

                    first_span = l['spans'][0]
                    font_size = round(first_span["size"])
                    is_bold = "bold" in first_span["font"].lower()
                    rect = fitz.Rect(l['bbox'])
                    
                    heading_level = None
                    
                    # --- Classification Logic ---
                    if page_num == 0 and font_size > H1_SIZE and is_centered(rect, page.rect.width):
                         heading_level = "Title"
                    elif any(line_text.lower().strip() == ch for ch in ["abstract", "introduction", "method", "discussion", "conclusion", "references", "acknowledgments"]):
                        heading_level = "H1"
                    elif re.match(r"^(table|figure)\s+\d+", line_text.lower()):
                        heading_level = "H2"
                    elif font_size >= H1_SIZE and is_bold:
                        heading_level = "H1"
                    elif font_size >= H2_SIZE and is_bold:
                        heading_level = "H2"
                    elif font_size > body_size and is_bold and len(line_text.split()) < 10:
                        heading_level = "H3"

                    if heading_level:
                        classified_blocks.append({
                            "level": heading_level, "text": line_text, "page": page_num + 1,
                            "bbox": fitz.Rect(l['bbox'])
                        })

    # --- Pass 2: Merge multi-line headings ---
    merged_outline = []
    if classified_blocks:
        current_block = classified_blocks[0]
        for i in range(1, len(classified_blocks)):
            next_block = classified_blocks[i]
            if (next_block["level"] == current_block["level"] and
                next_block["page"] == current_block["page"] and
                abs(next_block["bbox"].y0 - current_block["bbox"].y1) < 20):
                current_block["text"] += " " + next_block["text"]
                current_block["bbox"] |= next_block["bbox"]
            else:
                merged_outline.append(current_block)
                current_block = next_block
        merged_outline.append(current_block)

    # --- Step 3: Extract Content for each Heading ---
    final_outline = []
    potential_title = ""
    title_candidates = [b for b in merged_outline if b['level'] == 'Title']
    if title_candidates:
        potential_title = max(title_candidates, key=lambda b: b['bbox'].width * b['bbox'].height)['text']
    else:
        potential_title = os.path.basename(pdf_path)

    # Sort all headings by page and vertical position
    merged_outline.sort(key=lambda b: (b['page'], b['bbox'].y0))

    for i, heading in enumerate(merged_outline):
        if heading['level'] == 'Title' or heading['text'].lower() == potential_title.lower():
            continue

        # Define the content area: from the bottom of the current heading
        # to the top of the next one.
        start_page = heading['page'] - 1
        end_page = doc.page_count
        
        content_rect = fitz.Rect(
            0, heading['bbox'].y1,
            doc[start_page].rect.width, doc[start_page].rect.height
        )
        
        next_heading_rect = fitz.Rect(0, 0, 0, 0)
        if i + 1 < len(merged_outline):
            next_heading = merged_outline[i+1]
            if next_heading['page'] == heading['page']:
                # Next heading is on the same page
                next_heading_rect.y0 = next_heading['bbox'].y0
                content_rect.y1 = next_heading['bbox'].y0
            else:
                # Next heading is on a different page
                end_page = next_heading['page'] -1

        # Extract text from the defined content area
        content_text = ""
        for page_num in range(start_page, end_page):
             page = doc[page_num]
             # For the first page of the section, clip from the heading downwards
             clip_rect = content_rect if page_num == start_page else page.rect
             content_text += page.get_text(clip=clip_rect).strip() + " "
        
        # For sections spanning multiple pages, handle the last page
        if end_page > start_page and end_page < doc.page_count:
             page = doc[end_page]
             clip_rect = fitz.Rect(0, 0, page.rect.width, next_heading_rect.y0)
             content_text += page.get_text(clip=clip_rect).strip()

        final_outline.append({
            "level": heading["level"],
            "text": re.sub(r'\s+', ' ', heading['text']).strip(),
            "page": heading["page"],
            "content": re.sub(r'\s+', ' ', content_text).strip()
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

