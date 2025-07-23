import fitz  # PyMuPDF
import json
import os
import re
from collections import Counter

def get_font_statistics(doc):
    """
    Analyzes the entire document to find the most common font size and family.
    This is crucial for establishing a 'baseline' for normal text.

    Args:
        doc: A fitz.Document object.

    Returns:
        A tuple containing the most common font size and the most common font family.
    """
    sizes = []
    fonts = []
    for page in doc:
        # Extract text blocks with detailed information
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:  # Code for a text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        sizes.append(round(s["size"]))
                        fonts.append(s["font"])

    if not sizes:
        return 0, ""

    # Use Counter to find the most frequent size and font
    most_common_size = Counter(sizes).most_common(1)[0][0]
    most_common_font = Counter(fonts).most_common(1)[0][0]

    return most_common_size, most_common_font

def extract_pdf_structure(pdf_path):
    """
    Extracts a structured outline (Title, H1, H2, H3) from a PDF file.

    This function uses font size and style heuristics to identify headings.
    It establishes a base font size for the document's body text and then
    classifies headings based on how much larger their font size is.

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

    # --- Step 1: Establish Baseline Font ---
    # Get the most common font size and family to identify body text.
    base_size, base_font = get_font_statistics(doc)
    if base_size == 0:
        print(f"Could not determine base font size for {pdf_path}. Aborting.")
        return None

    # --- Step 2: Define Heuristics for Headings ---
    # These thresholds are a starting point and can be refined.
    # We assume H1 is significantly larger than body, H2 is moderately larger, etc.
    H1_THRESHOLD = base_size * 1.8
    H2_THRESHOLD = base_size * 1.4
    H3_THRESHOLD = base_size * 1.2

    # --- Step 3: Iterate Through Pages and Extract Potential Headings ---
    outline = []
    potential_title = ""
    title_found = False

    for page_num, page in enumerate(doc):
        # Using "dict" gives us detailed info like font size, flags (bold), etc.
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:  # This is a text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if not text:
                            continue

                        font_size = s["size"]
                        is_bold = "bold" in s["font"].lower()

                        # Simple Title Heuristic: First large, bold text on the first page.
                        if page_num == 0 and not title_found and font_size > H1_THRESHOLD and is_bold:
                            potential_title = text
                            title_found = True
                            continue # Don't also classify the title as a heading

                        heading_level = None
                        # Check font size against our thresholds
                        if font_size >= H1_THRESHOLD:
                            heading_level = "H1"
                        elif font_size >= H2_THRESHOLD:
                            heading_level = "H2"
                        elif font_size >= H3_THRESHOLD:
                            # Add an extra check for H3 to be bold to reduce false positives
                            if is_bold:
                                heading_level = "H3"

                        if heading_level:
                            outline.append({
                                "level": heading_level,
                                "text": text,
                                "page": page_num + 1  # Page numbers are 1-indexed for users
                            })

    # --- Step 4: Assemble Final JSON Structure ---
    # If no specific title was found, use the filename as a fallback.
    if not potential_title:
        potential_title = os.path.basename(pdf_path)

    result = {
        "title": potential_title,
        "outline": outline
    }

    return result

if __name__ == '__main__':
    # --- Configuration ---
    # This script is designed to be run in a Docker container as per the challenge.
    # The input and output paths will be mounted volumes.
    INPUT_DIR = "/home/necro/Desktop/Adobe/code/input"
    OUTPUT_DIR = "/home/necro/Desktop/Adobe/code/output"

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Processing ---
    # Process all PDF files in the input directory.
    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith(".pdf"):
                print(f"Processing {filename}...")
                pdf_path = os.path.join(INPUT_DIR, filename)

                # Extract the structure
                structure_data = extract_pdf_structure(pdf_path)

                if structure_data:
                    # Define the output path
                    output_filename = os.path.splitext(filename)[0] + ".json"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)

                    # Write the JSON output
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(structure_data, f, indent=4, ensure_ascii=False)
                    print(f"Successfully created {output_path}")
    else:
        print(f"Input directory '{INPUT_DIR}' not found. Please create it and add PDF files.")


