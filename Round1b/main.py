# main.py
#
# This script represents the complete, end-to-end solution for the
# Adobe India Hackathon 2025, Round 1B: Persona-Driven Document Intelligence.
#
# ##############################################################################
# # LOCAL PC VERSION - PAGE-AWARE V2
# ##############################################################################
# This version has been significantly updated to fix page number inaccuracies.
#
# - The `extract_pdf_structure` function has been re-architected to create
#   granular, page-aware chunks. Instead of one large content block per
#   heading, it now creates smaller chunks for each text block on each page,
#   ensuring every chunk has an accurate page number.
# - The HybridRetriever now indexes these smaller, more precise chunks.
# - The final output now correctly reports the page where the specific
#   `refined_text` was found, not just the page of the parent heading.
#
##############################################################################
import nltk
import os
import json
import re
import datetime
import time
from pathlib import Path
from collections import defaultdict

# --- Core ML/NLP Libraries ---
import numpy as np
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

# --- NLTK for Sentence Splitting ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt', quiet=True)
import nltk


# --- Configuration ---
# Create directories for local execution
INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

REQUEST_FILE = INPUT_DIR / "request.json"
# Use the Hugging Face model name directly. It will be downloaded automatically.
MODEL_NAME = 'Alibaba-NLP/gte-large-en-v1.5'
TOP_K_RESULTS = 10 # Number of final sections to return
HYBRID_SEARCH_ALPHA = 0.7 # Weight for semantic search score in fusion

# ##############################################################################
# # 1. Ingestion and Structuring Module (INTEGRATED FROM ROUND 1A)
# ##############################################################################

def get_font_styles(doc):
    """
    Analyzes the document to get statistics on font sizes and styles.
    This helps create dynamic thresholds instead of fixed ones.
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
    tolerance = page_width * 0.20
    return abs(block_center - page_center) < tolerance

def extract_pdf_structure(pdf_path, doc_name):
    """
    Extracts a structured outline from a PDF, creating granular, page-aware chunks
    for each text block under a given heading.
    """
    print(f"Running page-aware structure extraction on {doc_name}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}")
        return []

    font_styles, body_size = get_font_styles(doc)
    if body_size == 0:
        print(f"Could not determine font styles for {pdf_path}.")
        return []

    size_hierarchy = sorted([s[0] for s in font_styles if s[0] > body_size], reverse=True)
    
    H1_SIZE = size_hierarchy[0] if len(size_hierarchy) > 0 else body_size * 1.5
    H2_SIZE = size_hierarchy[1] if len(size_hierarchy) > 1 else body_size * 1.25
    H3_SIZE = body_size * 1.15

    # --- Pass 1: Classify all text lines to find headings ---
    classified_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    line_text = "".join(s['text'] for s in l['spans']).strip()
                    if not line_text or not l['spans']: continue
                    
                    first_span = l['spans'][0]
                    font_size = round(first_span["size"])
                    is_bold = "bold" in first_span["font"].lower()
                    rect = fitz.Rect(l['bbox'])

                    heading_level = None
                    if page_num == 0 and font_size > H1_SIZE and is_centered(rect, page.rect.width):
                        heading_level = "Title"
                    elif font_size >= H1_SIZE and is_bold:
                        heading_level = "H1"
                    elif font_size >= H2_SIZE and is_bold:
                        heading_level = "H2"
                    elif font_size > body_size and (is_bold or font_size >= H3_SIZE) and len(line_text.split()) < 15:
                         heading_level = "H3"
                    
                    if heading_level:
                        classified_blocks.append({
                            "level": heading_level, "text": line_text, "page": page_num + 1,
                            "bbox": rect
                        })

    # --- Pass 2: Merge multi-line headings ---
    merged_headings = []
    if classified_blocks:
        current_heading = classified_blocks[0]
        for i in range(1, len(classified_blocks)):
            next_heading = classified_blocks[i]
            if (next_heading["level"] == current_heading["level"] and
                next_heading["page"] == current_heading["page"] and
                abs(next_heading["bbox"].y0 - current_heading["bbox"].y1) < 20):
                current_heading["text"] += " " + next_heading["text"]
                current_heading["bbox"] |= next_heading["bbox"]
            else:
                merged_headings.append(current_heading)
                current_heading = next_heading
        merged_headings.append(current_heading)

    # --- Step 3: Create Granular, Page-Aware Chunks ---
    final_chunks = []
    sorted_headings = sorted([h for h in merged_headings if h['level'] != 'Title'], key=lambda b: (b['page'], b['bbox'].y0))

    for i, heading in enumerate(sorted_headings):
        start_page_idx = heading['page'] - 1
        start_y = heading['bbox'].y1
        
        # Determine the end boundary for this heading's content
        end_page_idx = doc.page_count - 1
        end_y = doc[end_page_idx].rect.height
        if i + 1 < len(sorted_headings):
            next_heading = sorted_headings[i+1]
            end_page_idx = next_heading['page'] - 1
            end_y = next_heading['bbox'].y0

        # Iterate through the pages belonging to this section
        for page_num in range(start_page_idx, end_page_idx + 1):
            page = doc[page_num]
            
            # Define the vertical clip area for the current page
            clip_y0 = start_y if page_num == start_page_idx else 0
            clip_y1 = end_y if page_num == end_page_idx else page.rect.height
            clip_rect = fitz.Rect(0, clip_y0, page.rect.width, clip_y1)
            
            # Extract text blocks within the clipped area
            blocks = page.get_text("blocks", clip=clip_rect)
            for b in blocks:
                block_text = b[4].strip()
                if b[6] == 0 and block_text: # Ensure it's a text block with content
                    final_chunks.append({
                        "document": doc_name,
                        "page_number": page_num + 1, # The accurate page number for this block
                        "section_title": re.sub(r'\s+', ' ', heading['text']).strip(),
                        "content": re.sub(r'\s+', ' ', block_text)
                    })
    
    print(f"Generated {len(final_chunks)} granular chunks for {doc_name}.")
    return final_chunks

# ##############################################################################
# # 2. Query Processing Module
# ##############################################################################

def decompose_query(request_data):
    """
    Deconstructs the persona and job from the request data object into multiple,
    specific sub-queries to improve retrieval accuracy.
    """
    print("Decomposing query from request data...")
    
    persona_role = request_data.get("persona", {}).get("role")
    job_task = request_data.get("job_to_be_done", {}).get("task")

    if not persona_role or not job_task:
        print("Error: Could not find 'persona.role' or 'job_to_be_done.task' in request.json.")
        return []

    tasks = re.split(r',\s*|\s+and\s+', job_task)
    
    sub_queries = []
    base_query = f"{persona_role} focused on: {job_task}"
    sub_queries.append(base_query)

    for task in tasks:
        task_cleaned = task.strip()
        if task_cleaned:
            sub_query = f"As a {persona_role}, I need to analyze: {task_cleaned}"
            sub_queries.append(sub_query)
            
    print(f"Generated {len(sub_queries)} sub-queries.")
    return sub_queries

# ##############################################################################
# # 3. Hybrid Retrieval Engine Module
# ##############################################################################

class HybridRetriever:
    """
    Encapsulates the entire hybrid search process, combining lexical (BM25)
    and semantic search.
    """
    def __init__(self, model_name):
        print("Initializing Hybrid Retriever...")
        st = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        print(f"Model loaded in {time.time() - st:.2f}s")
        self.chunks = []
        self.bm25 = None
        self.embeddings = None

    def index(self, chunks):
        """Builds the lexical and semantic indexes from the document chunks."""
        print(f"Building indexes for {len(chunks)} chunks...")
        st = time.time()
        self.chunks = chunks
        
        contents = [chunk.get('content', '') for chunk in chunks]
        self.bm25 = BM25Okapi([doc.split(" ") for doc in contents])
        
        self.embeddings = self.model.encode(contents, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        print(f"Indexing completed in {time.time() - st:.2f}s")

    def search(self, sub_queries):
        """
        Performs the hybrid search and returns a ranked list of chunks.
        """
        print("Performing hybrid search...")
        st = time.time()
        query_embeddings = self.model.encode(sub_queries, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        semantic_scores = np.max(self.embeddings @ query_embeddings.T, axis=1)
        
        tokenized_query = sub_queries[0].split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)

        norm_semantic = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        final_scores = (HYBRID_SEARCH_ALPHA * norm_semantic) + ((1 - HYBRID_SEARCH_ALPHA) * norm_bm25)
        
        top_indices = np.argsort(final_scores)[::-1][:TOP_K_RESULTS * 2]
        
        ranked_chunks = [(self.chunks[i], final_scores[i]) for i in top_indices]
        print(f"Search completed in {time.time() - st:.2f}s")
        return ranked_chunks, query_embeddings

# ##############################################################################
# # 4. Sub-section Analysis Module
# ##############################################################################

def analyze_subsection(chunk, query_embeddings, retriever):
    """
    Performs sentence-level analysis on a smaller chunk to find the most
    relevant "Refined Text" and a more accurate relevance score.
    """
    content = chunk.get('content', '')
    sentences = nltk.sent_tokenize(content)
    if not sentences:
        return content, 0.0

    sentence_embeddings = retriever.model.encode(sentences, convert_to_numpy=True)
    sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    sim_matrix = sentence_embeddings @ query_embeddings.T
    sentence_scores = np.max(sim_matrix, axis=1)
    
    max_sentence_score = np.max(sentence_scores) if sentence_scores.size > 0 else 0.0

    # Extract top 2 sentences for the refined text
    top_sentence_indices = np.argsort(sentence_scores)[::-1][:2]
    top_sentence_indices.sort()
    
    refined_text = " ".join([sentences[i] for i in top_sentence_indices])
    return refined_text, max_sentence_score


# ##############################################################################
# # 5. Main Orchestration Logic
# ##############################################################################

def main():
    """Main execution function."""
    total_start_time = time.time()
    print("--- Starting Persona-Driven Document Intelligence (Page-Aware V2) ---")

    if not REQUEST_FILE.exists():
        print(f"\nFATAL: Request file not found at '{REQUEST_FILE}'.")
        print("Please create the file with your persona and job_to_be_done, then run again.")
        return

    with open(REQUEST_FILE, 'r') as f:
        request_data = json.load(f)
    
    all_chunks = []
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("\nFATAL: No PDF files found in the 'input' directory.")
        return
        
    doc_names = [p.name for p in pdf_files]
    
    for pdf_path in pdf_files:
        chunks = extract_pdf_structure(pdf_path, pdf_path.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("FATAL: No chunks could be extracted from documents. Exiting.")
        return

    sub_queries = decompose_query(request_data)
    if not sub_queries:
        return

    retriever = HybridRetriever(model_name=MODEL_NAME)
    retriever.index(all_chunks)
    top_ranked_chunks, query_embeddings = retriever.search(sub_queries)

    print("Performing final analysis and re-ranking...")
    analysis_results = []
    for chunk, initial_score in top_ranked_chunks:
        refined_text, max_sentence_score = analyze_subsection(chunk, query_embeddings, retriever)
        final_score = (initial_score * 0.4) + (max_sentence_score * 0.6)
        analysis_results.append({
            "chunk": chunk,
            "refined_text": refined_text,
            "score": final_score
        })
        
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = analysis_results[:TOP_K_RESULTS]

    extracted_sections = []
    sub_section_analysis = []
    for rank, result in enumerate(final_results, 1):
        chunk = result['chunk']
        extracted_sections.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"], # This is now accurate
            "section_title": chunk["section_title"],
            "importance_rank": rank
        })
        sub_section_analysis.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"], # This is now accurate
            "refined_text": result["refined_text"],
            "importance_rank": rank
        })

    output_data = {
        "metadata": {
            "input_documents": doc_names,
            "persona": request_data.get("persona"),
            "job_to_be_done": request_data.get("job_to_be_done"),
            "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }
    
    output_path = OUTPUT_DIR / "output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    total_time = time.time() - total_start_time
    print(f"\n--- Processing Complete in {total_time:.2f} seconds ---")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()

