# main.py
#
# This script represents the complete, end-to-end solution for the
# Adobe India Hackathon 2025, Round 1B: Persona-Driven Document Intelligence.
#
# ##############################################################################
# # INTEGRATION NOTES:
# ##############################################################################
# This version integrates the advanced PDF structure extraction logic from the
# user's Round 1A solution.
#
# - Replaced `PyPDF2` with `PyMuPDF` (`fitz`) for superior parsing.
# - Replaced the placeholder `extract_text_from_pdf` and `intelligent_chunking`
#   functions with the user's `extract_pdf_structure` and its helpers.
# - The main loop now iterates through PDFs, calls `extract_pdf_structure`,
#   and transforms the output into the chunk format required for the retriever.
# - Optimized the `analyze_subsection` function to reuse the main model
#   instead of reloading it from disk for every chunk, providing a significant
#   performance boost.
#
##############################################################################

import os
import json
import re
import datetime
import time
from pathlib import Path
from collections import defaultdict

# --- Core ML/NLP Libraries ---
import numpy as np
import fitz  # PyMuPDF, integrated from Round 1A solution
from rank_bm25 import BM25Okapi
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# --- NLTK for Sentence Splitting ---
# The Dockerfile ensures these are pre-downloaded.
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt', quiet=True)


# --- Configuration ---
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
REQUEST_FILE = INPUT_DIR / "request.json"
MODEL_PATH = Path("./gte-large-onnx-quantized") # Path to the optimized model
TOP_K_RESULTS = 10 # Number of final sections to return
HYBRID_SEARCH_ALPHA = 0.7 # Weight for semantic search score in fusion

# ##############################################################################
# # 1. Ingestion and Structuring Module (INTEGRATED FROM ROUND 1A)
# ##############################################################################

def get_font_styles(doc):
    """
    Analyzes the document to get statistics on font sizes and styles.
    This helps create dynamic thresholds instead of fixed ones.
    (From user's Round 1A solution)
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

def extract_pdf_structure(pdf_path, doc_name):
    """
    Extracts a structured outline from a PDF, including the text content under each heading.
    This is the core logic from the user's Round 1A solution, adapted to return chunks.
    """
    print(f"Running advanced structure extraction on {doc_name}...")
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
    H3_SIZE = body_size * 1.15 # Define a size for H3

    # --- Pass 1: Classify all text lines ---
    classified_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0: # Text blocks
                for l in b["lines"]:
                    line_text = "".join(s['text'] for s in l['spans']).strip()
                    if not line_text: continue
                    
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

    # --- Step 3: Extract Content and Format as Chunks ---
    final_chunks = []
    merged_outline.sort(key=lambda b: (b['page'], b['bbox'].y0))

    for i, heading in enumerate(merged_outline):
        if heading['level'] == 'Title':
            continue

        start_page = heading['page'] - 1
        start_rect = fitz.Rect(0, heading['bbox'].y1, doc[start_page].rect.width, doc[start_page].rect.height)
        
        end_page = doc.page_count
        end_rect = doc[end_page - 1].rect

        if i + 1 < len(merged_outline):
            next_heading = merged_outline[i+1]
            end_page = next_heading['page'] - 1
            end_rect = fitz.Rect(0, 0, doc[end_page].rect.width, next_heading['bbox'].y0)

        content_text = ""
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]
            clip_rect = page.rect
            if page_num == start_page:
                clip_rect.y0 = start_rect.y0
            if page_num == end_page:
                clip_rect.y1 = end_rect.y1
            
            text = page.get_text(clip=clip_rect).strip()
            if text:
                content_text += text + " "

        final_chunks.append({
            "document": doc_name,
            "page_number": heading["page"],
            "section_title": re.sub(r'\s+', ' ', heading['text']).strip(),
            "content": re.sub(r'\s+', ' ', content_text).strip()
        })
    
    print(f"Generated {len(final_chunks)} chunks for {doc_name} using advanced parsing.")
    return final_chunks

# ##############################################################################
# # 2. Query Processing Module
# ##############################################################################

def decompose_query(persona, job_to_be_done):
    """
    Deconstructs the persona and job into multiple, specific sub-queries
    to improve retrieval accuracy.
    """
    print("Decomposing query...")
    tasks = re.split(r',\s*|\s+and\s+', job_to_be_done)
    
    sub_queries = []
    base_query = f"{persona} focused on: {job_to_be_done}"
    sub_queries.append(base_query)

    for task in tasks:
        task_cleaned = task.strip()
        if task_cleaned:
            sub_query = f"As a {persona}, I need to analyze: {task_cleaned}"
            sub_queries.append(sub_query)
            
    print(f"Generated {len(sub_queries)} sub-queries.")
    return sub_queries

# ##############################################################################
# # 3. Hybrid Retrieval Engine Module
# ##############################################################################

class HybridRetriever:
    """
    Encapsulates the entire hybrid search process, combining lexical (BM25)
    and semantic (ONNX model) search.
    """
    def __init__(self, model_path):
        print("Initializing Hybrid Retriever...")
        st = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_path)
        print(f"Model loaded in {time.time() - st:.2f}s")
        self.chunks = []
        self.bm25 = None
        self.embeddings = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts, batch_size=32):
        """Generates embeddings for a list of texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                    return_tensors='np', max_length=512)
            outputs = self.model(**inputs)
            pooled_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            all_embeddings.append(pooled_embeddings)
        return np.vstack(all_embeddings)

    def index(self, chunks):
        """Builds the lexical and semantic indexes from the document chunks."""
        print("Building lexical and semantic indexes...")
        st = time.time()
        self.chunks = chunks
        
        contents = [chunk.get('content', '') for chunk in chunks]
        self.bm25 = BM25Okapi([doc.split(" ") for doc in contents])
        
        self.embeddings = self.encode(contents)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        print(f"Indexing completed in {time.time() - st:.2f}s")

    def search(self, sub_queries):
        """
        Performs the hybrid search and returns a ranked list of chunks.
        """
        print("Performing hybrid search...")
        st = time.time()
        query_embeddings = self.encode(sub_queries)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        semantic_scores = np.max(self.embeddings @ query_embeddings.T, axis=1)
        
        tokenized_query = sub_queries[0].split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)

        norm_semantic = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)
        norm_bm25 = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        final_scores = (HYBRID_SEARCH_ALPHA * norm_semantic) + ((1 - HYBRID_SEARCH_ALPHA) * norm_bm25)
        
        top_indices = np.argsort(final_scores)[::-1][:TOP_K_RESULTS * 2] # Get more results for better analysis
        
        ranked_chunks = [(self.chunks[i], final_scores[i]) for i in top_indices]
        print(f"Search completed in {time.time() - st:.2f}s")
        return ranked_chunks, query_embeddings

# ##############################################################################
# # 4. Sub-section Analysis Module
# ##############################################################################

def analyze_subsection(chunk, query_embeddings, retriever):
    """
    Performs sentence-level analysis on a chunk to find the most
    relevant "Refined Text".
    
    OPTIMIZATION: Passes the 'retriever' object to reuse its 'encode' method
    and avoid reloading the model from disk.
    """
    sentences = nltk.sent_tokenize(chunk['content'])
    if not sentences:
        return chunk['content'], 0.0

    # Reuse the main retriever's encoder
    sentence_embeddings = retriever.encode(sentences)
    sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    sim_matrix = sentence_embeddings @ query_embeddings.T
    sentence_scores = np.max(sim_matrix, axis=1)
    
    max_sentence_score = np.max(sentence_scores)

    # Get top 3 sentences
    top_sentence_indices = np.argsort(sentence_scores)[::-1][:3]
    top_sentence_indices.sort() # Sort to maintain original order
    
    refined_text = " ".join([sentences[i] for i in top_sentence_indices])
    return refined_text, max_sentence_score


# ##############################################################################
# # 5. Main Orchestration Logic
# ##############################################################################

def main():
    """Main execution function."""
    total_start_time = time.time()
    print("--- Starting Persona-Driven Document Intelligence (Integrated v2) ---")

    with open(REQUEST_FILE, 'r') as f:
        request_data = json.load(f)
    persona = request_data["persona"]
    job_to_be_done = request_data["job_to_be_done"]
    
    # --- 2. Ingest and Chunk Documents using Round 1A logic ---
    all_chunks = []
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    doc_names = [p.name for p in pdf_files]
    
    for pdf_path in pdf_files:
        # The advanced function now generates chunks directly
        chunks = extract_pdf_structure(pdf_path, pdf_path.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("FATAL: No chunks could be extracted from documents. Exiting.")
        return

    # --- 3. Process Query & Run Retrieval ---
    sub_queries = decompose_query(persona, job_to_be_done)
    retriever = HybridRetriever(model_path=MODEL_PATH)
    retriever.index(all_chunks)
    top_ranked_chunks, query_embeddings = retriever.search(sub_queries)

    # --- 5. Perform Re-ranking and Sub-section Analysis ---
    print("Performing final analysis and re-ranking...")
    analysis_results = []
    for chunk, initial_score in top_ranked_chunks:
        refined_text, max_sentence_score = analyze_subsection(chunk, query_embeddings, retriever)
        
        # Re-rank based on the sentence-level analysis score
        final_score = (initial_score * 0.4) + (max_sentence_score * 0.6)

        analysis_results.append({
            "chunk": chunk,
            "refined_text": refined_text,
            "score": final_score
        })
        
    # Sort by the new final score and take the top K
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = analysis_results[:TOP_K_RESULTS]

    # --- 6. Generate Final Output JSON ---
    extracted_sections = []
    sub_section_analysis = []
    for rank, result in enumerate(final_results, 1):
        chunk = result['chunk']
        extracted_sections.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "section_title": chunk["section_title"],
            "importance_rank": rank
        })
        sub_section_analysis.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "refined_text": result["refined_text"],
            "importance_rank": rank
        })

    output_data = {
        "metadata": {"input_documents": doc_names, "persona": persona, "job_to_be_done": job_to_be_done,
                     "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"},
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }
    
    output_path = OUTPUT_DIR / "output.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    total_time = time.time() - total_start_time
    print(f"--- Processing Complete in {total_time:.2f} seconds ---")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    main()

