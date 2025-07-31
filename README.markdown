![graphviz (1)](https://github.com/user-attachments/assets/f63a41b1-af7c-487f-b0d2-56823acbbe7f)# Document Distillation pipeline for resource constrained Environment

## Project Overview
This repository contains the complete source code and documentation for a high-performance system designed for the **Adobe India Hackathon 2025, Round 1B: Persona-Driven Document Intelligence**. The solution implements an end-to-end pipeline that ingests a collection of PDF documents, analyzes them based on a specified user persona and job-to-be-done, and extracts a ranked list of the most relevant sections.

The system is architected to run efficiently in a resource-constrained, offline, CPU-only Docker environment, adhering to all competition requirements. Additionally, the repository includes a separate solution for Round 1A, which addresses a related but independent part of the problem, housed in its own directory with a similar structure.

![Uploading gra<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1028pt" height="535pt" viewBox="0.00 0.00 1028.00 534.53">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(4 530.53)">
<title>G</title>
<polygon fill="white" stroke="none" points="-4,4 -4,-530.53 1024,-530.53 1024,4 -4,4"/>
<g id="clust1" class="cluster">
<title>cluster_optimization</title>
<polygon fill="#fafafa" stroke="black" stroke-dasharray="5,2" points="8,-331.2 8,-518.53 326,-518.53 326,-331.2 8,-331.2"/>
<text text-anchor="middle" x="167" y="-504.63" font-family="Helvetica,sans-Serif" font-size="11.00">Offline Model Optimization (Build Time)</text>
</g>
<g id="clust2" class="cluster">
<title>cluster_main_pipeline</title>
<polygon fill="#f0f2f5" stroke="black" points="352,-8 352,-415.33 1012,-415.33 1012,-8 352,-8"/>
<text text-anchor="middle" x="682" y="-401.43" font-family="Helvetica,sans-Serif" font-size="11.00">Runtime Pipeline</text>
</g>
<g id="clust3" class="cluster">
<title>cluster_io</title>
</g>
<g id="clust4" class="cluster">
<title>cluster_retrieval</title>
<polygon fill="white" stroke="black" points="368,-113 368,-293.8 671,-293.8 671,-113 368,-113"/>
<text text-anchor="middle" x="519.5" y="-279.9" font-family="Helvetica,sans-Serif" font-size="11.00">Hybrid Retrieval</text>
</g>
<!-- opt_src -->
<g id="node1" class="node">
<title>opt_src</title>
<ellipse fill="#fffbe6" stroke="#ffe58f" cx="167" cy="-471.33" rx="84.18" ry="18"/>
<text text-anchor="middle" x="167" y="-468.63" font-family="Helvetica,sans-Serif" font-size="9.00">{Alibaba-NLP/gte-large-en-v1.5}</text>
</g>
<!-- opt_proc -->
<g id="node2" class="node">
<title>opt_proc</title>
<ellipse fill="#fffbe6" stroke="#ffe58f" cx="167" cy="-360.13" rx="150.85" ry="20.93"/>
<text text-anchor="middle" x="167" y="-362.83" font-family="Helvetica,sans-Serif" font-size="9.00">{Post-Training Static Quantization (FP32 -&gt; INT8)</text>
<text text-anchor="middle" x="167" y="-352.03" font-family="Helvetica,sans-Serif" font-size="9.00">using ONNX}</text>
</g>
<!-- opt_src&#45;&gt;opt_proc -->
<g id="edge1" class="edge">
<title>opt_src-&gt;opt_proc</title>
<path fill="none" stroke="#434343" d="M167,-452.97C167,-452.97 167,-388.69 167,-388.69"/>
<polygon fill="#434343" stroke="#434343" points="169.1,-388.69 167,-382.69 164.9,-388.69 169.1,-388.69"/>
</g>
<!-- indexing -->
<g id="node8" class="node">
<title>indexing</title>
<path fill="#e6f7ff" stroke="#91d5ff" d="M387.73,-226.5C387.73,-226.5 532.27,-226.5 532.27,-226.5 538.27,-226.5 544.27,-232.5 544.27,-238.5 544.27,-238.5 544.27,-252.1 544.27,-252.1 544.27,-258.1 538.27,-264.1 532.27,-264.1 532.27,-264.1 387.73,-264.1 387.73,-264.1 381.73,-264.1 375.73,-258.1 375.73,-252.1 375.73,-252.1 375.73,-238.5 375.73,-238.5 375.73,-232.5 381.73,-226.5 387.73,-226.5"/>
<text text-anchor="middle" x="460" y="-252" font-family="Helvetica,sans-Serif" font-size="9.00">Hybrid Indexing</text>
<polyline fill="none" stroke="#91d5ff" points="375.73,-245.3 544.27,-245.3"/>
<text text-anchor="middle" x="413.74" y="-233.2" font-family="Helvetica,sans-Serif" font-size="9.00">Lexical (BM25)</text>
<polyline fill="none" stroke="#91d5ff" points="451.75,-226.5 451.75,-245.3"/>
<text text-anchor="middle" x="498.01" y="-233.2" font-family="Helvetica,sans-Serif" font-size="9.00">Semantic (Vectors)</text>
</g>
<!-- opt_proc&#45;&gt;indexing -->
<g id="edge10" class="edge">
<title>opt_proc-&gt;indexing:sem</title>
<path fill="none" stroke="#434343" stroke-dasharray="5,2" d="M167,-338.8C167,-305.59 167,-245.3 167,-245.3 167,-245.3 443.77,-245.3 443.77,-245.3"/>
<polygon fill="#434343" stroke="#434343" points="443.77,-247.4 449.77,-245.3 443.77,-243.2 443.77,-247.4"/>
</g>
<!-- input_pdf -->
<g id="node3" class="node">
<title>input_pdf</title>
<path fill="#d9f7be" stroke="#b7eb8f" d="M682.51,-378.13C682.51,-378.13 594.35,-378.13 594.35,-378.13 588.35,-378.13 578.59,-373.45 574.84,-368.77 574.84,-368.77 561,-351.49 561,-351.49 557.25,-346.81 559.49,-342.13 565.49,-342.13 565.49,-342.13 653.65,-342.13 653.65,-342.13 659.65,-342.13 669.41,-346.81 673.16,-351.49 673.16,-351.49 687,-368.77 687,-368.77 690.75,-373.45 688.51,-378.13 682.51,-378.13"/>
<text text-anchor="middle" x="624" y="-357.43" font-family="Helvetica,sans-Serif" font-size="9.00">PDF Documents</text>
</g>
<!-- ingestion -->
<g id="node6" class="node">
<title>ingestion</title>
<path fill="#e6f7ff" stroke="#91d5ff" d="M372.47,-341.33C372.47,-341.33 455.53,-341.33 455.53,-341.33 461.53,-341.33 467.53,-347.33 467.53,-353.33 467.53,-353.33 467.53,-366.93 467.53,-366.93 467.53,-372.93 461.53,-378.93 455.53,-378.93 455.53,-378.93 372.47,-378.93 372.47,-378.93 366.47,-378.93 360.47,-372.93 360.47,-366.93 360.47,-366.93 360.47,-353.33 360.47,-353.33 360.47,-347.33 366.47,-341.33 372.47,-341.33"/>
<text text-anchor="middle" x="414" y="-366.83" font-family="Helvetica,sans-Serif" font-size="9.00">Ingestion &amp; Structuring</text>
<polyline fill="none" stroke="#91d5ff" points="360.47,-360.13 467.53,-360.13"/>
<text text-anchor="middle" x="414" y="-348.03" font-family="Helvetica,sans-Serif" font-size="9.00">Page-Aware Chunking</text>
</g>
<!-- input_pdf&#45;&gt;ingestion -->
<g id="edge2" class="edge">
<title>input_pdf-&gt;ingestion</title>
<path fill="none" stroke="#434343" d="M567.45,-360.13C567.45,-360.13 475.27,-360.13 475.27,-360.13"/>
<polygon fill="#434343" stroke="#434343" points="475.27,-358.03 469.27,-360.13 475.27,-362.23 475.27,-358.03"/>
</g>
<!-- input_request -->
<g id="node4" class="node">
<title>input_request</title>
<path fill="#d9f7be" stroke="#b7eb8f" d="M836.9,-378.13C836.9,-378.13 771.97,-378.13 771.97,-378.13 765.97,-378.13 756.76,-373.07 753.54,-368 753.54,-368 743.53,-352.26 743.53,-352.26 740.32,-347.19 743.1,-342.13 749.1,-342.13 749.1,-342.13 814.03,-342.13 814.03,-342.13 820.03,-342.13 829.24,-347.19 832.46,-352.26 832.46,-352.26 842.47,-368 842.47,-368 845.68,-373.07 842.9,-378.13 836.9,-378.13"/>
<text text-anchor="middle" x="793" y="-357.43" font-family="Helvetica,sans-Serif" font-size="9.00">request.json</text>
</g>
<!-- query_decomp -->
<g id="node7" class="node">
<title>query_decomp</title>
<path fill="#e6f7ff" stroke="#91d5ff" d="M599.24,-227.3C599.24,-227.3 650.76,-227.3 650.76,-227.3 656.76,-227.3 662.76,-233.3 662.76,-239.3 662.76,-239.3 662.76,-251.3 662.76,-251.3 662.76,-257.3 656.76,-263.3 650.76,-263.3 650.76,-263.3 599.24,-263.3 599.24,-263.3 593.24,-263.3 587.24,-257.3 587.24,-251.3 587.24,-251.3 587.24,-239.3 587.24,-239.3 587.24,-233.3 593.24,-227.3 599.24,-227.3"/>
<text text-anchor="middle" x="625" y="-248.2" font-family="Helvetica,sans-Serif" font-size="9.00">Query</text>
<text text-anchor="middle" x="625" y="-237.4" font-family="Helvetica,sans-Serif" font-size="9.00">Decomposition</text>
</g>
<!-- input_request&#45;&gt;query_decomp -->
<g id="edge5" class="edge">
<title>input_request-&gt;query_decomp</title>
<path fill="none" stroke="#434343" d="M783.44,-341.68C783.44,-309.22 783.44,-245.3 783.44,-245.3 783.44,-245.3 670.46,-245.3 670.46,-245.3"/>
<polygon fill="#434343" stroke="#434343" points="670.46,-243.2 664.46,-245.3 670.46,-247.4 670.46,-243.2"/>
</g>
<!-- output_json -->
<g id="node5" class="node">
<title>output_json</title>
<path fill="#d9f7be" stroke="#b7eb8f" d="M983.61,-378.13C983.61,-378.13 925.51,-378.13 925.51,-378.13 919.51,-378.13 910.48,-372.96 907.44,-367.78 907.44,-367.78 898.47,-352.48 898.47,-352.48 895.43,-347.31 898.39,-342.13 904.39,-342.13 904.39,-342.13 962.49,-342.13 962.49,-342.13 968.49,-342.13 977.52,-347.31 980.56,-352.48 980.56,-352.48 989.53,-367.78 989.53,-367.78 992.57,-372.96 989.61,-378.13 983.61,-378.13"/>
<text text-anchor="middle" x="944" y="-357.43" font-family="Helvetica,sans-Serif" font-size="9.00">output.json</text>
</g>
<!-- ingestion&#45;&gt;indexing -->
<g id="edge3" class="edge">
<title>ingestion-&gt;indexing:lex</title>
<path fill="none" stroke="#434343" d="M406.33,-340.99C406.33,-340.99 406.33,-253 406.33,-253"/>
<polygon fill="#434343" stroke="#434343" points="408.43,-253 406.33,-247 404.23,-253 408.43,-253"/>
<text text-anchor="middle" x="433.65" y="-303.9" font-family="Helvetica,sans-Serif" font-size="7.00">Text Chunks</text>
</g>
<!-- ingestion&#45;&gt;indexing -->
<g id="edge4" class="edge">
<title>ingestion-&gt;indexing:sem</title>
<path fill="none" stroke="#434343" d="M436.93,-340.99C436.93,-340.99 436.93,-243.46 436.93,-243.46"/>
<polygon fill="#434343" stroke="#434343" points="439.03,-243.46 436.93,-237.46 434.83,-243.46 439.03,-243.46"/>
</g>
<!-- retrieval -->
<g id="node9" class="node">
<title>retrieval</title>
<path fill="#e6f7ff" stroke="#91d5ff" d="M585.24,-121.5C585.24,-121.5 650.76,-121.5 650.76,-121.5 656.76,-121.5 662.76,-127.5 662.76,-133.5 662.76,-133.5 662.76,-147.1 662.76,-147.1 662.76,-153.1 656.76,-159.1 650.76,-159.1 650.76,-159.1 585.24,-159.1 585.24,-159.1 579.24,-159.1 573.24,-153.1 573.24,-147.1 573.24,-147.1 573.24,-133.5 573.24,-133.5 573.24,-127.5 579.24,-121.5 585.24,-121.5"/>
<text text-anchor="middle" x="618" y="-147" font-family="Helvetica,sans-Serif" font-size="9.00">Retrieval &amp; Fusion</text>
<polyline fill="none" stroke="#91d5ff" points="573.24,-140.3 662.76,-140.3"/>
<text text-anchor="middle" x="618" y="-128.2" font-family="Helvetica,sans-Serif" font-size="9.00">(α=0.7)</text>
</g>
<!-- query_decomp&#45;&gt;retrieval -->
<g id="edge6" class="edge">
<title>query_decomp-&gt;retrieval</title>
<path fill="none" stroke="#434343" d="M625,-227.01C625,-227.01 625,-166.96 625,-166.96"/>
<polygon fill="#434343" stroke="#434343" points="627.1,-166.96 625,-160.96 622.9,-166.96 627.1,-166.96"/>
<text text-anchor="middle" x="640.65" y="-190.7" font-family="Helvetica,sans-Serif" font-size="7.00">Sub-Queries</text>
</g>
<!-- indexing&#45;&gt;retrieval -->
<g id="edge7" class="edge">
<title>indexing-&gt;retrieval</title>
<path fill="none" stroke="#434343" d="M544.49,-245.3C564.54,-245.3 580.24,-245.3 580.24,-245.3 580.24,-245.3 580.24,-167.07 580.24,-167.07"/>
<polygon fill="#434343" stroke="#434343" points="582.34,-167.07 580.24,-161.07 578.14,-167.07 582.34,-167.07"/>
</g>
<!-- analysis -->
<g id="node10" class="node">
<title>analysis</title>
<path fill="#e6f7ff" stroke="#91d5ff" d="M728.23,-16.5C728.23,-16.5 817.77,-16.5 817.77,-16.5 823.77,-16.5 829.77,-22.5 829.77,-28.5 829.77,-28.5 829.77,-42.1 829.77,-42.1 829.77,-48.1 823.77,-54.1 817.77,-54.1 817.77,-54.1 728.23,-54.1 728.23,-54.1 722.23,-54.1 716.23,-48.1 716.23,-42.1 716.23,-42.1 716.23,-28.5 716.23,-28.5 716.23,-22.5 722.23,-16.5 728.23,-16.5"/>
<text text-anchor="middle" x="773" y="-42" font-family="Helvetica,sans-Serif" font-size="9.00">Re-Ranking &amp; Synthesis</text>
<polyline fill="none" stroke="#91d5ff" points="716.23,-35.3 829.77,-35.3"/>
<text text-anchor="middle" x="773" y="-23.2" font-family="Helvetica,sans-Serif" font-size="9.00">Sentence-Level Scoring</text>
</g>
<!-- retrieval&#45;&gt;analysis -->
<g id="edge8" class="edge">
<title>retrieval-&gt;analysis</title>
<path fill="none" stroke="#434343" d="M663.14,-140.3C692.99,-140.3 726.66,-140.3 726.66,-140.3 726.66,-140.3 726.66,-62.07 726.66,-62.07"/>
<polygon fill="#434343" stroke="#434343" points="728.76,-62.07 726.66,-56.07 724.56,-62.07 728.76,-62.07"/>
<text text-anchor="middle" x="694.71" y="-85.7" font-family="Helvetica,sans-Serif" font-size="7.00">Top Candidates</text>
</g>
<!-- analysis&#45;&gt;output_json -->
<g id="edge9" class="edge">
<title>analysis-&gt;output_json</title>
<path fill="none" stroke="#434343" d="M829.94,-35.3C879.68,-35.3 944,-35.3 944,-35.3 944,-35.3 944,-334.22 944,-334.22"/>
<polygon fill="#434343" stroke="#434343" points="941.9,-334.22 944,-340.22 946.1,-334.22 941.9,-334.22"/>
</g>
</g>
</svg>phviz (1).svg…]()


## Table of Contents
- [System Architecture](#system-architecture)
- [Technical Implementation Details](#technical-implementation-details)
  - [Ingestion and Structuring](#ingestion-and-structuring)
  - [Query Decomposition](#query-decomposition)
  - [Hybrid Retrieval Engine](#hybrid-retrieval-engine)
  - [Re-ranking and Sub-Section Analysis](#re-ranking-and-sub-section-analysis)
- [Model Selection and Optimization](#model-selection-and-optimization)
- [Usage Instructions](#usage-instructions)
- [Directory Structure](#directory-structure)
- [Build and Run](#build-and-run)
- [Project Files](#project-files)

## System Architecture
The solution is implemented as a sequential, multi-stage pipeline designed to maximize both recall and precision, directly addressing the hackathon's scoring criteria.

The architectural flow is as follows:

1. **Ingestion and Structuring**: Each PDF in the input collection is parsed to extract raw text and identify structural elements (headings, text blocks). This output is segmented into semantically coherent, page-aware chunks.
2. **Indexing**: The processed text chunks are indexed in two parallel systems: a lexical index using the BM25Okapi algorithm and a semantic vector index using embeddings generated by the core neural model.
3. **Query Decomposition**: The input JSON file (containing the persona and job-to-be-done) is deconstructed into a set of distinct, specific sub-queries to improve search focus.
4. **Stage 1: Hybrid Retrieval**: A broad search is executed across the document collection. The lexical and semantic indexes are queried in parallel, and the results are fused using a weighted scoring mechanism to produce a candidate list of promising document sections.
5. **Stage 2: Re-ranking and Analysis**: The candidate list from Stage 1 is subjected to a more focused, precision-oriented analysis. This stage re-ranks the candidates to determine the final importance rank and performs a granular, sentence-level analysis within each section to generate the refined_text.
6. **Output Generation**: The final ranked and analyzed results are formatted into the required output json structure, including all specified metadata.

## Technical Implementation Details

### Ingestion and Structuring
This module is responsible for converting raw PDF files into a structured, searchable format.

- **PDF Parsing**: The PyMuPDF library is used for its high performance and detailed text extraction capabilities, including font information, boldness, and bounding box coordinates for every text span.
- **Structural Analysis**: A dynamic heading detection logic is employed. Instead of relying on static font sizes, the module first analyzes the font distribution across the entire document to establish a baseline for body text. Headings (H1, H2, H3) are then identified based on relative font size, bold weight, and positional information (e.g., centered text for titles).
- **Page-Aware Chunking**: Creation of granular, page-aware chunks on basis of which content is segmented based not just on headings, but on the individual text blocks on each page under that heading. This ensures that every resulting chunk has an accurate page number, which is critical for the final output's citation accuracy.

### Query Decomposition
To effectively handle the complex user intent, the user specified job string is parsed into multiple sub-queries. This is achieved with a rule-based approach, splitting the task description by common delimiters such as commas and conjunction. Each resulting sub-task is then combined with the persona's role to form a set of focused queries.

### Hybrid Retrieval Engine
This module combines two distinct search paradigms to ensure comprehensive retrieval.

- **Lexical Search**: Implemented using the `rank_bm25` library. This component creates an inverted index of the document chunks and scores them based on the Okapi BM25 algorithm, used for matching exact keywords and domain-specific terms.
- **Semantic Search**: Implemented using the sentence-transformer library. It generates dense vector embeddings for each document chunk. Retrieval is performed by calculating the cosine similarity between the query embeddings and the document chunk embeddings.
- **Score Fusion**: The scores from both search systems are normalized to a common scale (0-1) and then combined using a weighted linear formula:
  ```
  final_score = (α * normalized_semantic_score) + ((1 - α) * normalized_bm25_score)
  ```
  
### Re-ranking and Sub-Section Analysis
This final stage refines the results from the hybrid retriever.

- **Sentence Segmentation**: For each top-ranked chunk, the content is first split into individual sentences.
- **Sentence-Level Scoring**: Each sentence is then embedded using the same core model. Its relevance is determined by calculating the maximum cosine similarity against the set of decomposed query vectors.
- **Refined Text Generation**: The top 2-3 highest-scoring sentences are concatenated to form the refined text for the output. This provides a concise, highly relevant summary of the larger section.
- **Final Ranking**: The importance rank is determined by re-sorting the candidate chunks based on the maximum sentence similarity score found within them.

## Model Selection and Optimization
The performance of the system is critically dependent on the choice and optimization of the embedding model.

- **Model Selection**: The `Alibaba-NLP/gte-large-en-v1.5` model was selected as the base model. The rationale for this choice is its optimal balance of high retrieval performance on the MTEB leaderboard, a manageable initial size (~434MB), and a large 8192-token context window, which is advantageous for processing long document sections.
- **Inference Engine**: ONNX Runtime is used as the execution backend.
- **Quantization**: Post-Training Static Quantization (PTQ) was applied. The FP32 model was converted to INT8. This process reduces the on-disk model size to approximately 110MB and provides a significant (4x) inference speedup on CPU. The quantization process uses calibration dataset to pre-calculate activation scales, preserving accuracy more effectively than dynamic quantization.

## Usage Instructions
The solution is fully containerized with Docker for easy and reproducible execution. Ensure your project directory is structured as follows before running the Docker commands. The repository contains two main folders: round1a and round1b, each with its own independent solution and Docker environment.

```
.
├── round1a/
│   ├── input/
│   │   ├── 51.pdf
│   │   └── request.json
│   ├── output/
│   ├── model/
│   │   └── model.onnx
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── round1b/
│   ├── input/
│   │   ├── 51.pdf
│   │   └── request.json
│   ├── output/
│   ├── model/
│   │   └── model.onnx
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── README.md
```

- **`round1a/`**: Contains the solution for Round 1A, including its own `Dockerfile`, `main.py`, `requirements.txt`, and dedicated `input/`, `output/`, and `model/` directories.
- **`round1b/`**: Contains the solution for Round 1B (Persona-Driven Document Intelligence), with a similar structure to `round1a/` but with its own independent implementation and `Dockerfile`.

## Build and Run

Each round's solution can be built and run independently using its respective `Dockerfile`.

### Round 1A

```bash
cd round1a
```

**Build the Docker Image**:
```bash
docker build --platform linux/amd64 -t structurer-round1a:latest .
```

**Run the Container**:
```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  structurer-round1a:latest
```

### Round 1B

```bash
cd round1b
```

**Build the Docker Image**:
```bash
docker build --platform linux/amd64 -t persona-solver-round1b:latest .
```

**Run the Container**:
```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  persona-solver-round1b:latest
```

For both rounds, the script will process all PDFs in the respective `input/` directory and generate a single `output.json` file in the corresponding `output/` directory.

## Project Files
Each round (`round1a/` and `round1b/`) contains the following files:

- `main.py`: The main Python script containing the end-to-end application logic for the respective round.
- `Dockerfile`: Defines the Docker image, dependencies, and execution environment for the respective round.
- `requirements.txt`: Lists all required Python packages for the respective round.
- `model/` : Contains the pre-optimized `model.onnx` file and any associated configuration files required by ONNX Runtime for the respective round.
