#
# This script performs a one-time conversion of a Hugging Face Sentence Transformer
# model to a quantized ONNX model for high-performance CPU inference.
# It should be run as part of the Docker build process to generate the
# optimized model artifacts that the main application will use.
##############################################################################
import os
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, AutoModel

# --- Configuration ---
# The base model to be optimized
BASE_MODEL_NAME = 'Alibaba-NLP/gte-large-en-v1.5'
# The final directory where the optimized model will be saved inside the container
QUANTIZED_MODEL_PATH = "./model"

def main():
    """
    Downloads the base model, converts it to ONNX, and applies INT8 static
    quantization. The final artifacts are saved to the specified path.
    """
    print(f"--- Starting model optimization for {BASE_MODEL_NAME} ---")

    if not os.path.exists(QUANTIZED_MODEL_PATH):
        os.makedirs(QUANTIZED_MODEL_PATH)

    # --- Step 1: Download base model and tokenizer from Hugging Face ---
    print("Downloading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # Save the base model locally to prepare for ONNX export
    base_model_dir = "./base_model"
    tokenizer.save_pretrained(base_model_dir)
    model.save_pretrained(base_model_dir)

    # --- Step 2: Create a quantizer from the local model files ---
    print(f"Loading model from {base_model_dir} for quantization...")
    quantizer = ORTQuantizer.from_pretrained(base_model_dir)

    # --- Step 3: Define the quantization configuration ---
    print("Creating INT8 static quantization configuration...")
    # Using a configuration optimized for modern CPUs with AVX2/AVX-512 VNNI support
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)

    # --- Step 4: Define a calibration dataset ---
    # A small, representative dataset is needed for static quantization to
    # analyze the distribution of activations and calculate optimal scales.
    print("Defining calibration dataset...")
    calibration_dataset = [
        "This is a sample sentence for model calibration.",
        "Persona-driven document intelligence requires deep semantic understanding.",
        "The investment analyst needs to analyze revenue trends and market positioning.",
        "Quantizing the model significantly improves CPU inference speed.",
        "The quick brown fox jumps over the lazy dog."
    ]

    def calibration_data_reader(dataset):
        for text in dataset:
            yield {"input_ids": tokenizer(text, return_tensors="np").input_ids}

    # --- Step 5: Run the quantization process ---
    print("Starting quantization process. This may take a few minutes...")
    quantizer.quantize(
        save_dir=QUANTIZED_MODEL_PATH,
        quantization_config=qconfig,
        # The calibration function is now a generator
        calibration_tensors_range=quantizer.fit(
            dataset=calibration_dataset,
            calibration_config=qconfig,
            operators_to_quantize=["MatMul", "Add"],
        ),
    )

    print(f"--- Optimization Complete ---")
    print(f"Quantized model and tokenizer saved to: {QUANTIZED_MODEL_PATH}")

