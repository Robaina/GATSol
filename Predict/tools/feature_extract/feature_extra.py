import os
import sys
from pathlib import Path
import pandas as pd
import iFeatureOmegaCLI
import torch
import numpy as np
from torch_geometric.data import Data
import warnings
import argparse
import time
import gc
import signal

warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm
import multiprocessing as mp
import logging
from torch_geometric.utils import add_self_loops
import contextlib
import io
import esm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./tools/feature_extract/log.log")
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent / "NEED_to_PREPARE"

# Global model cache (will be initialized in worker processes)
MODEL = None
BATCH_CONVERTER = None

# Set maximum timeout for processing a single file
MAX_FILE_PROCESSING_TIME = 300  # 5 minutes

def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutError("Processing timed out")

def initialize_model():
    """Initialize ESM-1b model (to be called once per process)"""
    global MODEL, BATCH_CONVERTER
    if MODEL is None:
        logger.info("Loading ESM-1b model...")
        start_time = time.time()
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        BATCH_CONVERTER = alphabet.get_batch_converter()
        MODEL = model
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        # Move model to GPU if available
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()
            logger.info("Model moved to GPU")
        MODEL.eval()  # Set to evaluation mode


def name_seq_dict(path):
    """Load sequence dictionary from CSV file"""
    pdb_chain_list = pd.read_csv(path, header=0)
    dict_pdb_chain = pdb_chain_list.set_index("id")["sequence"].to_dict()
    return dict_pdb_chain


def process_file(file):
    """Process a single protein file"""
    global MODEL, BATCH_CONVERTER
    
    # Initialize model if not already done
    if MODEL is None:
        initialize_model()
    
    dict_path = str(BASE_DIR / "list.csv")
    seq_dict = name_seq_dict(dict_path)
    
    # Path definitions
    cm_directory = str(BASE_DIR / "cm")
    fasta_directory = str(BASE_DIR / "fasta")
    pkl_directory = str(BASE_DIR / "pkl")

    # Create output directory if it doesn't exist
    os.makedirs(pkl_directory, exist_ok=True)
    
    # Skip if output already exists
    pkl_path = os.path.join(pkl_directory, f"{file}.pkl")
    if os.path.exists(pkl_path):
        return True, file, "Already processed"

    try:
        cm_path = os.path.join(cm_directory, f"{file}.cm")
        fasta_path = os.path.join(fasta_directory, f"{file}.fasta")
        
        # Check if input files exist
        if not os.path.exists(cm_path):
            return False, file, f"Contact map file not found: {cm_path}"
        if not os.path.exists(fasta_path):
            return False, file, f"FASTA file not found: {fasta_path}"

        # ESM feature extraction
        start_time = time.time()
        batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER([(file, seq_dict[file])])
        
        # Check if model is on GPU and move tokens accordingly
        model_device = next(MODEL.parameters()).device
        logger.info(f"Model is on device: {model_device}")
        batch_tokens = batch_tokens.to(model_device)
            
        with torch.no_grad():
            results = MODEL(batch_tokens, repr_layers=[33], return_contacts=True)
        
        # CRITICAL FIX: Always explicitly move results to CPU 
        cpu_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"Moving tensor '{k}' from {v.device} to cpu")
                cpu_results[k] = v.cpu().detach()
            else:
                cpu_results[k] = v

        # iFeature extraction (suppress stdout)
        protein = iFeatureOmegaCLI.iProtein(fasta_path)
        null_file = io.StringIO()
        with contextlib.redirect_stdout(null_file):
            protein.import_parameters("./tools/feature_extract/Protein_parameters_setting.json")
            protein.get_descriptor("BLOSUM62")

        # CRITICAL FIX: Ensure both tensors are explicitly on CPU before concatenation
        node_feature = torch.from_numpy((protein.encodings.values.reshape(-1, 20))).float()
        node_feature1 = cpu_results["representations"][33][0, 1:-1].reshape(-1, 1280)
        
        # Debug device info
        logger.info(f"Device check before concat - iFeature: {node_feature.device}, ESM: {node_feature1.device}")
        
        # Force both to explicit CPU
        node_feature = node_feature.cpu()
        node_feature1 = node_feature1.cpu()
        
        # Verify devices are the same now
        logger.info(f"Device check after forcing CPU - iFeature: {node_feature.device}, ESM: {node_feature1.device}")
        
        # Combine features
        node_features = torch.cat((node_feature, node_feature1), 1)

        # Load contact map
        with open(cm_path, "r") as f:
            content = f.read()
            data = np.array([
                list(map(float, line.split(",")))
                for line in content.split("\n")
                if line
            ])
            edges = torch.from_numpy(data).type(torch.LongTensor)

        # Create PyG data object
        label = torch.tensor(0).reshape(1,)
        data = Data(x=node_features, edge_index=edges.t().contiguous() - 1, y=label)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=node_features.shape[0])
        
        # Save to pickle
        with open(pkl_path, "wb") as fpkl:
            pickle.dump(data, fpkl)
            
        processing_time = time.time() - start_time
        return True, file, f"Processed in {processing_time:.2f}s"
        
    except Exception as e:
        return False, file, f"Error: {str(e)}"


def batch_process_feature_extraction(file_ids, batch_size=10, use_gpu=True):
    """Process feature extraction in small batches to manage memory"""
    total_files = len(file_ids)
    success_count = 0
    error_count = 0
    
    logger.info(f"Batch processing {total_files} files with batch size {batch_size}")
    
    for batch_idx in range(0, total_files, batch_size):
        batch_end = min(batch_idx + batch_size, total_files)
        batch_files = file_ids[batch_idx:batch_end]
        batch_size = len(batch_files)
        
        logger.info(f"Processing batch {batch_idx//batch_size + 1}: files {batch_idx+1}-{batch_end} of {total_files}")
        
        # Initialize model for this batch
        initialize_model()
        
        # Process batch
        batch_results = []
        for i, file_id in enumerate(batch_files):
            logger.info(f"Processing file {batch_idx+i+1}/{total_files}: {file_id}")
            success, _, message = process_file(file_id)
            batch_results.append((success, file_id, message))
            
            # Force cleanup after each file when using GPU
            if use_gpu and torch.cuda.is_available():
                # Explicitly clean up CUDA memory
                torch.cuda.empty_cache()
                gc.collect()
                
        # Count successes and errors
        batch_success = sum(1 for res in batch_results if res[0])
        batch_error = len(batch_results) - batch_success
        
        success_count += batch_success
        error_count += batch_error
        
        logger.info(f"Batch {batch_idx//batch_size + 1} completed: {batch_success} successes, {batch_error} errors")
        
        # Clean up model to free memory between batches
        global MODEL, BATCH_CONVERTER
        if MODEL is not None:
            if use_gpu and torch.cuda.is_available():
                MODEL.cpu()
            MODEL = None
            BATCH_CONVERTER = None
            gc.collect()
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Short pause between batches to let system stabilize
        time.sleep(2)
        
    return success_count, error_count


def process_cpu_only(file_ids):
    """Process feature extraction completely on CPU (no GPU)"""
    global MODEL, BATCH_CONVERTER
    
    # Force model to CPU
    logger.info("Loading ESM model on CPU only...")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    MODEL = model.cpu()
    BATCH_CONVERTER = alphabet.get_batch_converter()
    MODEL.eval()
    
    success_count = 0
    error_count = 0
    
    for i, file_id in enumerate(file_ids):
        logger.info(f"Processing file {i+1}/{len(file_ids)} on CPU: {file_id}")
        try:
            success, _, message = process_file(file_id)
            if success:
                success_count += 1
                logger.info(f"Success for {file_id}: {message}")
            else:
                error_count += 1
                logger.error(f"Failed for {file_id}: {message}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {file_id}: {str(e)}")
    
    return success_count, error_count


def main(args):
    """Main function for feature extraction"""
    # Load sequence dictionary
    dict_path = str(BASE_DIR / "list.csv")
    try:
        name_dict = name_seq_dict(dict_path)
        file_names = list(name_dict.keys())
    except Exception as e:
        logger.error(f"Error loading sequence dictionary: {str(e)}")
        return
    
    total_files = len(file_names)
    logger.info(f"Starting feature extraction for {total_files} proteins")
    
    # Create output directory
    pkl_directory = str(BASE_DIR / "pkl")
    os.makedirs(pkl_directory, exist_ok=True)
    
    # Check if CUDA is available
    use_gpu = torch.cuda.is_available() and not args.cpu_only
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    else:
        logger.info("Using CPU only for processing")
        
    # Print PyTorch version for debugging
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    start_time = time.time()
    
    # CPU-only mode (fallback for GPU issues)
    if args.cpu_only:
        logger.info("Using CPU-only mode as requested")
        success_count, error_count = process_cpu_only(file_names)
    
    # Batch processing mode
    elif args.batch:
        # Calculate optimal batch size
        if args.batch_size:
            batch_size = args.batch_size
        elif use_gpu:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem > 16:
                batch_size = 8 
            elif gpu_mem > 8:
                batch_size = 4
            else:
                batch_size = 2
        else:
            batch_size = 10
            
        success_count, error_count = batch_process_feature_extraction(
            file_names, batch_size=batch_size, use_gpu=use_gpu
        )
    
    # Sequential processing (fallback mode)
    else:
        logger.info("Processing files sequentially")
        success_count = 0
        error_count = 0
        
        # Initialize model
        initialize_model()
        
        for i, file in enumerate(file_names):
            logger.info(f"Processing file {i+1}/{total_files}: {file}")
            success, _, message = process_file(file)
            
            if success:
                success_count += 1
                logger.info(f"Success: {message}")
            else:
                error_count += 1
                logger.error(f"Error: {message}")
                
            # Clean up every few files to prevent memory build-up
            if (i+1) % 5 == 0 and use_gpu:
                gc.collect()
                torch.cuda.empty_cache()
    
    # Report final statistics
    elapsed = time.time() - start_time
    logger.info(
        f"Feature extraction completed in {elapsed:.2f} seconds - "
        f"Success: {success_count}/{total_files}, Errors: {error_count}/{total_files}"
    )
    
    # Verify outputs
    expected_files = total_files
    actual_files = len(list(Path(pkl_directory).glob("*.pkl")))
    logger.info(f"Expected {expected_files} output files, found {actual_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction for GATSol")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--sequential", action="store_true", help="Force sequential processing")
    parser.add_argument("--batch", action="store_true", help="Process in batches to manage memory")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU usage even if available")
    parser.add_argument("--file", type=str, help="Process a single file")
    parser.add_argument("--output", type=str, help="Output path for single file processing")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per file in seconds")
    args = parser.parse_args()
    
    # Update max processing time if specified
    if args.timeout:
        MAX_FILE_PROCESSING_TIME = args.timeout
        
    # Run main processing
    try:
        main(args)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)