#!/usr/bin/env python3
"""
GATSol Protein Solubility Predictor - Optimized Version

This script serves as the main entry point for the GATSol protein solubility prediction pipeline.
It handles the entire workflow from processing input files to generating predictions using
a Graph Attention Network (GAT) model, supporting both CPU and GPU execution.
"""

import argparse
import logging
import sys
import pickle
import shutil
import subprocess
import time
import os
import threading
import select
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOOLS_PATH = Path("/app/Predict/tools")
BASE_DIR = Path("/app/Predict/NEED_to_PREPARE")


class Timer:
    """Simple timer for profiling code execution."""
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f"Time for {self.name}: {self.interval:.2f} seconds")


def parse_fasta(fasta_path: Path) -> pd.DataFrame:
    """
    Parse a FASTA file into a DataFrame with 'id' and 'sequence' columns.

    Args:
        fasta_path: Path to the input FASTA file

    Returns:
        pd.DataFrame: DataFrame containing sequence IDs and sequences
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences.append(
                        {"id": current_id, "sequence": "".join(current_seq)}
                    )
                current_id = line[1:].split()[0]  # Take first word after '>' as ID
                current_seq = []
            else:
                current_seq.append(line)

    # Don't forget to add the last sequence
    if current_id is not None:
        sequences.append({"id": current_id, "sequence": "".join(current_seq)})

    return pd.DataFrame(sequences)


def load_input_sequences(input_path: Path) -> pd.DataFrame:
    """
    Load sequences from either a CSV file or a FASTA file.

    Args:
        input_path: Path to input file (either CSV or FASTA)

    Returns:
        pd.DataFrame: DataFrame containing sequence IDs and sequences
    """
    file_ext = input_path.suffix.lower()

    if file_ext == ".csv":
        df = pd.read_csv(input_path)
        if not {"id", "sequence"}.issubset(df.columns):
            raise ValueError("CSV file must contain 'id' and 'sequence' columns")
    elif file_ext in [".fasta", ".fa", ".faa"]:
        df = parse_fasta(input_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Please provide either a CSV or FASTA/FAA file."
        )

    return df


def setup_model_weights(weights_dir=None):
    """Set up ESM model weights from local directory if provided."""
    if weights_dir is None:
        return True

    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"Warning: Weights directory {weights_dir} does not exist")
        return False

    cache_dir = Path("/root/.cache/torch/hub/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_files = [
        "esm1b_t33_650M_UR50S.pt",
        "esm1b_t33_650M_UR50S-contact-regression.pt",
    ]

    success = True
    for filename in weight_files:
        src_file = weights_path / filename
        dst_file = cache_dir / filename

        if not src_file.exists():
            print(f"Warning: ESM weight file {filename} not found in {weights_dir}")
            success = False
            continue

        try:
            import shutil

            shutil.copy2(src_file, dst_file)
            print(f"Copied {filename} to PyTorch cache")
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            success = False

    return success


@dataclass
class ModelConfig:
    """Configuration parameters for the GAT model."""

    in_channels: int = 1300
    hidden_channels: int = 1024
    num_classes: int = 1
    num_heads: int = 16
    num_layers: int = 2
    batch_size: int = 16  # Increased from 1 to improve GPU utilization


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Determine the appropriate device (CPU/GPU) to use.

    Args:
        device_str: Optional string specifying device ('cpu', 'cuda', or None)

    Returns:
        torch.device: The selected device
    """
    if device_str is not None:
        if device_str not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)

    # Auto-detect device
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected and will be used for computation")
        return torch.device("cuda")
    else:
        logger.info("No CUDA GPU detected. Using CPU for computation")
        return torch.device("cpu")


class GATClassifier(nn.Module):
    """
    Graph Attention Network (GAT) for protein solubility prediction.
    """

    def __init__(self, in_channels, hidden_channels, num_heads, num_layers):
        super(GATClassifier, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GATConv(in_channels, hidden_channels, heads=num_heads)
                )
            else:
                self.convs.append(
                    GATConv(
                        hidden_channels * num_heads, hidden_channels, heads=num_heads
                    )
                )
        self.lin1 = nn.Linear(hidden_channels * num_heads, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x.squeeze()


def setup_directories(BASE_DIR: Path) -> Path:
    """Create and return the base working directory structure."""
    # Create all required directories
    for subdir in ["fasta", "pdb", "cm", "pkl"]:
        (BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {BASE_DIR / subdir}")


def write_fasta_files(sequences_df: pd.DataFrame, fasta_dir: Path) -> None:
    """
    Write FASTA files for each sequence in the dataframe.

    Args:
        sequences_df: DataFrame with 'id' and 'sequence' columns
        fasta_dir: Directory where FASTA files will be written
    """
    # Bulk write FASTA files rather than one-by-one
    with Timer("Writing FASTA files"):
        for _, row in sequences_df.iterrows():
            fasta_path = fasta_dir / f"{row['id']}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">{row['id']}\n{row['sequence']}\n")
        logger.debug(f"Created {len(sequences_df)} FASTA files")


def copy_pdb_files(pdb_dir: Path, target_dir: Path, sequence_ids: List[str]) -> None:
    """
    Copy required PDB files to the working directory.

    Args:
        pdb_dir: Source directory containing PDB files
        target_dir: Target directory for PDB files
        sequence_ids: List of sequence IDs to copy
    """
    with Timer("Copying PDB files"):
        missing_files = []
        
        for seq_id in sequence_ids:
            pdb_path = pdb_dir / f"{seq_id}.pdb"
            if not pdb_path.exists():
                missing_files.append(str(pdb_path))
                continue
            shutil.copy2(pdb_path, target_dir)
        
        if missing_files:
            if len(missing_files) <= 5:
                missing_str = ", ".join(missing_files)
            else:
                missing_str = f"{len(missing_files)} PDB files"
            raise FileNotFoundError(f"Required PDB files not found: {missing_str}")
        
        logger.debug(f"Copied {len(sequence_ids)} PDB files")


def process_pdb_to_cm(
    pdb_path: Path, cm_path: Path, contact_threshold: float = 10.0
) -> None:
    """
    Process a single PDB file to generate its contact map.

    Args:
        pdb_path: Path to input PDB file
        cm_path: Path to output contact map file
        contact_threshold: Distance threshold for contacts (default: 10.0 Angstroms)
    """
    # Skip if output file already exists
    if cm_path.exists():
        return
        
    try:
        subprocess.run(
            [
                "python",
                str(TOOLS_PATH / "pdb_to_cm/pdb_to_cm.py"),
                str(pdb_path),
                str(cm_path),
                "-t",
                str(contact_threshold),
            ],
            check=True,
            stdout=subprocess.DEVNULL,  # Reduce stdout noise
            stderr=subprocess.PIPE,     # Capture errors
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing PDB file {pdb_path}: {e.stderr.decode()}")
        raise


def process_pdb_to_cm_wrapper(args):
    """Wrapper for process_pdb_to_cm to use with concurrent.futures."""
    pdb_path, cm_path, contact_threshold = args
    try:
        process_pdb_to_cm(pdb_path, cm_path, contact_threshold)
        return True
    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}")
        return False


def process_input_files(BASE_DIR: Path, n_workers: int = None) -> None:
    """
    Process all PDB files to generate contact maps with parallel execution.

    Args:
        BASE_DIR: Base directory containing pdb and cm subdirectories
        n_workers: Number of parallel workers to use (default: CPU count)
    """
    if n_workers is None:
        n_workers = min(32, os.cpu_count())  # Limit to reasonable number
        
    pdb_dir = BASE_DIR / "pdb"
    cm_dir = BASE_DIR / "cm"

    # Ensure output directory exists
    cm_dir.mkdir(exist_ok=True)

    # Process each PDB file
    pdb_files = list(pdb_dir.glob("*.pdb"))
    cm_files = [cm_dir / f"{pdb_file.stem}.cm" for pdb_file in pdb_files]
    
    # List of (pdb_path, cm_path, threshold) for each file
    tasks = [(pdb, cm, 10.0) for pdb, cm in zip(pdb_files, cm_files)]
    
    logger.info(f"Processing {len(pdb_files)} PDB files using {n_workers} workers")
    
    with Timer("Processing PDB files to contact maps"):
        # Process in parallel
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(process_pdb_to_cm_wrapper, tasks))
                
            success_count = sum(results)
            if success_count < len(tasks):
                logger.warning(f"Failed to process {len(tasks) - success_count} PDB files")
        else:
            # Fallback to sequential processing
            for idx, (pdb_file, cm_file, threshold) in enumerate(tasks, 1):
                logger.info(f"Processing PDB file {idx}/{len(tasks)}: {pdb_file.stem}")
                process_pdb_to_cm(pdb_file, cm_file, threshold)

    logger.info("Successfully processed PDB files to contact maps")


def validate_feature_inputs(base_dir: Path) -> bool:
    """
    Validate that all required input files for feature extraction exist.
    
    Args:
        base_dir: Base directory containing subdirectories
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    cm_dir = base_dir / "cm"
    
    # Check if contact map directory exists and contains files
    if not cm_dir.exists():
        logger.error(f"Contact map directory not found: {cm_dir}")
        return False
    
    cm_files = list(cm_dir.glob("*.cm"))
    if not cm_files:
        logger.error("No contact map files found for feature extraction")
        return False
    
    # Sample validation on a few files
    for sample_file in cm_files[:3]:  # Check first 3 files
        try:
            with open(sample_file, 'r') as f:
                first_line = f.readline().strip()
                # Simple validation - contact map files typically start with a header or numbers
                if not first_line:
                    logger.warning(f"Empty contact map file detected: {sample_file}")
        except Exception as e:
            logger.error(f"Error reading contact map file {sample_file}: {str(e)}")
            return False
    
    logger.info(f"Found {len(cm_files)} valid contact map files for feature extraction")
    return True


def _stream_output(pipe, level):
    """Helper function to stream subprocess output to logger."""
    for line in iter(pipe.readline, ''):
        line = line.strip()
        if not line:
            continue
            
        if level == "INFO":
            logger.info(f"Feature extraction: {line}")
        else:
            logger.warning(f"Feature extraction: {line}")


def extract_features(tools_path: Path, batch_size: int = 4, timeout: int = 600, cpu_only: bool = False) -> None:
    """
    Extract features from processed files with enhanced batch control.
    
    Args:
        tools_path: Path to the tools directory
        batch_size: Number of proteins to process in each batch
        timeout: Maximum time in seconds to wait for processing a single protein
        cpu_only: Whether to force CPU-only processing
    """
    script_path = tools_path / "feature_extract" / "feature_extra.py"
    
    # Verify the script exists
    if not script_path.exists():
        # Try alternative name (in case of typo in path)
        alt_script_path = tools_path / "feature_extract" / "feature_extract.py"
        if alt_script_path.exists():
            script_path = alt_script_path
            logger.warning(f"Using alternative script path: {script_path}")
        else:
            raise FileNotFoundError(f"Feature extraction script not found at: {script_path}")
    
    # Build command with appropriate arguments
    cmd = [
        "python", 
        str(script_path),
        "--batch",  # Always use batch mode for better memory management
        "--batch-size", str(batch_size),
        "--timeout", str(timeout)
    ]
    
    # Add CPU-only flag if requested
    if cpu_only:
        cmd.append("--cpu-only")
        logger.info("Using CPU-only mode for feature extraction")
    
    logger.info(f"Starting feature extraction with batch size {batch_size}")
    
    # Try CPU-only as fallback if GPU version fails
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            # Run with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Monitor process output in real-time
            stdout_thread = threading.Thread(target=_stream_output, args=(process.stdout, "INFO"))
            stderr_thread = threading.Thread(target=_stream_output, args=(process.stderr, "ERROR"))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            exit_code = process.wait()
            
            # Wait for output threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Check return code
            if exit_code != 0:
                logger.error(f"Feature extraction failed with return code {exit_code}")
                if attempt < max_attempts - 1 and not cpu_only:
                    logger.warning("Retrying with CPU-only mode...")
                    cmd.append("--cpu-only")
                    continue
                raise subprocess.CalledProcessError(exit_code, cmd)
            
            logger.info("Successfully extracted features")
            break
            
        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}")
            
            # Try CPU-only mode on second attempt
            if attempt < max_attempts - 1 and not cpu_only:
                logger.warning("Retrying with CPU-only mode...")
                cmd.append("--cpu-only")
                continue
                
            # Check if we have any output files
            pkl_dir = BASE_DIR / "pkl"
            if pkl_dir.exists() and list(pkl_dir.glob("*.pkl")):
                logger.info("Some feature files were generated, continuing with partial results")
                break
            else:
                raise


def load_test_dataset(
    pkl_path: Path, file_names: List[str], device: torch.device
) -> List[Data]:
    """
    Load and prepare test dataset.

    Args:
        pkl_path: Path to directory containing pickle files
        file_names: List of file names to load
        device: torch device to use

    Returns:
        List of PyTorch Geometric Data objects
    """
    with Timer("Loading dataset"):
        test_dataset = []
        total_files = len(file_names)
        
        # Process in batches to reduce logging overhead
        batch_size = 10
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            logger.info(f"Loading dataset files {batch_start+1}-{batch_end}/{total_files}")
            
            for filename in file_names[batch_start:batch_end]:
                file_path = pkl_path / f"{filename}.pkl"
                with open(file_path, "rb") as f:
                    data = pickle.load(f).to(device)
                test_dataset.append(data)
                
        logger.info("Dataset loading completed")
    return test_dataset


def load_model(
    model: nn.Module, model_path: Union[str, Path], device: torch.device
) -> nn.Module:
    """
    Load model weights with proper device handling.

    Args:
        model: The model instance
        model_path: Path to the model weights
        device: Target device for the model

    Returns:
        The loaded model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    # Load the state dict with proper device mapping
    with Timer("Loading model"):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

    return model


def make_predictions(model, device, loader):
    """
    Generate predictions using the trained model.

    Args:
        model: The GAT model
        device: torch device to use
        loader: DataLoader containing test data

    Returns:
        Tensor of predictions
    """
    with Timer("Making predictions"):
        model.eval()
        y_hat = torch.tensor([]).to(device)

        total_batches = len(loader)
        with torch.no_grad():
            for idx, data in enumerate(loader, 1):
                if idx % max(1, total_batches // 10) == 0:  # Log approximately 10 times total
                    logger.info(f"Processing batch {idx}/{total_batches}")
                data = data.to(device)
                output = model(data)
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                y_hat = torch.cat((y_hat, output), 0)

        logger.info("Prediction generation completed")
    return y_hat


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    # Set up model weights
    if not setup_model_weights(args.esm_weights_dir):
        print("Error: Model weights could not be set up. Exiting.")
        sys.exit(1)

    logger.info("Starting GATSol prediction pipeline")

    # Setup device
    device = get_device(args.device)

    # Setup directories
    setup_directories(BASE_DIR)

    # Read sequence data - now supports both CSV and FASTA
    with Timer("Loading input sequences"):
        sequences_df = load_input_sequences(Path(args.input))

    # Save as CSV for compatibility with rest of pipeline
    sequences_df.to_csv(BASE_DIR / "list.csv", index=False)

    # Generate FASTA files from sequences
    write_fasta_files(sequences_df, BASE_DIR / "fasta")
    logger.info("Generated FASTA files from sequences")

    # Copy PDB files
    copy_pdb_files(Path(args.pdb_dir), BASE_DIR / "pdb", sequences_df["id"].tolist())
    logger.info("Copied PDB files to working directory")

    # Process files with parallelism if enabled
    process_input_files(BASE_DIR, n_workers=args.workers)
    
    # Validate contact map files before feature extraction
    if not args.skip_feature_validation:
        validate_feature_inputs(BASE_DIR)

    # Extract features with batch control
    extract_features(
        TOOLS_PATH, 
        batch_size=args.feature_batch_size,
        timeout=args.feature_timeout,
        cpu_only=args.cpu_only_features
    )

    # Verify feature extraction results
    pkl_dir = BASE_DIR / "pkl"
    expected_pkl_count = len(sequences_df)
    actual_pkl_count = len(list(pkl_dir.glob("*.pkl")))
    
    if actual_pkl_count < expected_pkl_count:
        logger.warning(f"Expected {expected_pkl_count} feature files, but found only {actual_pkl_count}")
        if actual_pkl_count < expected_pkl_count * 0.5:  # Less than 50% success
            logger.error("Feature extraction largely failed. Exiting.")
            sys.exit(1)
        else:
            logger.warning("Continuing with partial results")

    # Setup model
    config = ModelConfig()
    # Override batch size if specified
    if args.batch_size:
        config.batch_size = args.batch_size
        
    model = GATClassifier(
        config.in_channels, config.hidden_channels, config.num_heads, config.num_layers
    )

    # Load model weights with proper device handling
    model_path = Path(args.gatsol_weights_dir) / "best_model.pt"
    model = load_model(model, model_path, device)

    # Prepare dataset with available PKL files
    available_ids = [f.stem for f in pkl_dir.glob("*.pkl")]
    logger.info(f"Loading dataset with {len(available_ids)} processed proteins")
    
    test_dataset = load_test_dataset(
        pkl_dir, available_ids, device
    )
    
    # Optimize DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0 if device.type == 'cuda' else min(4, os.cpu_count() or 1),
        pin_memory=False #device.type == 'cuda'
    )
    
    predictions = make_predictions(model, device, test_loader)

    # Save results
    with Timer("Saving predictions"):
        # Create results dataframe
        results_df = pd.DataFrame({"id": available_ids})
        results_df["Solubility_hat"] = predictions.cpu().numpy()
        
        # Merge with original sequence data
        output_df = pd.merge(sequences_df, results_df, on="id", how="left")
        
        # Save to output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_dir / "predictions.csv", index=False)
        logger.info(f"Predictions saved to: {output_dir/'predictions.csv'}")

    # Cleanup temporary directories
    if not args.keep_temp:
        for temp_dir in ["cm", "pkl"]:
            temp_path = BASE_DIR / temp_dir
            if temp_path.exists():
                shutil.rmtree(temp_path)
                logger.debug(f"Cleaned up temporary directory: {temp_path}")
                
    logger.info("GATSol prediction pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GATSol: Protein solubility prediction using Graph Attention Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (CSV with id,sequence columns or FASTA format)",
    )

    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Directory containing PDB files (named as id.pdb)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predictions will be saved",
    )

    parser.add_argument(
        "--gatsol-weights-dir",
        type=str,
        required=True,
        help="Directory containing GATSol model weights (best_model.pt)",
    )

    parser.add_argument(
        "--esm-weights-dir",
        type=str,
        default=None,
        required=False,
        help="Directory containing ESM model weights (esm1b_t33_650M_UR50S.pt, esm1b_t33_650M_UR50S-contact-regression.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: auto-detect)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for model inference (default: 16 for GPU, 1 for CPU)",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel execution (default: CPU count)",
    )

    parser.add_argument(
        "--keep-temp", action="store_true", 
        help="Keep temporary files after prediction"
    )

    parser.add_argument(
        "--debug", action="store_true", 
        help="Enable debug logging"
    )
    
    # New feature extraction control arguments
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=4,
        help="Batch size for feature extraction (number of proteins per batch)",
    )

    parser.add_argument(
        "--feature-timeout",
        type=int,
        default=600,
        help="Maximum time in seconds for processing a single protein during feature extraction",
    )

    parser.add_argument(
        "--cpu-only-features",
        action="store_true",
        help="Force CPU-only processing for feature extraction, even if GPU is available",
    )
    
    parser.add_argument(
        "--skip-feature-validation", action="store_true",
        help="Skip validation of contact map files before feature extraction"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        main(args)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        exit(1)