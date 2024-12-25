#!/usr/bin/env python3
"""
GATSol Protein Solubility Predictor

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Updated import path
from torch_geometric.nn import GATConv, global_mean_pool

# Removed tqdm as it's not needed in container environment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TOOLS_PATH = Path("/app/Predict/tools")
BASE_DIR = Path("/app/Predict/NEED_to_PREPARE")


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
    batch_size: int = 1


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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.squeeze()

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
    for _, row in sequences_df.iterrows():
        fasta_path = fasta_dir / f"{row['id']}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">{row['id']}\n{row['sequence']}\n")
        logger.debug(f"Created FASTA file: {fasta_path}")


def copy_pdb_files(pdb_dir: Path, target_dir: Path, sequence_ids: List[str]) -> None:
    """
    Copy required PDB files to the working directory.

    Args:
        pdb_dir: Source directory containing PDB files
        target_dir: Target directory for PDB files
        sequence_ids: List of sequence IDs to copy
    """
    for seq_id in sequence_ids:
        pdb_path = pdb_dir / f"{seq_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Required PDB file not found: {pdb_path}")
        shutil.copy2(pdb_path, target_dir)
        logger.debug(f"Copied PDB file: {pdb_path}")


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
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing PDB file {pdb_path}: {e}")
        raise


def process_input_files(BASE_DIR: Path) -> None:
    """
    Process all PDB files to generate contact maps.

    Args:
        BASE_DIR: Base directory containing pdb and cm subdirectories
    """
    pdb_dir = BASE_DIR / "pdb"
    cm_dir = BASE_DIR / "cm"

    # Ensure output directory exists
    cm_dir.mkdir(exist_ok=True)

    # Process each PDB file
    pdb_files = list(pdb_dir.glob("*.pdb"))
    for idx, pdb_file in enumerate(pdb_files, 1):
        protein_id = pdb_file.stem  # filename without extension
        cm_file = cm_dir / f"{protein_id}.cm"

        logger.info(f"Processing PDB file {idx}/{len(pdb_files)}: {protein_id}")
        process_pdb_to_cm(pdb_file, cm_file)

    logger.info("Successfully processed all PDB files to contact maps")


def extract_features(tools_path: Path) -> None:
    """Extract features from processed files."""
    script_path = tools_path / "feature_extract" / "feature_extra.py"
    try:
        subprocess.run(["python", str(script_path)], check=True)
        logger.info("Successfully extracted features")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting features: {e}")
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
    test_dataset = []
    total_files = len(file_names)
    for idx, filename in enumerate(file_names, 1):
        logger.info(f"Loading dataset file {idx}/{total_files}: {filename}")
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
    model.eval()
    y_hat = torch.tensor([]).to(device)

    total_batches = len(loader)
    with torch.no_grad():
        for idx, data in enumerate(loader, 1):
            if idx % 10 == 0:  # Log every 10th batch
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

    # Read sequence data
    sequences_df = pd.read_csv(args.sequences)
    shutil.copy2(args.sequences, BASE_DIR / "list.csv")
    if not {"id", "sequence"}.issubset(sequences_df.columns):
        raise ValueError("Sequences file must contain 'id' and 'sequence' columns")

    # Generate FASTA files from sequences
    write_fasta_files(sequences_df, BASE_DIR / "fasta")
    logger.info("Generated FASTA files from sequences")

    # Copy PDB files
    copy_pdb_files(Path(args.pdb_dir), BASE_DIR / "pdb", sequences_df["id"].tolist())
    logger.info("Copied PDB files to working directory")

    # Process files
    process_input_files(BASE_DIR)
    extract_features(TOOLS_PATH)

    # Setup model
    config = ModelConfig()
    model = GATClassifier(
        config.in_channels, config.hidden_channels, config.num_heads, config.num_layers
    )

    # Load model weights with proper device handling
    model_path = Path(args.gatsol_weights_dir) / "best_model.pt"
    model = load_model(model, model_path, device)

    # Prepare dataset and make predictions
    test_dataset = load_test_dataset(
        BASE_DIR / "pkl", sequences_df["id"].tolist(), device
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    predictions = make_predictions(model, device, test_loader)

    # Save results
    sequences_df["Solubility_hat"] = predictions.cpu().numpy()

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info(f"Predictions saved to: {output_dir/'predictions.csv'}")

    # Cleanup temporary directories
    if not args.keep_temp:
        for temp_dir in ["cm", "pkl"]:
            temp_path = BASE_DIR / temp_dir
            if temp_path.exists():
                shutil.rmtree(temp_path)
                logger.debug(f"Cleaned up temporary directory: {temp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GATSol: Protein solubility prediction using Graph Attention Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sequences",
        type=str,
        required=True,
        help="Path to CSV file containing sequence IDs and sequences (columns: id, sequence)",
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
        "--keep-temp", action="store_true", help="Keep temporary files after prediction"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        main(args)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        exit(1)
