#!/usr/bin/env python3
"""
GATSol Protein Solubility Predictor

This script serves as the main entry point for the GATSol protein solubility prediction pipeline.
It handles the entire workflow from processing input files to generating predictions using
a Graph Attention Network (GAT) model.
"""

import argparse
import logging
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration parameters for the GAT model."""

    in_channels: int = 1300
    hidden_channels: int = 1024
    num_classes: int = 1
    num_heads: int = 16
    num_layers: int = 2
    batch_size: int = 1


class GATClassifier(nn.Module):
    """
    Graph Attention Network (GAT) for protein solubility prediction.

    Args:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        num_heads: Number of attention heads
        num_layers: Number of GAT layers
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, num_heads: int, num_layers: int
    ) -> None:
        super().__init__()

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


def setup_directories() -> Path:
    """Create and return the base working directory structure."""
    base_dir = Path("/app/Predict/NEED_to_PREPARE")

    # Create all required directories
    for subdir in ["fasta", "pdb", "cm", "pkl"]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {base_dir / subdir}")

    return base_dir


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
                "/app/Predict/tools/pdb_to_cm/pdb_to_cm.py",
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


def process_input_files(base_dir: Path) -> None:
    """
    Process all PDB files to generate contact maps.

    Args:
        base_dir: Base directory containing pdb and cm subdirectories
    """
    pdb_dir = base_dir / "pdb"
    cm_dir = base_dir / "cm"

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
    for filename in tqdm(file_names, desc="Loading test dataset"):
        file_path = pkl_path / f"{filename}.pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f).to(device)
        test_dataset.append(data)
    return test_dataset


def make_predictions(
    model: nn.Module, device: torch.device, loader: DataLoader
) -> torch.Tensor:
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

    with torch.no_grad():
        for data in tqdm(loader, desc="Generating predictions"):
            data = data.to(device)
            output = model(data)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            y_hat = torch.cat((y_hat, output), 0)

    return y_hat


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    logger.info("Starting GATSol prediction pipeline")

    # Read sequence data
    sequences_df = pd.read_csv(args.sequences)
    if not {"id", "sequence"}.issubset(sequences_df.columns):
        raise ValueError("Sequences file must contain 'id' and 'sequence' columns")

    # Setup directories
    base_dir = setup_directories()

    # Generate FASTA files from sequences
    write_fasta_files(sequences_df, base_dir / "fasta")
    logger.info("Generated FASTA files from sequences")

    # Copy PDB files
    copy_pdb_files(Path(args.pdb_dir), base_dir / "pdb", sequences_df["id"].tolist())
    logger.info("Copied PDB files to working directory")

    # Save sequences dataframe for later use
    sequences_df.to_csv(base_dir / "list.csv", index=False)

    # Process files
    process_input_files(base_dir)
    tools_path = Path("/app/Predict/tools")
    extract_features(tools_path)

    # Setup model
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATClassifier(
        config.in_channels, config.hidden_channels, config.num_heads, config.num_layers
    ).to(device)

    # Load model weights
    model_path = Path(args.model_dir) / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    model.load_state_dict(torch.load(model_path))

    # Prepare dataset and make predictions
    test_dataset = load_test_dataset(
        base_dir / "pkl", sequences_df["id"].tolist(), device
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
            temp_path = base_dir / temp_dir
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
        "--model-dir",
        type=str,
        default="/app/check_point/best_model",
        required=False,
        help="Directory containing model weights (best_model.pt)",
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
