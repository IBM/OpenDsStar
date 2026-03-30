"""
DataBench Dataset Loader

Loads DataBench datasets and provides utility functions for data analysis.
Includes automatic download from HuggingFace if datasets are not found locally.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def download_databench_from_huggingface(base_path: str = "databench_subset") -> bool:
    """
    Download DataBench datasets from HuggingFace if not already present.

    Args:
        base_path (str): Base directory to save datasets.

    Returns:
        bool: True if download successful, False otherwise.
    """
    try:
        from datasets import load_dataset

        print("Downloading DataBench from HuggingFace...")
        dataset = load_dataset("cardiffnlp/databench", "qa")
        train_data = dataset["train"]

        print(f"Loaded {len(train_data)} examples from HuggingFace")

        # Group questions by dataset
        dataset_questions = defaultdict(list)
        for example in train_data:
            dataset_name = example["dataset"]
            dataset_questions[dataset_name].append(example)

        # Dataset ID mapping
        dataset_mapping = {
            "forbes": "001_Forbes",
            "london": "006_London",
            "fifa": "007_Fifa",
            "roller": "013_Roller",
            "airbnb": "014_Airbnb",
            "real": "020_Real",
            "professionals": "030_Professionals",
            "speed": "040_Speed",
            "us": "058_US",
            "openfoodfacts": "070_OpenFoodFacts",
        }

        os.makedirs(base_path, exist_ok=True)

        # Process each dataset
        for hf_name, dataset_id in dataset_mapping.items():
            # Find matching dataset
            matching_key = None
            for key in dataset_questions.keys():
                if hf_name.lower() in key.lower():
                    matching_key = key
                    break

            if not matching_key:
                print(f"Warning: No data found for {dataset_id}")
                continue

            questions = dataset_questions[matching_key]

            # Create dataset directory
            dataset_dir = os.path.join(base_path, dataset_id)
            os.makedirs(dataset_dir, exist_ok=True)

            # Create questions CSV
            questions_file = os.path.join(dataset_dir, f"{dataset_id}_questions.csv")
            questions_df = pd.DataFrame(
                [
                    {
                        "question": q["question"],
                        "answer": q["answer"],
                        "type": q.get("type", ""),
                        "columns_used": q.get("columns_used", ""),
                        "column_types": q.get("column_types", ""),
                        "sample_answer": q.get("sample_answer", ""),
                    }
                    for q in questions
                ]
            )
            questions_df.to_csv(questions_file, index=False)
            print(f"Created {questions_file} ({len(questions_df)} questions)")

            # Create placeholder main CSV
            csv_file = os.path.join(dataset_dir, f"{dataset_id}.csv")
            if questions:
                first_q = questions[0]
                columns_used = first_q.get("columns_used", "")
                if columns_used:
                    cols = [c.strip() for c in columns_used.split(",")]
                    dummy_df = pd.DataFrame(columns=cols)
                    dummy_df.loc[0] = ["sample_data"] * len(cols)
                else:
                    dummy_df = pd.DataFrame({"data": ["sample"]})
                dummy_df.to_csv(csv_file, index=False)
                print(f"Created {csv_file} (placeholder)")

        print("Download complete!")
        return True

    except ImportError:
        print(
            "Error: 'datasets' library not installed. Install with: pip install datasets"
        )
        return False
    except Exception as e:
        print(f"Error downloading DataBench: {e}")
        return False


def load_databench_dataset(
    dataset_id: str, base_path: str = "databench_subset", auto_download: bool = True
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str], Optional[str]
]:
    """
    Loads a specific DataBench dataset (CSV and its questions).
    Automatically downloads from HuggingFace if not found locally.

    Args:
        dataset_id (str): The ID of the dataset (e.g., "001_Forbes").
        base_path (str): Base directory containing datasets.
        auto_download (bool): If True, automatically download from HuggingFace if not found.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The main dataset CSV.
            - pd.DataFrame: The questions CSV.
            - str: Path to the main dataset CSV.
            - str: Path to the questions CSV.
    """
    dataset_dir = os.path.join(base_path, dataset_id)

    csv_file = os.path.join(dataset_dir, f"{dataset_id}.csv")
    questions_file = os.path.join(dataset_dir, f"{dataset_id}_questions.csv")

    # Check if files exist, if not try to download
    if not os.path.exists(csv_file) or not os.path.exists(questions_file):
        if auto_download:
            print(
                "Dataset not found locally. Attempting to download from HuggingFace..."
            )
            if download_databench_from_huggingface(base_path):
                # Check again after download
                if not os.path.exists(csv_file) or not os.path.exists(questions_file):
                    print("Error: Dataset files still not found after download")
                    return None, None, None, None
            else:
                print("Error: Failed to download dataset")
                return None, None, None, None
        else:
            print(f"Error: Dataset CSV not found at {csv_file}")
            print(f"Error: Questions CSV not found at {questions_file}")
            return None, None, None, None

    df = pd.read_csv(csv_file)
    questions_df = pd.read_csv(questions_file)

    return df, questions_df, csv_file, questions_file


def get_available_datasets(base_path: str = "databench_subset") -> List[str]:
    """
    Get list of available dataset IDs.

    Args:
        base_path (str): Base directory containing datasets.

    Returns:
        List[str]: List of available dataset IDs.
    """
    available_datasets = [
        "001_Forbes",
        "006_London",
        "007_Fifa",
        "013_Roller",
        "014_Airbnb",
        "020_Real",
        "030_Professionals",
        "040_Speed",
        "058_US",
        "070_OpenFoodFacts",
    ]
    return available_datasets


def get_dataset_info(
    dataset_id: str, base_path: str = "databench_subset"
) -> Dict[str, Any]:
    """
    Get information about a dataset without loading the full data.

    Args:
        dataset_id (str): The ID of the dataset.
        base_path (str): Base directory containing datasets.

    Returns:
        Dict[str, Any]: Dictionary with dataset information.
    """
    df, questions_df, csv_path, questions_path = load_databench_dataset(
        dataset_id, base_path
    )

    if df is None or questions_df is None:
        return {"error": f"Failed to load dataset {dataset_id}"}

    return {
        "dataset_id": dataset_id,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "num_questions": len(questions_df),
        "csv_path": csv_path,
        "questions_path": questions_path,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }


def get_dataset_sample(
    dataset_id: str, n_rows: int = 5, base_path: str = "databench_subset"
) -> Dict[str, Any]:
    """
    Get a sample of rows from a dataset.

    Args:
        dataset_id (str): The ID of the dataset.
        n_rows (int): Number of rows to return.
        base_path (str): Base directory containing datasets.

    Returns:
        Dict[str, Any]: Dictionary with sample data.
    """
    df, questions_df, _, _ = load_databench_dataset(dataset_id, base_path)

    if df is None:
        return {"error": f"Failed to load dataset {dataset_id}"}

    return {
        "dataset_id": dataset_id,
        "sample": df.head(n_rows).to_dict(orient="records"),
        "columns": df.columns.tolist(),
    }


def get_dataset_questions(
    dataset_id: str, base_path: str = "databench_subset"
) -> Dict[str, Any]:
    """
    Get questions for a dataset.

    Args:
        dataset_id (str): The ID of the dataset.
        base_path (str): Base directory containing datasets.

    Returns:
        Dict[str, Any]: Dictionary with questions.
    """
    _, questions_df, _, _ = load_databench_dataset(dataset_id, base_path)

    if questions_df is None:
        return {"error": f"Failed to load questions for dataset {dataset_id}"}

    return {
        "dataset_id": dataset_id,
        "questions": questions_df.to_dict(orient="records"),
        "num_questions": len(questions_df),
    }


def get_dummies_for_dataset(
    dataset_id: str,
    columns: Optional[List[str]] = None,
    base_path: str = "databench_subset",
) -> Dict[str, Any]:
    """
    Get dummy variables (one-hot encoding) for categorical columns.

    Args:
        dataset_id (str): The ID of the dataset.
        columns (List[str], optional): Specific columns to encode. If None, encodes all object columns.
        base_path (str): Base directory containing datasets.

    Returns:
        Dict[str, Any]: Dictionary with dummy-encoded data info.
    """
    df, _, _, _ = load_databench_dataset(dataset_id, base_path)

    if df is None:
        return {"error": f"Failed to load dataset {dataset_id}"}

    # If no columns specified, use all object/categorical columns
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return {
            "dataset_id": dataset_id,
            "message": "No categorical columns found or specified",
            "original_shape": df.shape,
        }

    # Create dummy variables
    df_dummies = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep="_")

    return {
        "dataset_id": dataset_id,
        "original_shape": df.shape,
        "encoded_shape": df_dummies.shape,
        "original_columns": df.columns.tolist(),
        "encoded_columns": df_dummies.columns.tolist(),
        "new_columns": [col for col in df_dummies.columns if col not in df.columns],
        "encoded_sample": df_dummies.head(3).to_dict(orient="records"),
    }


def stream_dataset_statistics(
    dataset_id: str, base_path: str = "databench_subset"
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a dataset.

    Args:
        dataset_id (str): The ID of the dataset.
        base_path (str): Base directory containing datasets.

    Returns:
        Dict[str, Any]: Dictionary with dataset statistics.
    """
    df, _, _, _ = load_databench_dataset(dataset_id, base_path)

    if df is None:
        return {"error": f"Failed to load dataset {dataset_id}"}

    # Basic statistics
    stats = {
        "dataset_id": dataset_id,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        stats["numeric_statistics"] = df[numeric_cols].describe().to_dict()

    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        stats["categorical_statistics"] = {
            col: {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }
            for col in categorical_cols
        }

    return stats
