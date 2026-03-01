"""
Utility functions for FMEA Generator
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml
import json
from datetime import datetime


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str = 'config/config.yaml'):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_output_directory(base_path: str = 'output') -> Path:
    """
    Create output directory with timestamp
    
    Args:
        base_path: Base output directory
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def validate_file_path(file_path: str) -> bool:
    """
    Validate that file exists and is readable
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(file_path)
    return path.exists() and path.is_file()


def format_rpn_color(rpn: int) -> str:
    """
    Get color code based on RPN value
    
    Args:
        rpn: Risk Priority Number
        
    Returns:
        Hex color code
    """
    if rpn >= 500:
        return '#d62728'  # Red - Critical
    elif rpn >= 250:
        return '#ff7f0e'  # Orange - High
    elif rpn >= 100:
        return '#ffbb78'  # Light Orange - Medium
    else:
        return '#2ca02c'  # Green - Low


def generate_summary_report(fmea_df) -> str:
    """
    Generate text summary of FMEA results
    
    Args:
        fmea_df: FMEA DataFrame
        
    Returns:
        Summary text
    """
    total_failures = len(fmea_df)
    critical_count = len(fmea_df[fmea_df['Action Priority'] == 'Critical'])
    high_count = len(fmea_df[fmea_df['Action Priority'] == 'High'])
    avg_rpn = fmea_df['Rpn'].mean()
    max_rpn = fmea_df['Rpn'].max()
    
    summary = f"""
FMEA SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}

OVERALL STATISTICS:
- Total Failure Modes Identified: {total_failures}
- Critical Priority Items: {critical_count}
- High Priority Items: {high_count}
- Average RPN: {avg_rpn:.2f}
- Maximum RPN: {max_rpn:.0f}

TOP 5 RISKS:
"""
    
    top_5 = fmea_df.nlargest(5, 'Rpn')
    for idx, row in top_5.iterrows():
        summary += f"\n{idx + 1}. {row['Failure Mode']}"
        summary += f"\n   RPN: {row['Rpn']} | Priority: {row['Action Priority']}"
        summary += f"\n   Effect: {row['Effect']}"
        summary += f"\n   Recommended Action: {row['Recommended Action']}\n"
    
    return summary


def export_to_json(fmea_df, output_path: str):
    """
    Export FMEA to JSON format
    
    Args:
        fmea_df: FMEA DataFrame
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict format
    fmea_dict = fmea_df.to_dict(orient='records')
    
    with open(output_path, 'w') as f:
        json.dump(fmea_dict, f, indent=2)
    
    logging.info(f"FMEA exported to JSON: {output_path}")


def calculate_risk_reduction(original_rpn: int, current_rpn: int) -> Dict[str, Any]:
    """
    Calculate risk reduction metrics
    
    Args:
        original_rpn: Original RPN value
        current_rpn: Current RPN value after improvements
        
    Returns:
        Dictionary with reduction metrics
    """
    reduction = original_rpn - current_rpn
    reduction_percent = (reduction / original_rpn * 100) if original_rpn > 0 else 0
    
    return {
        'original_rpn': original_rpn,
        'current_rpn': current_rpn,
        'reduction': reduction,
        'reduction_percent': reduction_percent,
        'status': 'Improved' if reduction > 0 else 'No Change' if reduction == 0 else 'Worsened'
    }


def merge_fmea_files(file_paths: list) -> 'pd.DataFrame':
    """
    Merge multiple FMEA files into one
    
    Args:
        file_paths: List of FMEA file paths
        
    Returns:
        Merged DataFrame
    """
    import pandas as pd
    
    dfs = []
    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            continue
        
        df['source_file'] = path.name
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid FMEA files to merge")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df


def fuzzy_deduplicate_modes(extracted_data, threshold: float = 0.85):
    """
    Apply semantic deduplication of failure modes using fuzzy matching.
    
    Merges similar failure modes by clustering based on string similarity.
    When merging:
    - occurrence (count) → summed across cluster members
    - severity → maximum value in cluster
    - detection → maximum value in cluster
    - failure_mode → most descriptive (longest string)
    
    Args:
        extracted_data: List of dicts or DataFrame with extracted failure information.
                       Expected columns: 'failure_mode', 'effect', 'cause', 'severity', 
                       'occurrence', 'detection', etc.
        threshold: Similarity threshold (0-1) above which failure modes are considered 
                  duplicates. Default: 0.85
    
    Returns:
        DataFrame with deduplicated failure modes, ready for RPN calculation.
    """
    import pandas as pd
    from difflib import SequenceMatcher
    
    logger = logging.getLogger(__name__)
    logger.info(f"Applying fuzzy deduplication with threshold={threshold}")
    
    # Convert list of dicts to DataFrame if needed
    if isinstance(extracted_data, list):
        if not extracted_data:
            logger.warning("Empty extracted data provided to fuzzy deduplication")
            return pd.DataFrame()
        df = pd.DataFrame(extracted_data)
    else:
        df = extracted_data.copy()
    
    if df.empty or 'failure_mode' not in df.columns:
        logger.warning("No failure_mode column found; skipping deduplication")
        return df
    
    # Helper function: compute similarity between two strings
    def similarity_score(a: str, b: str) -> float:
        """Compute similarity between two failure mode strings."""
        if not a or not b:
            return 0.0
        
        a_clean = str(a).lower().strip()
        b_clean = str(b).lower().strip()
        
        # Use SequenceMatcher ratio as primary similarity metric
        seq_ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
        return seq_ratio
    
    # Cluster similar failure modes
    df_reset = df.reset_index(drop=True)
    failure_modes = df_reset['failure_mode'].astype(str).tolist()
    
    clusters: List[List[int]] = []
    processed = set()
    
    for i, mode_i in enumerate(failure_modes):
        if i in processed:
            continue
        
        cluster = [i]
        processed.add(i)
        
        # Find all modes similar to mode_i
        for j, mode_j in enumerate(failure_modes):
            if j <= i or j in processed:
                continue
            
            sim = similarity_score(mode_i, mode_j)
            if sim >= threshold:
                cluster.append(j)
                processed.add(j)
        
        clusters.append(cluster)
    
    # Merge clusters
    merged_rows = []
    
    for cluster_indices in clusters:
        cluster_df = df_reset.iloc[cluster_indices]
        
        # Choose most descriptive failure_mode (longest string)
        modes = cluster_df['failure_mode'].astype(str)
        most_descriptive = modes.loc[modes.str.len().idxmax()]
        
        merged_row = {
            'failure_mode': most_descriptive,
        }
        
        # Merge other fields
        for col in cluster_df.columns:
            if col == 'failure_mode':
                continue
            
            if col == 'occurrence':
                # Sum occurrences if numeric
                occ_vals = pd.to_numeric(cluster_df[col], errors='coerce').dropna()
                if not occ_vals.empty:
                    merged_row[col] = int(occ_vals.sum())
                else:
                    merged_row[col] = cluster_df[col].iloc[0]
            
            elif col in ['severity', 'detection']:
                # Take maximum if numeric
                val_series = pd.to_numeric(cluster_df[col], errors='coerce')
                if not val_series.dropna().empty:
                    merged_row[col] = int(val_series.max())
                else:
                    merged_row[col] = cluster_df[col].iloc[0]
            
            else:
                # For other fields, take first non-null value
                non_null = cluster_df[col].dropna()
                if not non_null.empty:
                    merged_row[col] = non_null.iloc[0]
                else:
                    merged_row[col] = None
        
        merged_rows.append(merged_row)
    
    result_df = pd.DataFrame(merged_rows)
    
    removed_count = len(df) - len(result_df)
    if removed_count > 0:
        logger.info(f"Fuzzy deduplication removed {removed_count} duplicate entries. "
                   f"Merged into {len(result_df)} unique failure modes.")
    
    return result_df


class ProgressTracker:
    """Track progress of FMEA generation"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step_name: str):
        """Update progress"""
        self.current_step += 1
        progress_percent = (self.current_step / self.total_steps) * 100
        
        elapsed_time = datetime.now() - self.start_time
        
        logging.info(f"Progress: {progress_percent:.1f}% - {step_name} (Elapsed: {elapsed_time})")
    
    def complete(self):
        """Mark completion"""
        total_time = datetime.now() - self.start_time
        logging.info(f"Process completed in {total_time}")


if __name__ == "__main__":
    # Test utilities
    setup_logging('INFO')
    
    config = load_config('../config/config.yaml')
    print("Configuration loaded successfully")
    
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    
    # Test progress tracker
    tracker = ProgressTracker(5)
    tracker.update("Step 1: Load data")
    tracker.update("Step 2: Preprocess")
    tracker.update("Step 3: Extract")
    tracker.update("Step 4: Score")
    tracker.update("Step 5: Generate output")
    tracker.complete()
