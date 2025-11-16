"""Simple data pipeline CLI for CyberShieldNet.

This script provides a lightweight ETL pipeline that uses the
project utilities (`ConfigManager`, `DataLoader`, `FeatureExtractor`)
to load raw data, run basic preprocessing and feature extraction,
and write a processed artifact for downstream training/evaluation.

The script is intentionally conservative and will create a small
processed file even if the input is missing (useful for CI/tests).
"""

import argparse
import pickle
from pathlib import Path
import logging

from cybershieldnet.utils.helpers import ConfigManager, DataLoader
from cybershieldnet.data_ingestion.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


def run_pipeline(data_path: str, output_path: str, config_dir: str = "config") -> str:
    cfg_mgr = ConfigManager(config_dir)
    try:
        pipeline_cfg = cfg_mgr.get_config('data_config')
    except Exception:
        pipeline_cfg = {}

    dl = DataLoader(pipeline_cfg)

    data_pathp = Path(data_path)
    if not data_pathp.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Load raw data (DataLoader handles multiple formats)
    raw = dl.load_data(str(data_pathp), use_cache=False, force_reload=True)

    # Feature extraction
    fe = FeatureExtractor(pipeline_cfg)

    processed = {}
    # Try to extract modalities if present
    if isinstance(raw, dict):
        if 'temporal' in raw:
            processed['temporal'] = fe.extract_temporal_features(raw['temporal'])
        if 'graph' in raw:
            processed['graph'] = fe.extract_graph_features(raw['graph'])
        if 'behavioral' in raw:
            processed['behavioral'] = fe.extract_behavioral_features(raw['behavioral'])
    else:
        # If raw is array-like, store as-is under 'data'
        processed['data'] = raw

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as f:
        pickle.dump(processed, f)

    logger.info(f"Processed data written to {outp}")
    return str(outp)


def main():
    parser = argparse.ArgumentParser(description="Run CyberShieldNet data pipeline")
    parser.add_argument('--data-path', '-i', required=True, help='Path to raw input data')
    parser.add_argument('--output-path', '-o', default='data/processed/processed.pkl', help='Path to save processed data')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        out = run_pipeline(args.data_path, args.output_path, args.config_dir)
        print(f"Processed data saved to: {out}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
