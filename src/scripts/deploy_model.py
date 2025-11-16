"""Deployment helper script for CyberShieldNet.

Provides a minimal interface to prepare model artifacts for deployment.
Currently supports exporting a registered model to ONNX via
`ModelManager.export_model` and copying artifacts to a target
deployment directory.
"""

import argparse
import logging
from pathlib import Path
import shutil

from cybershieldnet.models.model_manager import ModelManager
from cybershieldnet.utils.helpers import ConfigManager

logger = logging.getLogger(__name__)


def export_model_to_onnx(model_name: str, model_path: str, out_dir: str = 'deploy') -> str:
    cfg_mgr = ConfigManager()
    cfg = cfg_mgr.get_config('deployment_config') if Path('config/deployment_config.yaml').exists() else {}

    mm = ModelManager(cfg)
    # Load model into manager
    mm.register_model(model_name, None, {'version': '1.0.0'})
    try:
        # If a real path exists and matches manager expectations, load then export
        if Path(model_path).exists():
            mm.load_model(model_name, model_path)
    except Exception:
        # If load fails, continue: export may still work if manager reconstructs model
        logger.warning("ModelManager.load_model failed; attempting export with registered model placeholder")

    out = mm.export_model(model_name, export_format='onnx')

    # Copy exported artifact to deployment directory
    out_dirp = Path(out_dir)
    out_dirp.mkdir(parents=True, exist_ok=True)
    dest = out_dirp / Path(out).name
    shutil.copy(out, dest)

    logger.info(f"Model exported to {dest}")
    return str(dest)


def main():
    parser = argparse.ArgumentParser(description='Prepare model artifacts for deployment')
    parser.add_argument('--model-name', required=True, help='Logical model name')
    parser.add_argument('--model-path', required=True, help='Path to saved model file')
    parser.add_argument('--out-dir', default='deploy', help='Output deployment directory')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        out = export_model_to_onnx(args.model_name, args.model_path, args.out_dir)
        print(f"Deployment artifact ready at: {out}")
    except Exception as e:
        logger.error(f"Deployment preparation failed: {e}")
        raise


if __name__ == '__main__':
    main()
