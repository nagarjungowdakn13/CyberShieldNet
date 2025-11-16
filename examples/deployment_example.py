"""Deployment example demonstrating ONNX export and packaging.

Attempts to export a model to ONNX using the project's `ModelManager`.
If PyTorch or the ModelManager is unavailable, the script prints the
expected steps and a small example manifest structure.
"""

from pathlib import Path
import json


def prepare_deployment_manifest(model_name: str, onnx_path: str, out_dir: str = 'deploy') -> str:
	outp = Path(out_dir)
	outp.mkdir(parents=True, exist_ok=True)

	manifest = {
		'model_name': model_name,
		'onnx_path': str(onnx_path),
		'runtime': 'onnxruntime',
		'description': 'Exported CyberShieldNet model for deployment'
	}

	manifest_path = outp / f"{model_name}_manifest.json"
	with open(manifest_path, 'w') as f:
		json.dump(manifest, f, indent=2)

	return str(manifest_path)


def run_export(model_name: str = 'cybershieldnet', model_path: str = 'models/cybershieldnet.pth') -> str:
	try:
		# import inside function to avoid top-level failures
		from cybershieldnet.models.model_manager import ModelManager
		mm = ModelManager({'model_dir': 'models'})
		try:
			mm.load_model(model_name, model_path)
		except Exception:
			mm.register_model(model_name, None, {'version': '1.0.0'})

		onnx_path = mm.export_model(model_name, export_format='onnx')
		manifest = prepare_deployment_manifest(model_name, onnx_path)
		return manifest
	except Exception as e:
		print('ModelManager or export dependencies unavailable:', e)
		print('Expected steps:')
		print(' - Load or register model into ModelManager')
		print(' - Call export_model(..., export_format="onnx")')
		fallback_manifest = prepare_deployment_manifest(model_name, f"{model_name}.onnx")
		return fallback_manifest


def main():
	manifest = run_export()
	print(f"Deployment manifest created: {manifest}")


if __name__ == '__main__':
	main()

