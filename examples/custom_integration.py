"""Example showing integration of the FastAPI app and making a local request.

This version avoids importing the FastAPI app at module-import time so
the example can still be opened or linted even when the project
dependencies are not installed. It will attempt to use FastAPI's test
client if available.
"""

def make_dummy_payload():
	# minimal JSON payload representing expected shapes
	return {
		"graph_data": {
			"nodes": [{"id": "n1"}, {"id": "n2"}],
			"edges": [{"source": "n1", "target": "n2"}],
			"node_features": [[0.1]*8, [0.2]*8]
		},
		"temporal_data": {"sequences": [[0.1]*10, [0.2]*10]},
		"behavioral_data": {"features": [{"f1": 0.1}, {"f1": 0.2}]},
		"context_data": {"assets": [{"id": "a1", "name": "server1"}], "vulnerabilities": []}
	}


def main():
	try:
		from fastapi.testclient import TestClient
		from cybershieldnet.api.fastapi import app
	except Exception as e:
		print("FastAPI test client or project app not available:", e)
		print("This example demonstrates the payload shape. Sample payload:\n")
		print(make_dummy_payload())
		return

	client = TestClient(app)
	payload = make_dummy_payload()
	resp = client.post('/api/v1/predict', json=payload)
	print('Status code:', resp.status_code)
	try:
		print('Response JSON:', resp.json())
	except Exception:
		print('Response text:', resp.text)


if __name__ == '__main__':
	main()

