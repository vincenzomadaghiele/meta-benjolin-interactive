import time
from pathlib import Path
from unittest.mock import MagicMock

import importlib.util
import numpy as np


def _load_parameter_handler(repo_root: Path):
    module_path = repo_root / "python_server" / "parameter_handler.py"
    spec = importlib.util.spec_from_file_location("parameter_handler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load parameter_handler module spec"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.ParameterHandler


class TestParameterHandler:
    def test_getparameters_handler_uses_index0_coords_for_first_item_params(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "python_server" / "latent_param_dataset_16.npz"
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

        dataset = np.load(dataset_path)
        param_matrix = dataset["parameter_matrix"]
        reduced_latent = dataset["reduced_latent_matrix"]

        first_params = np.asarray(param_matrix[0]).astype(int)
        first_8_params = first_params[:8]
        print(f"Params selected in test: {first_8_params.tolist()}")
        print(f"Params in test: {first_params.tolist()}")

        x_expected, y_expected, z_expected = reduced_latent[0]
        print(f"Latent coords in test: x={x_expected}, y={y_expected}, z={z_expected}")

        client_mock = MagicMock()

        ParameterHandler = _load_parameter_handler(repo_root)
        handler = ParameterHandler(client_mock, latent=reduced_latent)
        handler.data_dir = str(dataset_path)
        handler.buffer_duration_seconds = 0.01

        handler.getparameters_handler("/getparams", *first_8_params.tolist())

        time.sleep(0.1)

        drawbox_calls = [c for c in client_mock.send_message.call_args_list if c.args and c.args[0] == "/drawBox"]
        assert drawbox_calls, "Expected at least one /drawBox message to be sent"

        last_call = drawbox_calls[-1]
        assert len(last_call.args) >= 2, "send_message should be called with address and payload"
        payload = last_call.args[1]

        assert isinstance(payload, (list, tuple)) and len(payload) >= 6, "Unexpected /drawBox payload structure"

        x_sent, y_sent, z_sent = payload[0], payload[1], payload[2]
        selected_index = payload[4]

        assert x_sent == x_expected and y_sent == y_expected and z_sent == z_expected, (
            f"/drawBox coords do not match dataset index 0. "
            f"Expected ({x_expected}, {y_expected}, {z_expected}), got ({x_sent}, {y_sent}, {z_sent})"
        )

        assert int(selected_index) == 0, f"Expected selected index 0, got {selected_index}"
