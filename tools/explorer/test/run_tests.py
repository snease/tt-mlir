# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import model_explorer

import requests
import time
import multiprocessing
import pytest
import glob

HOST = "localhost"
PORT = 8002
COMMAND_URL = "http://" + HOST + ":" + str(PORT) + "/apipost/v1/send_command"
TEST_LOAD_MODEL_PATHS = [
    "test/ttmlir/Dialect/TTNN/mnist_sharding.mlir",
    "tools/explorer/test/models/*.mlir",
]
TEST_EXECUTE_MODEL_PATHS = [
    "test/ttmlir/Silicon/TTNN/sharded/mnist_sharding_tiled.mlir",
]


def get_test_files(paths):
    files = []
    for path in paths:
        files.extend(glob.glob(path))
    return files


def execute_command(model_path, settings):
    cmd = {
        "extensionId": "tt_adapter",
        "cmdId": "execute",
        "modelPath": model_path,
        "deleteAfterConversion": False,
        "settings": settings,
    }

    result = requests.post(COMMAND_URL, json=cmd)
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False


@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    server_thread = multiprocessing.Process(
        target=model_explorer.visualize,
        kwargs={"extensions": ["tt_adapter"], "host": HOST, "port": PORT},
    )
    server_thread.start()
    time.sleep(1)

    request.addfinalizer(lambda: server_thread.terminate())


@pytest.mark.parametrize("model_path", get_test_files(TEST_LOAD_MODEL_PATHS))
def test_load_model(model_path):
    cmd = {
        "extensionId": "tt_adapter",
        "cmdId": "convert",
        "modelPath": model_path,
        "deleteAfterConversion": False,
        "settings": {},
    }

    result = requests.post(COMMAND_URL, json=cmd)
    assert result.ok
    if "error" in result.json():
        print(result.json())
        assert False


@pytest.mark.parametrize("model_path", get_test_files(TEST_EXECUTE_MODEL_PATHS))
def test_execute_model(model_path):
    execute_command(model_path, {"optimizationPolicy": "DF Sharding"})


def test_execute_mnist_l1_interleaved():
    execute_command(
        "test/ttmlir/Dialect/TTNN/mnist_sharding.mlir",
        {"optimizationPolicy": "L1 Interleaved"},
    )


def test_execute_mnist_optimizer_disabled():
    execute_command(
        "test/ttmlir/Dialect/TTNN/mnist_sharding.mlir",
        {"optimizationPolicy": "Optimizer Disabled"},
    )


def test_execute_model_invalid_policy():
    with pytest.raises(AssertionError):
        execute_command(
            TEST_EXECUTE_MODEL_PATHS[0], {"optimizationPolicy": "Invalid Policy"}
        )
