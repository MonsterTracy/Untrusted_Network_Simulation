"""Keep training runtime controls local to each ToM test module."""

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def restore_torch_runtime_after_module():
    """Include module-fixture setup/teardown without changing training controls."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
