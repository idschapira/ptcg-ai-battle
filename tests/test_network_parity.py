"""Parity tests: torch PolicyValueNet vs the pure-numpy runtime forward.

The submission runs inference in numpy only (src/rl_models/network_numpy.py);
training happens in torch (src/rl_models/network.py). These tests pin the
two implementations together: any architecture change that lands in one
side but not the other fails here.

Skips (never fails) when torch is not installed — runtime environments
and CI without requirements-dev.txt still run the rest of the suite.

Run from the repo root:  python -m unittest tests.test_network_parity
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rl_models.encoding import ENCODING_DIM, OPTION_DIM
from src.rl_models.network_numpy import NumpyPolicyValueNet, export_state_dict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

N_SAMPLES = 32
MAX_K = 24
SEED = 7
ATOL = 1e-5


@unittest.skipUnless(HAS_TORCH, "torch not installed (dev-only dependency)")
class TestTorchNumpyParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from src.rl_models.network import PolicyValueNet
        torch.manual_seed(SEED)
        cls.torch_net = PolicyValueNet()
        cls.torch_net.eval()
        cls._tmp = tempfile.TemporaryDirectory()
        weights_path = Path(cls._tmp.name) / "weights.npz"
        export_state_dict(cls.torch_net.state_dict(), weights_path)
        cls.numpy_net = NumpyPolicyValueNet.load(weights_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_export_loads_back(self) -> None:
        self.assertIsNotNone(self.numpy_net)

    def test_logits_and_value_match(self) -> None:
        rng = np.random.default_rng(SEED)
        for _ in range(N_SAMPLES):
            k = int(rng.integers(1, MAX_K + 1))
            state = rng.normal(0, 1, ENCODING_DIM).astype(np.float32)
            options = rng.normal(0, 1, (k, OPTION_DIM)).astype(np.float32)

            np_logits, np_value = self.numpy_net.forward(state, options)

            with torch.no_grad():
                t_logits, t_value = self.torch_net(
                    torch.from_numpy(state).unsqueeze(0),
                    torch.from_numpy(options).unsqueeze(0),
                    torch.ones((1, k), dtype=torch.bool),
                )
            np.testing.assert_allclose(np_logits, t_logits[0].numpy(),
                                       atol=ATOL, rtol=0)
            self.assertAlmostEqual(np_value, float(t_value[0]), delta=ATOL)

    def test_missing_weights_load_returns_none(self) -> None:
        self.assertIsNone(NumpyPolicyValueNet.load(
            Path(self._tmp.name) / "does_not_exist.npz"))


if __name__ == "__main__":
    unittest.main()
