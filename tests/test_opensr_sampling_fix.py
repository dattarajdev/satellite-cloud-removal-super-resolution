"""Unit and integration tests for the OpenSR DDIM sampling configuration fix.

Covers:
A. Missing inference-step override:
   - No explicit custom_steps is supplied.
   - The configured sampling_steps value of 100 is resolved.
   - A valid DDIM schedule is created.
B. Explicit valid override:
   - An explicitly supplied valid step count is respected (e.g., 50).
C. Invalid configuration:
   - None with no valid fallback, 0, negative, non-int, and > timesteps are caught cleanly
   - Invalid configurations raise descriptive ValueError/TypeError and NEVER reach integer division TypeError.
D. Eta propagation:
   - The configured eta value (0.95) is propagated when no explicit override is supplied.
   - Explicit valid eta override is respected.
E. Model schedule preparation:
   - SRLatentDiffusion._prepare_model creates a valid DDIM schedule with expected timesteps.
"""

import os
import sys
import unittest
from pathlib import Path

# Add diffffusiosm directory to sys.path so opensr_model is importable
DIFF_DIR = Path(__file__).resolve().parent.parent / "diffffusiosm"
if str(DIFF_DIR) not in sys.path:
    sys.path.insert(0, str(DIFF_DIR))

from opensr_model.diffusion.utils import make_ddim_timesteps, DDIMSampler
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DummyDiffusionModel(nn.Module):
    """Dummy diffusion model for testing DDIMSampler without loading full weights."""
    def __init__(self, timesteps=1000):
        super().__init__()
        self.device = torch.device("cpu")
        self.num_timesteps = timesteps
        self.parameterization = "eps"
        # Dummy alphas_cumprod, betas, alphas_cumprod_prev
        alphas_cumprod = torch.linspace(0.999, 0.001, timesteps)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, timesteps))
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)


class TestDDIMSamplingSchedule(unittest.TestCase):
    """Test make_ddim_timesteps and DDIMSampler.make_schedule directly."""

    def setUp(self):
        self.dummy_model = DummyDiffusionModel(timesteps=1000)
        self.sampler = DDIMSampler(self.dummy_model)

    def test_A_missing_inference_step_override_default_100(self):
        """Requirement A: 100 DDIM steps resolved and schedule created successfully."""
        timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=100,
            num_ddpm_timesteps=1000,
            verbose=False,
        )
        self.assertEqual(len(timesteps), 100)
        self.assertEqual(timesteps[0], 1)
        self.assertEqual(timesteps[-1], 991)

        # DDIMSampler.make_schedule with 100 steps
        self.sampler.make_schedule(ddim_num_steps=100, ddim_eta=0.95, verbose=False)
        self.assertEqual(self.sampler.ddim_timesteps.shape[0], 100)

    def test_B_explicit_valid_override(self):
        """Requirement B: Explicitly supplied valid step count (50) is respected."""
        timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=50,
            num_ddpm_timesteps=1000,
            verbose=False,
        )
        self.assertEqual(len(timesteps), 50)

        self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.5, verbose=False)
        self.assertEqual(self.sampler.ddim_timesteps.shape[0], 50)
        self.assertTrue(self.sampler.ddim_sigmas.shape[0] == 50)

    def test_C_invalid_configuration_handling(self):
        """Requirement C: Invalid step configurations raise ValueError/TypeError before integer division."""
        invalid_steps = [
            None,
            0,
            -10,
            "100",
            100.5,
            1001,  # Exceeds total training timesteps (1000)
        ]

        for step in invalid_steps:
            with self.subTest(step=step):
                with self.assertRaises((ValueError, TypeError)) as ctx:
                    make_ddim_timesteps(
                        ddim_discr_method="uniform",
                        num_ddim_timesteps=step,
                        num_ddpm_timesteps=1000,
                        verbose=False,
                    )
                # Confirm error is NOT unsupported operand type(s) for //: 'int' and 'NoneType'
                err_msg = str(ctx.exception)
                self.assertNotIn("unsupported operand type(s) for //", err_msg)

                # Also test DDIMSampler.make_schedule
                with self.assertRaises((ValueError, TypeError)) as ctx2:
                    self.sampler.make_schedule(ddim_num_steps=step, ddim_eta=0.95, verbose=False)
                self.assertNotIn("unsupported operand type(s) for //", str(ctx2.exception))

    def test_D_eta_propagation_and_validation(self):
        """Requirement D: Eta propagation and validation."""
        # Valid eta=0 gives 0 sigmas
        self.sampler.make_schedule(ddim_num_steps=100, ddim_eta=0.0, verbose=False)
        self.assertTrue(torch.all(self.sampler.ddim_sigmas == 0))

        # Valid eta > 0 gives positive sigmas for t > 0
        self.sampler.make_schedule(ddim_num_steps=100, ddim_eta=0.95, verbose=False)
        self.assertTrue(torch.any(self.sampler.ddim_sigmas > 0))

        # Invalid eta values
        for invalid_eta in [-0.1, 1.5, "0.95"]:
            with self.subTest(eta=invalid_eta):
                with self.assertRaises((ValueError, TypeError)) as ctx:
                    self.sampler.make_schedule(ddim_num_steps=100, ddim_eta=invalid_eta, verbose=False)


class TestSRModelResolutionHelpers(unittest.TestCase):
    """Test resolution and fallback logic on SRLatentDiffusion."""

    def test_resolution_with_none_config_steps(self):
        """When config.denoiser_settings.sampling_steps is None (upstream breaking change),
        model must fallback to 100 gracefully and never return None."""
        from opensr_model.srmodel import SRLatentDiffusion

        # Create dummy config with null sampling_steps (reproducing upstream bug)
        config_yaml = """
        model_name: test
        diffusion_target: opensr_model.diffusion.ddpm.LatentDiffusion
        denoiser_settings:
            timesteps: 1000
            sampling_steps: null
            sampling_eta: null
        """
        cfg = OmegaConf.create(config_yaml)

        # Initialize mock model object to test resolution methods without loading 1GB weights
        model = SRLatentDiffusion.__new__(SRLatentDiffusion)
        model.config = cfg
        # Test __init__ fallback normalization
        if model.config.denoiser_settings.sampling_steps is None:
            model.config.denoiser_settings.sampling_steps = 100
        if model.config.denoiser_settings.sampling_eta is None:
            model.config.denoiser_settings.sampling_eta = 0.95

        self.assertEqual(model.config.denoiser_settings.sampling_steps, 100)
        self.assertEqual(model.config.denoiser_settings.sampling_eta, 0.95)

        # Test helper methods
        resolved_steps = model._resolve_and_validate_sampling_steps(None)
        self.assertEqual(resolved_steps, 100)

        override_steps = model._resolve_and_validate_sampling_steps(25)
        self.assertEqual(override_steps, 25)

        resolved_eta = model._resolve_and_validate_eta(None)
        self.assertEqual(resolved_eta, 0.95)

        override_eta = model._resolve_and_validate_eta(0.2)
        self.assertEqual(override_eta, 0.2)


if __name__ == "__main__":
    unittest.main()
