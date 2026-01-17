
import sys
import os
from pathlib import Path


# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "LTX-2", "packages", "ltx-core", "src"))

# MOCK einops to avoid dependency issues since we only need imports to pass
# and we mock the actual patchifier logic anyway
from unittest.mock import MagicMock
mock_einops = MagicMock()
sys.modules["einops"] = mock_einops

import torch
from dataclasses import dataclass
import sys

# Import ltx_core components

# We need to simulate the latent space size and the VideoConditionByLatentIndex logic
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.types import VideoLatentShape

def test_latent_index_mismatch():
    print("Testing VideoConditionByLatentIndex with pixel vs latent indices...")
    
    # LTX-2 Configuration
    # Latent patch size is usually 1 (time) x 1 x 1 for VideoLatentPatchifier if we look at the code?
    # Wait, in patchifiers.py: self._patch_size = (1, patch_size, patch_size)
    # But temporal compression comes from the VAE, so the LatentState has fewer frames than pixels.
    # The patchifier works on the ALREADY COMPRESSED latent state.
    
    # Assume 5 seconds at 24fps -> ~120 frames
    # LTX-2 VAE has 8x temporal downsampling
    # So latent frames = 120 / 8 = 15 frames
    
    pixel_frames = 121 # 8k + 1 rule: 15 * 8 + 1 = 121
    # VideoLatentShape.from_pixel_shape: (frames - 1) // 8 + 1
    # (121 - 1) // 8 + 1 = 15 + 1 = 16
    latent_frames = 16
    height = 64
    width = 64
    # batch, channels, frames, height, width
    latent_shape_torch = (1, 128, latent_frames, height, width) 
    
    # Mock latent tools
    patchifier = VideoLatentPatchifier(patch_size=1) 
    # Since we mocked einops, VideoLatentPatchifier.patchify won't work properly
    # We need to mock patchify to return a proper tensor, because the result is assigned
    # to latent_state.latent[:] which expects a tensor, not a Mock
    def mock_patchify(latents):
        # latents: (b, c, f, h, w)
        # output: (b, tokens, c*p*p*p) 
        # flattening spatial/temporal dims
        b, c, f, h, w = latents.shape
        tokens = f * h * w
        # c is embedding dim
        return torch.zeros((b, tokens, c))

    patchifier.patchify = MagicMock(side_effect=mock_patchify)

    # Mocking target shape
    target_shape = MagicMock()
    target_shape.to_torch_shape.return_value = latent_shape_torch
    
    # Mock LatentTools
    latent_tools = MagicMock()
    latent_tools.target_shape = target_shape
    latent_tools.patchifier = patchifier
    
    # Helper to create a fake latent state
    @dataclass
    class MockLatentState:
        latent: torch.Tensor
        clean_latent: torch.Tensor
        denoise_mask: torch.Tensor
        
        def clone(self):
            return MockLatentState(
                self.latent.clone(),
                self.clean_latent.clone(),
                self.denoise_mask.clone()
            )

    # Initial state (empty/noise)
    # The latent state is "patchified", meaning it's flattened.
    # patchify: "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)"
    # video_state.latent shape check
    # Let's just trust patchifier.get_token_count logic
    
    # VideoLatentPatchifier.get_token_count:
    # math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)
    # frames * height * width
    
    num_tokens = latent_frames * height * width
    embedding_dim = 128
    
    latent_tensor = torch.zeros((1, num_tokens, embedding_dim))
    clean_tensor = torch.zeros((1, num_tokens, embedding_dim))
    mask_tensor = torch.zeros((1, num_tokens))
    
    state = MockLatentState(latent_tensor, clean_tensor, mask_tensor)

    # Now create the condition
    # For params.end_frame_data, we create a dummy tensor
    # shape (1, 128, 1, 64, 64) - SINGLE FRAME latent
    cond_latent = torch.randn(1, 128, 1, height, width)
    
    # SCENARIO 1: The BUG
    # We pass the PIXEL index (e.g. 120) instead of LATENT index
    pixel_end_index = 120
    
    print(f"\n--- Scenario 1: Using Pixel Index {pixel_end_index} (The Bug) ---")
    condition_bug = VideoConditionByLatentIndex(
        latent=cond_latent,
        strength=1.0,
        latent_idx=pixel_end_index
    )
    
    try:
        # This calls patchifier.get_token_count with frames=120
        # In latent_cond.py:
        # start_token = latent_tools.patchifier.get_token_count(
        #     latent_tools.target_shape._replace(frames=self.latent_idx)
        # )
        # A mock might fail to _replace, let's fix the mock
        def target_replace(frames):
            m = MagicMock()
            # If frames=120, dimensions are huge -> huge token count
            # It blindly calculates token count for 120 frames * 64 * 64
            # Then tries to slice latent_state.latent[:, start_token:stop_token]
            # But latent_state ONLY has 15 frames worth of tokens!
            # So start_token will be way out of bounds.
            dims = list(latent_shape_torch)
            dims[2] = frames
            m.to_torch_shape.return_value = tuple(dims)
            return m
            
        latent_tools.target_shape._replace = target_replace
        
        condition_bug.apply_to(state, latent_tools)
        print("Scenairo 1 SUCCESS (Unexpected)")
    except Exception as e:
        print(f"Scenario 1 FAILED as expected with error: {e}")
        # We expect a slice error or index error

    # SCENARIO 2: The FIX
    # We pass the LATENT index (e.g. 15)
    latent_end_index = pixel_end_index // 8
    
    print(f"\n--- Scenario 2: Using Latent Index {latent_end_index} (The Fix) ---")
    condition_fix = VideoConditionByLatentIndex(
        latent=cond_latent,
        strength=1.0,
        latent_idx=latent_end_index
    )
    
    try:
        condition_fix.apply_to(state, latent_tools)
        print("Scenario 2 SUCCESS (Expected)")
    except Exception as e:
        print(f"Scenario 2 FAILED (Unexpected) with error: {e}")

if __name__ == "__main__":
    test_latent_index_mismatch()
