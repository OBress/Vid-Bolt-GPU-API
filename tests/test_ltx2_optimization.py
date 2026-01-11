
import unittest
from unittest.mock import MagicMock
import torch
from app.services.ltx2_generator import LTX2Generator
from app.config import Settings

class TestLTX2Optimization(unittest.TestCase):
    def setUp(self):
        self.settings = MagicMock(spec=Settings)
        # Mock settings required for init
        self.settings.ltx2_dry_run = True
        self.generator = LTX2Generator(self.settings)

    def test_crop_and_trim_video(self):
        # Create a dummy video tensor: (Frames, Height, Width, Channels)
        # 10 frames, 128x128, 3 channels
        frames, h, w, c = 10, 128, 128, 3
        video = torch.zeros((frames, h, w, c))
        
        # Mark the center to ensure cropping is correct
        # Center is at 64, 64
        video[:, 60:68, 60:68, :] = 1.0
        
        target_w, target_h = 64, 64
        target_frames = 5
        
        cropped = self.generator._crop_and_trim_video(
            video,
            current_width=w,
            current_height=h,
            target_width=target_w,
            target_height=target_h,
            target_frames=target_frames
        )
        
        # Check dimensions
        self.assertEqual(cropped.shape, (target_frames, target_h, target_w, c))
        
        # Check content (should be all 1.0 because we cropped the center 8x8 which is inside 64x64)
        # The center 8x8 of 128x128 corresponds to indices [60:68].
        # The crop window for 64x64 from 128x128 is [32:96].
        # So [60:68] is inside [32:96].
        # In the new cropped tensor, indices will be [60-32 : 68-32] = [28:36].
        
        center_val = cropped[0, 28:36, 28:36, :].mean().item()
        self.assertEqual(center_val, 1.0)
        
        # Ensure outside content is removed/not present (implicit by shape check and value check)

    def test_trim_audio(self):
        # Audio: (Samples, Channels)
        # 48kHz, 10 seconds = 480k samples
        sample_rate = 48000
        original_duration = 10.0
        target_duration = 5.0
        
        samples = int(original_duration * sample_rate)
        audio = torch.zeros((samples, 2))
        
        trimmed = self.generator._trim_audio(
            audio,
            target_duration=target_duration,
            sample_rate=sample_rate
        )
        
        expected_samples = int(target_duration * sample_rate)
        self.assertEqual(trimmed.shape[0], expected_samples)
        self.assertEqual(trimmed.shape[1], 2)

    def test_trim_audio_1d(self):
        # Audio: (Samples,)
        sample_rate = 48000
        original_duration = 10.0
        target_duration = 5.0
        
        samples = int(original_duration * sample_rate)
        audio = torch.zeros((samples,))
        
        trimmed = self.generator._trim_audio(
            audio,
            target_duration=target_duration,
            sample_rate=sample_rate
        )
        
        expected_samples = int(target_duration * sample_rate)
        self.assertEqual(trimmed.shape[0], expected_samples)

if __name__ == '__main__':
    unittest.main()
