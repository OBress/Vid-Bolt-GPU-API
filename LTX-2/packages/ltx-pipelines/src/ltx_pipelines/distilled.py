import logging
import threading
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

        # Pre-load models to keep them in VRAM
        logging.info("Loading DistilledPipeline models into VRAM...")
        logging.info("  [1/7] Loading text_encoder...")
        self.text_encoder = self.model_ledger.text_encoder()
        logging.info("  [2/7] Loading video_encoder...")
        self.video_encoder = self.model_ledger.video_encoder()
        logging.info("  [3/7] Loading transformer (this may take 1-2 minutes)...")
        self.transformer = self.model_ledger.transformer()
        logging.info("  [4/7] Loading spatial_upsampler...")
        self.spatial_upsampler = self.model_ledger.spatial_upsampler()
        logging.info("  [5/7] Loading video_decoder...")
        self.video_decoder = self.model_ledger.video_decoder()
        logging.info("  [6/7] Loading audio_decoder...")
        self.audio_decoder = self.model_ledger.audio_decoder()
        logging.info("  [7/7] Loading vocoder...")
        self.vocoder = self.model_ledger.vocoder()
        
        # Move models to device (if not handled by ledger builder)
        # Ledger returns models already on device, so this is just a sanity check/comment.
        logging.info("DistilledPipeline models loaded.")
        
        # Thread lock for text encoder (not thread-safe due to HuggingFace tokenizer)
        self._text_encoder_lock = threading.Lock()

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.text_encoder
        
        # Serialize text encoder access (tokenizer/Gemma not thread-safe)
        with self._text_encoder_lock:
            if enhance_prompt:
                prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.video_encoder
        transformer = self.transformer
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self.spatial_upsampler
        )

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        logging.info(f"DEBUG: video_state.latent (FP8 check): shape={video_state.latent.shape}, dtype={video_state.latent.dtype}, min={video_state.latent.min()}, max={video_state.latent.max()}, mean={video_state.latent.float().mean()}")
        
        # Check if latent is all zeros or NaNs
        if torch.all(video_state.latent == 0):
            logging.error("DEBUG CRITICAL: Video latent is ALL ZEROS before VAE decode!")
        if torch.isnan(video_state.latent).any():
            logging.error("DEBUG CRITICAL: Video latent contains NaNs before VAE decode!")

        decoded_video = vae_decode_video(video_state.latent, self.video_decoder, tiling_config)
        
        logging.info(f"DEBUG: decoded_video stats: shape={decoded_video.shape}, dtype={decoded_video.dtype}, min={decoded_video.min()}, max={decoded_video.max()}, mean={decoded_video.float().mean()}")
        
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.audio_decoder, self.vocoder
        )
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
