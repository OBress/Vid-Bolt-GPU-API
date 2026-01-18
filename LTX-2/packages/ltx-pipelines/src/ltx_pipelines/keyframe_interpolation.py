import logging
import threading
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
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
    guider_denoising_func,
    image_conditionings_by_adding_guiding_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class KeyframeInterpolationPipeline:
    """
    Keyframe-based Two-stage video interpolation pipeline.
    Interpolates between keyframes to generate a video with smoother transitions.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
        # Pre-loaded components for weight sharing (optional)
        # When provided, these are reused instead of loading new instances
        shared_text_encoder=None,
        shared_video_encoder=None,
        shared_transformer=None,
        shared_spatial_upsampler=None,
        shared_video_decoder=None,
        shared_audio_decoder=None,
        shared_vocoder=None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        
        # Check if we're using shared components (VRAM optimization)
        using_shared = shared_transformer is not None
        
        if not using_shared:
            # No shared components - create model ledgers for loading
            self.stage_1_model_ledger = ModelLedger(
                dtype=self.dtype,
                device=device,
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root_path=gemma_root,
                loras=loras,
                fp8transformer=fp8transformer,
            )
            self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
                loras=distilled_lora,
            )
        
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        
        if using_shared:
            # Use shared components from DistilledPipeline (saves ~60% VRAM)
            logging.info("Using shared components from DistilledPipeline (VRAM optimization)")
            self.text_encoder = shared_text_encoder
            self.video_encoder = shared_video_encoder
            self.transformer_stage_1 = shared_transformer
            self.spatial_upsampler = shared_spatial_upsampler
            self.transformer_stage_2 = shared_transformer  # Same transformer for both stages
            self.video_decoder = shared_video_decoder
            self.audio_decoder = shared_audio_decoder
            self.vocoder = shared_vocoder
            logging.info("KeyframeInterpolationPipeline initialized with shared components.")
        else:
            # Pre-load models to keep them in VRAM (original behavior)
            logging.info("Loading KeyframeInterpolationPipeline models into VRAM...")
            
            # Stage 1 Models
            self.text_encoder = self.stage_1_model_ledger.text_encoder()
            self.video_encoder = self.stage_1_model_ledger.video_encoder()
            self.transformer_stage_1 = self.stage_1_model_ledger.transformer()
            
            # Stage 2 Models
            self.spatial_upsampler = self.stage_2_model_ledger.spatial_upsampler()
            self.transformer_stage_2 = self.stage_2_model_ledger.transformer()
            self.video_decoder = self.stage_2_model_ledger.video_decoder()
            self.audio_decoder = self.stage_2_model_ledger.audio_decoder()
            self.vocoder = self.stage_2_model_ledger.vocoder()
            
            logging.info("KeyframeInterpolationPipeline models loaded.")
        
        # Thread lock for text encoder (not thread-safe due to HuggingFace tokenizer)
        self._text_encoder_lock = threading.Lock()

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        use_distilled_schedule: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """
        Generate a video by interpolating between keyframe images.

        Args:
            prompt: Text prompt describing the desired video content.
            negative_prompt: Text prompt describing what to avoid (ignored if use_distilled_schedule=True).
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate (should follow 8k+1 rule).
            frame_rate: Frames per second for the output video.
            num_inference_steps: Number of denoising steps (ignored if use_distilled_schedule=True).
            cfg_guidance_scale: CFG guidance scale (ignored if use_distilled_schedule=True).
            images: List of (image_path, frame_idx, strength) tuples for keyframe conditioning.
            tiling_config: Optional tiling configuration for VAE decoding.
            enhance_prompt: Whether to enhance the prompt using Gemma.
            use_distilled_schedule: If True, use 8-step distilled schedule for Stage 1 (faster).
                                    This disables CFG guidance and uses simple denoising.

        Returns:
            Tuple of (video_chunks_iterator, audio_tensor).
        """
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self.text_encoder
        
        # Serialize text encoder access (tokenizer/Gemma not thread-safe)
        with self._text_encoder_lock:
            if enhance_prompt:
                prompt = generate_enhanced_prompt(
                    text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
                )
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.video_encoder
        transformer = self.transformer_stage_1
        
        # Use distilled schedule (8 steps) or dynamic scheduler
        if use_distilled_schedule:
            sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
            logging.info("Using distilled 8-step schedule for Stage 1 (fast mode)")
        else:
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
            logging.info(f"Using standard scheduler with {num_inference_steps} steps for Stage 1")

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            # Choose denoising function based on schedule type
            # Distilled schedule: simple denoising (no CFG overhead)
            # Standard schedule: CFG-guided denoising for quality
            if use_distilled_schedule:
                denoise_fn = simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                )
            else:
                denoise_fn = guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,  # noqa: F821
                )
            
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_adding_guiding_latent(
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
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.spatial_upsampler,
        )

        transformer = self.transformer_stage_2
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_adding_guiding_latent(
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
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        decoded_video = vae_decode_video(video_state.latent, self.video_decoder, tiling_config)
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.audio_decoder, self.vocoder
        )
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
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
