import inspect
import json
import sys
import types
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from app.config import Settings, get_settings
from app.models.job import JobInfo, JobStatus
from app.models.internal import VideoSegmentationParams
from app.models.segmentation import ImageSegmentRequest, VideoSegmentRequest
from app.routers.segmentation import _hydrate_segmentation_operations
from app.services.job_manager import JobManager
from app.services.sam3_generator import SAM3Generator
from app.services.segmentation_effects import EffectsPipeline


def _base_video_request(**overrides):
    payload = {
        "job_id": "seg-job-001",
        "input_video_url": "https://example.com/input.mp4",
        "save_url": "https://example.com/output.json",
    }
    payload.update(overrides)
    return payload


def _two_object_scene():
    image = Image.new("RGB", (6, 4), color=(255, 255, 255))
    left_mask = np.zeros((4, 6), dtype=bool)
    right_mask = np.zeros((4, 6), dtype=bool)
    left_mask[:, :3] = True
    right_mask[:, 3:] = True
    boxes = [(0, 0, 2, 3), (3, 0, 5, 3)]
    return image, left_mask, right_mask, boxes


def test_video_segment_request_rejects_conflicting_prompt_modes():
    with pytest.raises(PydanticValidationError, match="Only one of text_prompt, text_prompts, or object_prompts"):
        VideoSegmentRequest(
            **_base_video_request(
                text_prompt="person",
                text_prompts=["dog"],
            )
        )


def test_video_segment_request_rejects_duplicate_text_prompts():
    with pytest.raises(PydanticValidationError, match="text_prompts must be unique"):
        VideoSegmentRequest(
            **_base_video_request(
                text_prompts=["person", "person"],
            )
        )


def test_video_segment_request_rejects_visual_prompts_in_multi_prompt_mode():
    with pytest.raises(
        PydanticValidationError,
        match="point_prompts and box_prompts are only supported with the legacy single text_prompt video mode",
    ):
        VideoSegmentRequest(
            **_base_video_request(
                text_prompts=["person", "dog"],
                point_prompts=[[20, 20]],
            )
        )


def test_video_segment_request_rejects_duplicate_object_prompt_labels():
    with pytest.raises(PydanticValidationError, match="object_prompts labels must be unique"):
        VideoSegmentRequest(
            **_base_video_request(
                object_prompts=[
                    {"label": "person", "text": "person"},
                    {"label": "person", "text": "person in red"},
                ],
            )
        )


def test_effects_pipeline_selects_object_by_stable_id_for_mask_operations():
    image, left_mask, right_mask, boxes = _two_object_scene()
    pipeline = EffectsPipeline(
        image=image,
        masks=[left_mask, right_mask],
        boxes=boxes,
        labels=["person", "dog"],
        object_ids=[101, 202],
    )

    pipeline.apply(
        [
            {"type": "select", "target": "mask", "object_id": 202},
            {"type": "redact", "color": [255, 0, 0]},
        ]
    )

    result = np.array(pipeline.image.convert("RGB"))
    assert np.all(result[:, :3] == 255)
    assert np.all(result[:, 3:, 0] == 255)
    assert np.all(result[:, 3:, 1] == 0)
    assert np.all(result[:, 3:, 2] == 0)


def test_effects_pipeline_selects_object_by_label_for_object_aware_drawing():
    image, left_mask, right_mask, boxes = _two_object_scene()
    pipeline = EffectsPipeline(
        image=image,
        masks=[left_mask, right_mask],
        boxes=boxes,
        labels=["person", "dog"],
        object_ids=[101, 202],
    )

    pipeline.apply(
        [
            {"type": "select", "target": "mask", "object_label": "person"},
            {"type": "bounding_box", "color": [0, 255, 0, 255], "thickness": 1},
        ]
    )

    result = np.array(pipeline.image.convert("RGB"))
    assert tuple(result[0, 0]) == (0, 255, 0)
    assert tuple(result[0, 5]) == (255, 255, 255)


@pytest.mark.asyncio
async def test_hydrate_segmentation_operations_downloads_background_image(sample_image_bytes):
    storage = AsyncMock()
    storage.download_from_url = AsyncMock(return_value=sample_image_bytes)
    operations = [{"type": "replace_background", "image_url": "https://example.com/bg.png"}]

    hydrated = await _hydrate_segmentation_operations(storage, operations)

    assert "_bg_image_data" not in operations[0]
    assert hydrated[0]["_bg_image_data"] == sample_image_bytes


@pytest.mark.asyncio
async def test_sam3_dry_run_video_returns_prompt_mappings_and_tracking_metadata():
    generator = SAM3Generator(get_settings())
    params = VideoSegmentationParams(
        job_id="seg-job-002",
        input_video_data=b"mock-video",
        text_prompts=["person", "dog"],
        include_tracking_metadata=True,
    )

    result = await generator.segment_video(params)
    payload = json.loads(result.result_data.decode("utf-8"))

    assert result.prompt_to_obj_ids == {"person": [1], "dog": [2]}
    assert result.object_id_to_prompt_label == {1: "person", 2: "dog"}
    assert payload["prompt_to_obj_ids"] == {"person": [1], "dog": [2]}
    assert payload["object_id_to_prompt_label"] == {"1": "person", "2": "dog"}
    assert payload["frames"]["0"]["1"]["label"] == "person"
    assert payload["frames"]["0"]["2"]["label"] == "dog"
    assert payload["include_tracking_metadata"] is True
    assert result.model_version == get_settings().sam3_video_model_version


def test_segmentation_stack_exposes_no_prompt_enhancement_surface():
    assert "enhance_prompt" not in ImageSegmentRequest.model_fields
    assert "enhance_prompt" not in VideoSegmentRequest.model_fields

    sam3_source = inspect.getsource(SAM3Generator)
    assert "enhance_prompt" not in sam3_source
    assert "apply_chat_template" not in sam3_source


def test_sam3_disables_fa3_on_blackwell_even_if_module_is_present(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        )
    )

    monkeypatch.setattr(
        "app.services.sam3_generator.importlib.util.find_spec",
        lambda name: object() if name == "flash_attn_interface" else None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    enabled, reason = SAM3Generator._resolve_video_use_fa3()

    assert enabled is False
    assert "Blackwell" in reason


def test_sam3_load_models_passes_sdpa_fallback_to_video_builder(monkeypatch):
    settings = Settings(mock_mode=False, api_key="test-key", sam3_dry_run_override=False)
    generator = SAM3Generator(settings)
    captured = {}

    fake_sam3_pkg = types.ModuleType("sam3")
    fake_sam3_pkg.__path__ = []
    fake_sam3_model_pkg = types.ModuleType("sam3.model")
    fake_sam3_model_pkg.__path__ = []
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_processor_module = types.ModuleType("sam3.model.sam3_image_processor")

    def fake_download_ckpt_from_hf(version):
        return f"/tmp/{version}.pt"

    def fake_build_sam3_image_model(**kwargs):
        captured["image_kwargs"] = kwargs
        return object()

    def fake_build_sam3_predictor(**kwargs):
        captured["video_kwargs"] = kwargs
        return object()

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

    fake_model_builder.download_ckpt_from_hf = fake_download_ckpt_from_hf
    fake_model_builder.build_sam3_image_model = fake_build_sam3_image_model
    fake_model_builder.build_sam3_predictor = fake_build_sam3_predictor
    fake_processor_module.Sam3Processor = FakeProcessor

    monkeypatch.setitem(sys.modules, "sam3", fake_sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model", fake_sam3_model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", fake_processor_module)
    monkeypatch.setattr(
        SAM3Generator,
        "_resolve_video_use_fa3",
        staticmethod(lambda: (False, "detected Blackwell GPU")),
    )

    generator.load_models()

    assert captured["image_kwargs"]["checkpoint_path"] == "/tmp/sam3.1.pt"
    assert captured["image_kwargs"]["load_from_HF"] is False
    assert captured["video_kwargs"]["version"] == settings.sam3_video_model_version
    assert captured["video_kwargs"]["use_fa3"] is False
    assert generator.get_status()["video_attention_backend"] == "torch_sdpa"


def test_sam3_coerce_output_sequence_handles_numpy_arrays():
    assert SAM3Generator._coerce_output_sequence(None) == []
    assert SAM3Generator._coerce_output_sequence(np.array([], dtype=np.int64)) == []
    assert SAM3Generator._coerce_output_sequence(np.array([1, 2], dtype=np.int64)) == [1, 2]


def test_job_manager_pending_check_can_ignore_current_job_ids():
    manager = JobManager(get_settings())
    current_job = JobInfo(job_id="seg-job-1", status=JobStatus.PENDING, created_at=0.0)

    manager._jobs[current_job.job_id] = current_job
    manager._pending_jobs_set.add(current_job.job_id)

    assert manager.has_pending_or_active_jobs() is True
    assert manager.has_pending_or_active_jobs(ignore_job_ids={current_job.job_id}) is False
