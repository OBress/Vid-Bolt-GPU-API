"""Tests for music generation request handling and ACE-Step backend selection."""

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.dependencies import get_job_manager
from app.main import app
from app.models.music_generation import MusicGenerateRequest
from app.services.acestep_generator import ACEStepGenerator


def _fake_torch(device_name: str, cuda_available: bool = True) -> ModuleType:
    """Build a minimal torch module stub for backend selection tests."""
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda *_args, **_kwargs: device_name,
    )
    return module


def test_music_request_normalizes_lyrics_list():
    """Lyrics arrays should be accepted and converted into ACE-Step's multiline string."""
    request = MusicGenerateRequest(
        job_id="music-job-1",
        prompt="Warm ambient underscore",
        lyrics=[
            "[Ambient Bed]",
            "[Texture - warm analog pad, barely audible drone, spacious]",
        ],
        duration_seconds=30,
        save_url="https://example.com/output.wav",
    )

    assert request.lyrics == (
        "[Ambient Bed]\n"
        "[Texture - warm analog pad, barely audible drone, spacious]"
    )


def test_music_generate_endpoint_accepts_lyrics_list(client, api_key_headers):
    """The API should queue music jobs even when the client sends lyrics as a list."""
    mock_job_manager = MagicMock()
    mock_job_manager.try_submit_job = AsyncMock(return_value=True)
    app.dependency_overrides[get_job_manager] = lambda: mock_job_manager

    try:
        response = client.post(
            "/api/v1/music/generate",
            headers=api_key_headers,
            json={
                "job_id": "music-job-2",
                "prompt": "Warm ambient underscore",
                "lyrics": [
                    "[Ambient Bed]",
                    "[Texture - gentle swell then recede, atmospheric]",
                ],
                "duration_seconds": 30,
                "save_url": "https://example.com/output.wav",
                "webhook_url": "https://example.com/webhook",
            },
        )
    finally:
        app.dependency_overrides.pop(get_job_manager, None)

    assert response.status_code == 202
    assert mock_job_manager.try_submit_job.await_count == 1

    submitted_params = mock_job_manager.try_submit_job.await_args.kwargs["params"]
    assert submitted_params.lyrics == (
        "[Ambient Bed]\n"
        "[Texture - gentle swell then recede, atmospheric]"
    )


def test_acestep_blackwell_prefers_pytorch_backend():
    """Blackwell GPUs should skip nano-vllm's unstable CUDA graph capture path."""
    fake_torch = _fake_torch("NVIDIA RTX PRO 6000 Blackwell Server Edition")

    with patch.dict("sys.modules", {"torch": fake_torch}):
        backend, reason = ACEStepGenerator._resolve_llm_backend()

    assert backend == "pt"
    assert "Blackwell" in reason


def test_acestep_non_blackwell_keeps_vllm_backend():
    """Non-Blackwell CUDA GPUs should keep the nano-vllm backend."""
    fake_torch = _fake_torch("NVIDIA RTX 4090")

    with patch.dict("sys.modules", {"torch": fake_torch}):
        backend, reason = ACEStepGenerator._resolve_llm_backend()

    assert backend == "vllm"
    assert "nano-vllm" in reason
