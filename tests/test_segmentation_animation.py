import numpy as np
from PIL import Image

from app.services.segmentation_animation import AnimationInterpolator, AnimationPipeline
from app.services.segmentation_effects import EffectsPipeline


def test_animation_interpolator_infers_neutral_blur_and_grayscale_defaults():
    interpolator = AnimationInterpolator()

    blur_op = {
        "type": "blur",
        "strength": 25,
        "animation": {"mode": "transition", "duration": 3.0},
    }
    grayscale_op = {
        "type": "grayscale",
        "intensity": 1.0,
        "animation": {"mode": "transition", "duration": 3.0},
    }

    blur_start = interpolator.interpolate(blur_op, 0.0, 3.0, 0, 72)
    blur_mid = interpolator.interpolate(blur_op, 0.5, 3.0, 36, 72)
    blur_end = interpolator.interpolate(blur_op, 1.0, 3.0, 71, 72)

    grayscale_start = interpolator.interpolate(grayscale_op, 0.0, 3.0, 0, 72)
    grayscale_mid = interpolator.interpolate(grayscale_op, 0.5, 3.0, 36, 72)
    grayscale_end = interpolator.interpolate(grayscale_op, 1.0, 3.0, 71, 72)

    assert blur_start["strength"] == 0.0
    assert 0.0 < blur_mid["strength"] < 25
    assert blur_end["strength"] == 25

    assert grayscale_start["intensity"] == 0.0
    assert 0.0 < grayscale_mid["intensity"] < 1.0
    assert grayscale_end["intensity"] == 1.0


def test_effects_pipeline_blur_strength_zero_is_noop():
    image_arr = np.zeros((8, 8, 3), dtype=np.uint8)
    image_arr[:, :4] = [255, 0, 0]
    image_arr[:, 4:] = [0, 0, 255]
    image = Image.fromarray(image_arr, "RGB")

    full_mask = np.ones((8, 8), dtype=bool)
    pipeline = EffectsPipeline(image=image, masks=[full_mask])
    pipeline.apply(
        [
            {"type": "select", "target": "all"},
            {"type": "blur", "strength": 0},
        ]
    )

    result = np.array(pipeline.image.convert("RGB"))
    assert np.array_equal(result, image_arr)


def test_animation_pipeline_frame_diagnostics_detects_motion():
    image_arr = np.zeros((12, 12, 3), dtype=np.uint8)
    image_arr[:, :6] = [255, 0, 0]
    image_arr[:, 6:] = [0, 255, 0]
    image = Image.fromarray(image_arr, "RGB")

    pipeline = AnimationPipeline(
        image=image,
        masks=[np.zeros((12, 12), dtype=bool)],
        fps=3,
        duration=1.0,
    )

    start_frame = np.array(image.convert("RGB"))
    end_pipeline = EffectsPipeline(image=image, masks=[np.zeros((12, 12), dtype=bool)])
    end_pipeline.apply(
        [
            {"type": "select", "target": "all"},
            {"type": "grayscale", "intensity": 1.0},
        ]
    )
    end_frame = np.array(end_pipeline.image.convert("RGB"))

    metrics = pipeline._frame_delta_metrics(start_frame, end_frame)

    assert metrics["mean_abs_pixel_diff"] > 0.5
    assert metrics["changed_pixel_ratio"] > 0.1
