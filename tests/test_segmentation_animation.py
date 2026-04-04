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


def test_effects_pipeline_reveal_clips_overlay_progressively():
    image = Image.new("RGB", (40, 24), color=(255, 255, 255))
    mask = np.zeros((24, 40), dtype=bool)
    mask[:, 4:36] = True

    partial = EffectsPipeline(image=image, masks=[mask], boxes=[(4, 0, 35, 23)])
    partial.apply(
        [
            {"type": "select", "target": "mask"},
            {
                "type": "color_overlay",
                "color": [255, 0, 0, 255],
                "_animation_mode": "reveal",
                "_animation_progress": 0.25,
                "_animation_direction": "left",
            },
        ]
    )

    full = EffectsPipeline(image=image, masks=[mask], boxes=[(4, 0, 35, 23)])
    full.apply(
        [
            {"type": "select", "target": "mask"},
            {
                "type": "color_overlay",
                "color": [255, 0, 0, 255],
                "_animation_mode": "reveal",
                "_animation_progress": 1.0,
                "_animation_direction": "left",
            },
        ]
    )

    partial_arr = np.array(partial.image.convert("RGB"))
    full_arr = np.array(full.image.convert("RGB"))
    partial_red = np.count_nonzero(partial_arr[:, :, 1] == 0)
    full_red = np.count_nonzero(full_arr[:, :, 1] == 0)

    assert 0 < partial_red < full_red


def test_effects_pipeline_splash_clips_overlay_progressively():
    image = Image.new("RGB", (50, 30), color=(255, 255, 255))
    mask = np.zeros((30, 50), dtype=bool)
    mask[4:26, 6:44] = True

    partial = EffectsPipeline(image=image, masks=[mask], boxes=[(6, 4, 43, 25)])
    partial.apply(
        [
            {"type": "select", "target": "mask"},
            {
                "type": "color_overlay",
                "color": [0, 0, 255, 255],
                "_animation_mode": "splash",
                "_animation_progress": 0.2,
                "_animation_seed": 7,
            },
        ]
    )

    full = EffectsPipeline(image=image, masks=[mask], boxes=[(6, 4, 43, 25)])
    full.apply(
        [
            {"type": "select", "target": "mask"},
            {
                "type": "color_overlay",
                "color": [0, 0, 255, 255],
                "_animation_mode": "splash",
                "_animation_progress": 1.0,
                "_animation_seed": 7,
            },
        ]
    )

    partial_arr = np.array(partial.image.convert("RGB"))
    full_arr = np.array(full.image.convert("RGB"))
    partial_blue = np.count_nonzero(partial_arr[:, :, 0] == 0)
    full_blue = np.count_nonzero(full_arr[:, :, 0] == 0)

    assert 0 < partial_blue < full_blue


def test_effects_pipeline_stagger_sequences_objects_left_to_right():
    interpolator = AnimationInterpolator()
    image = Image.new("RGB", (60, 20), color=(255, 255, 255))
    left_mask = np.zeros((20, 60), dtype=bool)
    right_mask = np.zeros((20, 60), dtype=bool)
    left_mask[:, 4:22] = True
    right_mask[:, 38:56] = True
    base_op = {
        "type": "color_overlay",
        "color": [255, 0, 0, 255],
        "animation": {
            "mode": "stagger",
            "duration": 1.0,
            "stagger_delay": 0.3,
            "start": {"color": [255, 0, 0, 0]},
            "end": {"color": [255, 0, 0, 255]},
        },
    }
    early_op = interpolator.interpolate(base_op, 0.2, 1.0, 4, 20)
    late_op = interpolator.interpolate(base_op, 0.95, 1.0, 19, 20)

    early = EffectsPipeline(
        image=image,
        masks=[left_mask, right_mask],
        boxes=[(4, 0, 21, 19), (38, 0, 55, 19)],
    )
    early.apply([{"type": "select", "target": "mask"}, early_op])

    late = EffectsPipeline(
        image=image,
        masks=[left_mask, right_mask],
        boxes=[(4, 0, 21, 19), (38, 0, 55, 19)],
    )
    late.apply([{"type": "select", "target": "mask"}, late_op])

    early_arr = np.array(early.image.convert("RGB"))
    late_arr = np.array(late.image.convert("RGB"))
    early_left = np.count_nonzero(early_arr[:, 4:22, 1] < 255)
    early_right = np.count_nonzero(early_arr[:, 38:56, 1] < 255)
    late_right = np.count_nonzero(late_arr[:, 38:56, 1] < 255)

    assert early_left > 0
    assert early_right == 0
    assert late_right > 0


def test_bounding_box_progress_draws_partial_perimeter():
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    mask = np.ones((32, 32), dtype=bool)
    box = (8, 8, 23, 23)

    partial = EffectsPipeline(image=image, masks=[mask], boxes=[box])
    partial.apply(
        [
            {"type": "select", "target": "mask"},
            {"type": "bounding_box", "color": [255, 0, 0, 255], "thickness": 1, "progress": 0.25},
        ]
    )

    full = EffectsPipeline(image=image, masks=[mask], boxes=[box])
    full.apply(
        [
            {"type": "select", "target": "mask"},
            {"type": "bounding_box", "color": [255, 0, 0, 255], "thickness": 1, "progress": 1.0},
        ]
    )

    partial_arr = np.array(partial.image.convert("RGB"))
    full_arr = np.array(full.image.convert("RGB"))
    partial_red = np.count_nonzero(partial_arr[:, :, 1] == 0)
    full_red = np.count_nonzero(full_arr[:, :, 1] == 0)

    assert 0 < partial_red < full_red


def test_label_operation_tracks_distinct_positions_for_multiple_objects():
    image = Image.new("RGB", (120, 80), color=(240, 240, 240))
    left_mask = np.zeros((80, 120), dtype=bool)
    right_mask = np.zeros((80, 120), dtype=bool)
    left_mask[18:58, 12:42] = True
    right_mask[18:58, 72:108] = True
    annotation_state = {}

    pipeline = EffectsPipeline(
        image=image,
        masks=[left_mask, right_mask],
        boxes=[(12, 18, 41, 57), (72, 18, 107, 57)],
        labels=["alpha", "beta"],
        annotation_state=annotation_state,
    )
    pipeline.apply(
        [
            {"type": "select", "target": "mask"},
            {"type": "label", "background_color": [12, 20, 28, 220], "leader_line": True},
        ]
    )

    positions = annotation_state["label_positions"]
    rects = [tuple(state["rect"]) for state in positions.values()]

    assert len(rects) == 2
    assert rects[0] != rects[1]
    assert EffectsPipeline._rect_overlap_area(rects[0], rects[1]) == 0
