# SAM 3.1 Segmentation Stack Audit for GP API

Date: April 1, 2026

## Executive Conclusion

The GP API segmentation system **partially exposes** SAM 3.1 for agentic media-editing use cases. It is solid for segmentation-driven masking, tracking, redaction, stylization, and simple animated effects, but it does **not** fully take advantage of the official SAM 3.1 surface.

The main reason is architectural:

- The **video segmentation path** does use **SAM 3.1 Object Multiplex**.
- The **image segmentation path** still uses the base **SAM 3** image model, not a SAM 3.1 image checkpoint.
- The **image-to-video animation path** also starts from the base **SAM 3** image model, then runs a custom PIL/OpenCV animation pipeline.

So the system is best described as:

- **SAM 3 image segmentation**
- **SAM 3.1 video tracking**
- **custom GP API effects/animation/encoding layer**

From a client perspective, the stack is good for:

- image-to-image mask-driven edits
- video-to-video tracked mask-driven edits
- image-to-video animated effects

It is **not** yet a full-featured, agent-optimized SAM 3.1 interface because it does not expose several important official capabilities and best practices:

- no exemplar-prompt API
- no multi-text-prompt video API
- no prompt-to-object mapping in video results
- no stable object-ID-based effect targeting in video
- no semantic segmentation output
- no exposed iterative video refinement session

There are also a few documentation mismatches where the API docs promise more than the implementation currently delivers.

## Scope and Sources

This audit evaluates the GP API segmentation system as an external client would consume it:

- `POST /api/v1/segment/image`
- `POST /api/v1/segment/video`
- `POST /api/v1/segment/animate`
- the async job/polling/webhook contract around those endpoints
- the internal mask-to-effects pipeline that produces edited image/video outputs

Primary external sources used for SAM capability claims:

- [facebookresearch/sam3 README](https://raw.githubusercontent.com/facebookresearch/sam3/main/README.md)
- [facebookresearch/sam3 SAM 3.1 release notes](https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md)
- [SAM 3 paper on arXiv](https://arxiv.org/abs/2511.16719)
- [Hugging Face SAM3 docs](https://huggingface.co/docs/transformers/en/model_doc/sam3)
- [Hugging Face SAM3 Video docs](https://huggingface.co/docs/transformers/en/model_doc/sam3_video)

Primary internal sources used for GP API behavior:

- `app/routers/segmentation.py`
- `app/models/segmentation.py`
- `app/models/segmentation_animation.py`
- `app/services/sam3_generator.py`
- `app/services/segmentation_effects.py`
- `app/services/segmentation_animation.py`
- `app/models/job.py`
- `app/routers/jobs.py`
- `app/services/storage.py`
- `app/services/job_manager.py`
- `API.md`

## System Overview

At a high level, the GP API segmentation stack is not a generative model. It is a **segmentation-plus-effects** system:

1. SAM detects or tracks objects from prompts.
2. The API converts masks into base64 PNGs or applies ordered image/video effects.
3. The edited result is uploaded to a caller-provided presigned URL.
4. The job is surfaced through polling and optional webhooks.

This means the GP API can do **segmentation-driven editing**, not arbitrary scene regeneration.

### Async Job Contract

All three segmentation endpoints are async:

- They return HTTP `202 Accepted`.
- The immediate response includes `job_id` and `status_url`.
- Clients poll `GET /api/v1/jobs/{job_id}` for status.
- Optional webhooks can be configured.

Client-relevant behavior:

- Inputs are fetched from remote URLs before the job is queued.
- Outputs are uploaded via presigned `PUT` URLs.
- The returned `save_url` is the clean URL without query params.
- Job status supports `pending`, `processing`, `completed`, `failed`, `cancelled`.
- Queue position is exposed while pending.
- Progress fields exist on the generic job model, but the segmentation tasks do not currently provide rich stage-level progress updates.

### Segmentation-Specific Stack Boundaries

The segmentation system is separate from the rest of GP API media generation:

- `LightX2V` handles image editing elsewhere.
- `LTX-2` handles generative video elsewhere.
- The segmentation endpoints do not hand off to those generators.

So the segmentation stack should be judged on:

- segmentation quality
- promptability
- multi-object tracking
- edit composability
- agent-facing output stability

not on full generative transformation breadth.

## Endpoint-by-Endpoint Review

### 1. `POST /api/v1/segment/image`

### What the endpoint promises

Documented inputs:

- text prompts
- point prompts
- positive/negative box prompts
- `object_prompts` for named per-object targeting
- raw masks or processed image output
- ordered visual operations

### What it actually does

Implementation path:

- loads the image
- runs `build_sam3_image_model()` and `Sam3Processor`
- applies one of:
  - `object_prompts`
  - single `text_prompt`
  - labeled boxes
  - simple boxes
  - points
- extracts instance masks, boxes, scores
- optionally runs the CPU effects pipeline
- uploads either:
  - JSON list of base64 PNG masks
  - processed PNG image

### Strengths

- Good support for **segmentation-driven image editing**.
- `object_prompts` gives the image endpoint the best multi-object targeting surface in the stack.
- Ordered effects are composable and agent-friendly.
- Output metadata includes:
  - `object_count`
  - `width`
  - `height`
  - `boxes`
  - `scores`
  - `labels` when `object_prompts` are used

### Important limitations

- This path is **SAM 3, not SAM 3.1**.
- No exemplar prompt support.
- No semantic segmentation output.
- No exposed mask prompt API.
- No batch segmentation endpoint.
- No exposed embedding reuse/caching API for repeated prompt workloads.
- Complex relational prompts rely entirely on the raw text prompt; there is no SAM 3 Agent layer in the GP API.

### Client verdict

For image-to-image segmentation-driven editing, this endpoint is **useful and reasonably agent-friendly**, but it is **not a full SAM 3.1 exposure**.

---

### 2. `POST /api/v1/segment/video`

### What the endpoint promises

Documented inputs:

- one text prompt
- point prompts
- box prompts
- prompt frame index
- propagation direction
- masks JSON or processed MP4
- ordered per-frame operations

### What it actually does

Implementation path:

- writes uploaded video bytes to a temp MP4
- starts a SAM video session
- builds the predictor with `build_sam3_predictor(version="sam3.1")`
- sends one `add_prompt` request with any provided text/points/boxes
- propagates through the video with SAM 3.1
- either:
  - returns per-frame mask PNGs in JSON
  - applies framewise effects and encodes MP4

### Strengths

- This is the only path that clearly uses **SAM 3.1 Object Multiplex**.
- It supports text, point, and box prompting on video.
- It supports prompt application on an arbitrary frame.
- It preserves tracked object IDs in the raw JSON output.
- It processes a fully available video file, which aligns better with official best practice than weaker live-stream inference.
- It can apply a substantial library of effects to tracked masks and encode them back to MP4.

### Important limitations

- The public schema only accepts a **single** `text_prompt: string`.
  - Official SAM 3.1 video docs support **multiple text prompts at once** and return `prompt_to_obj_ids`.
- The API does **not** expose `prompt_to_obj_ids`.
- The API does not expose exemplar prompts.
- The API does not expose iterative session refinement.
  - Official SAM examples include adding/refining objects with point prompts after session start.
  - GP API wraps session lifecycle into one job, so callers cannot interactively refine mid-session.
- For edited video output, operations receive only a per-frame mask list.
  - No object labels are passed into the effects pipeline.
  - No object IDs are exposed to the effects selector.
  - `object_label` targeting does not work for video output.
  - `object_index` targeting is not a safe stable contract for video because the mask list is not externally guaranteed to map stably to a specific tracked object across frames.
- The API does not return per-frame boxes, scores, or prompt/object associations in the public JSON payload.

### Client verdict

For video-to-video segmentation-driven editing, this endpoint is **the strongest part of the stack**, but it still **does not fully expose SAM 3.1’s multi-object agentic potential**.

---

### 3. `POST /api/v1/segment/animate`

### What the endpoint promises

Documented behavior:

- segment an image
- animate effect parameters over time
- output MP4
- support easing, draw, pulse, reveal, loop, stagger, zoom, and pan

### What it actually does

Implementation path:

- loads the image
- runs the same image-side SAM model path used elsewhere
- extracts masks
- feeds them into a custom `AnimationPipeline`
- interpolates operation params over time
- applies effects frame-by-frame
- encodes MP4 with OpenCV

### Strengths

- Good custom animation layer for segmentation-driven motion graphics.
- Strong effect interpolation system.
- Useful for:
  - reveal videos
  - outlines
  - spotlight/bokeh sequences
  - Ken Burns style zoom/pan

### Important limitations

- This path also starts from **SAM 3 image segmentation**, not SAM 3.1.
- It is not a generative image-to-video model.
- It does not support `object_prompts`, so label-based per-object animation is missing.
- It does not expose stable object identities.
- It does not expose exemplar prompts, semantic segmentation, or iterative refinement.

### Client verdict

For image-to-video segmented animation, this is a **good custom effects pipeline**, but not a full SAM 3.1-powered image/video reasoning surface.

## Capability Matrix

| Capability | Official SAM 3 / 3.1 | GP API Docs | GP API Actual Implementation |
| --- | --- | --- | --- |
| Open-vocabulary image segmentation | Yes | Yes | Yes |
| Open-vocabulary video segmentation/tracking | Yes | Yes | Yes |
| SAM 3.1 Object Multiplex video tracking | Yes | Implied by “SAM 3.1” | Yes, on video path only |
| SAM 3.1 image checkpoint usage | Available via SAM 3.1 release family | Docs imply yes | No, image path uses `build_sam3_image_model()` with `sam3` checkpoint |
| Image exemplar prompts | Yes | No | No |
| Combined text + visual prompt refinement on image | Yes | Partially implied | Yes for text + negative box style image use cases |
| Multi-text-prompt video tracking | Yes | No | No |
| `prompt_to_obj_ids` mapping for video | Yes in official examples/docs | No | No |
| Interactive video refinement/additional prompts | Yes | No | No public API exposure |
| Semantic segmentation output | Yes | No | No |
| Batch/embedding reuse patterns | Yes | No | No public API exposure |
| Named per-object image targeting | Not a core SAM feature, but easy to build on top | Yes | Yes on image endpoint |
| Named per-object video targeting | Not built into base effects layer automatically | Docs imply same ops as image | No stable label-aware video targeting |
| Processed image output | Not native SAM feature | Yes | Yes |
| Processed video output | Not native SAM feature | Yes | Yes |
| Animation/easing layer | Not native SAM feature | Yes | Yes |

## Workflow Assessment

### Image-to-Image

Verdict: **Partially strong**

What works well:

- segmentation-driven masking
- multiple effects in sequence
- multi-object image editing through `object_prompts`
- machine-readable outputs for downstream agents

What does not:

- exemplar-driven prompting
- semantic outputs
- full SAM 3.1 parity
- advanced prompt decomposition

Best use cases:

- blur/redact/privacy workflows
- compositing/background changes
- selective stylization and overlays
- mask extraction for downstream tools

### Image-to-Video

Verdict: **Useful but custom, not full SAM 3.1**

What works well:

- animated effects on segmented subjects
- reveal/highlight/stylized motion graphics
- simple cinematic subject emphasis

What does not:

- generative motion synthesis
- label-aware multi-object animation
- exemplar or semantic prompt surfaces
- true SAM 3.1 image-side parity

Best use cases:

- social/video graphics
- spotlight/outline/glow reveals
- product or subject emphasis clips

### Video-to-Video

Verdict: **Strongest current path, still partial**

What works well:

- prompt-driven detection and tracking
- tracked blur, redaction, outline, grading, spotlight, etc.
- raw per-frame mask output for downstream post-processing

What does not:

- explicit multi-category prompt lists
- prompt-to-object mapping
- stable object-targeted effect routing
- iterative in-job refinement sessions

Best use cases:

- privacy redaction
- tracked overlays/highlights
- background treatment around detected subjects
- downstream agent post-processing from mask JSON

## Complex Prompt and Multi-Object Assessment

### Can it handle complex prompts with multiple objects and multiple effects?

**Image endpoint:** yes, within limits.

- `object_prompts` lets a client send multiple named prompts.
- Effects are ordered and composable.
- Per-object selection is workable on images.

**Video endpoint:** only partially.

- It can track multiple matching objects for one text prompt.
- It can also take point and box prompts.
- It can apply multiple effects in sequence.
- But it does not expose official multi-text video prompting, prompt-to-object mapping, or stable per-object effect routing.

**Animate endpoint:** only partially.

- It can animate many effects.
- It cannot label objects the way the image endpoint can.

### Agent-facing reliability

Good:

- async job contract is clean
- output locations are deterministic
- raw mask JSON is machine-readable
- image endpoint returns labels for `object_prompts`

Weak:

- video results do not tell the agent which prompt produced which tracked IDs
- video edited outputs do not support stable object-ID-based effect selection
- complex natural-language disambiguation is limited to the base text prompt path; no integrated agentic decomposition layer exists

## Best-Practice Review Against Official SAM 3.1

### Where the GP API aligns well

- Uses SAM 3.1 Object Multiplex on video tracking.
- Uses full-video session processing rather than a weaker live-stream-only pattern.
- Supports text, point, and box prompting on video.
- Preserves tracked IDs in raw JSON outputs.
- Builds practical product value on top of SAM through compositing and animation.

### Where it diverges

- Image and animate paths are still SAM 3, not clearly SAM 3.1.
- No exemplar prompts.
- No multi-text video prompts.
- No prompt-to-object mapping.
- No semantic segmentation output.
- No public iterative session API for refine/add/remove object workflows.
- No segmentation batching or embedding reuse surface for high-throughput agent use.

## Documentation and Contract Mismatches

These are the most important documented-vs-actual gaps.

### 1. “SAM 3.1” is overstated for the whole segmentation stack

The docs describe segmentation as powered by SAM 3.1, but the implementation loads:

- `build_sam3_image_model()` for image segmentation
- `build_sam3_predictor(version="sam3.1")` for video segmentation

So the stack is not uniformly SAM 3.1.

### 2. `replace_background.image_url` is documented but not wired

The effects pipeline only supports preloaded `_bg_image_data`, and the segmentation routers never download or inject `image_url` for that operation.

Result:

- `color` replacement works
- `image_url` replacement is documented but effectively unsupported

### 3. `text_label` is documented but functionally a no-op

The docs and models still mention `text_label`, but the effects pipeline explicitly no-ops it.

### 4. Video effect targeting is weaker than the docs imply

The docs say video operations are the same as image operations, but image-only label-aware selection semantics do not carry over cleanly to video output because labels are not supplied to the video effects pipeline.

### 5. `remove_background` semantics differ by medium

- On image output, transparency is preserved in PNG.
- On video output, frames are converted back to RGB for MP4 encoding, so transparency cannot survive.

## Prioritized Gaps

### High impact

1. Upgrade image-side segmentation to the SAM 3.1 checkpoint family or correct the docs everywhere they claim full SAM 3.1 coverage.
2. Add `text_prompts: string[]` for video and return `prompt_to_obj_ids`.
3. Add stable object-targeting for video effects based on tracked object ID, not just mask order.
4. Add exemplar-prompt support for image and, where practical, video initialization.

### Medium impact

5. Add a public iterative video session API for add/refine/remove object workflows.
6. Add `object_prompts`-style labeling for animation workflows.
7. Expose semantic segmentation output as an optional response mode.
8. Add segmentation batch endpoints or reusable embedding patterns for agent-heavy workloads.

### Low-to-medium impact

9. Wire `replace_background.image_url` properly or remove it from docs.
10. Remove `text_label` from docs/schema or implement it.
11. Clarify that video `remove_background` cannot preserve alpha in MP4.

## Recommendations

### Recommended next steps for agent-friendly parity

1. **Fix the truth in labeling first.**
   - Either move image-side segmentation to SAM 3.1 checkpoints or change docs to “SAM 3 for image, SAM 3.1 for video.”

2. **Make video prompting truly multi-object and agent-ready.**
   - Add `text_prompts: string[]`.
   - Return `prompt_to_obj_ids`.
   - Return per-frame boxes/scores as an optional richer JSON mode.

3. **Add stable effect routing for video.**
   - Support `object_id` in `select`.
   - Preserve object ordering/mapping through the video effects path.

4. **Expose exemplar and refinement workflows.**
   - Exemplar prompts are one of the most important official SAM differentiators missing from the API.
   - Iterative refine/add/remove is highly valuable for agent callers.

5. **Clean up the docs to match the product.**
   - Remove or fix `text_label`.
   - Remove or wire `replace_background.image_url`.
   - Clarify medium-specific behavior around alpha/transparency.

## Final Verdict

The GP API segmentation system is a **useful segmentation-driven editing product**, but it does **not fully expose SAM 3.1 as a client-facing capability platform**.

If the bar is:

“Can an agent call this API to do practical segmentation-driven image edits, tracked video edits, and animated mask-based effects?”

the answer is **yes**.

If the bar is:

“Does this API take full advantage of official SAM 3.1 capabilities across image-to-image, image-to-video, and video-to-video workflows, including rich multi-object promptability, exemplar prompts, stable object mappings, and advanced agent-facing semantics?”

the answer is **no, not yet**.

The strongest current path is **video segmentation/tracking**. The weakest parity points are **image-side SAM 3.1 coverage**, **exemplar support**, **multi-prompt video exposure**, and **stable agent-facing object control in edited video outputs**.
