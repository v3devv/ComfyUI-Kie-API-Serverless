"""Wan 2.7 Image Pro edit helper.

Implements the Alibaba Wan 2.7 Image Pro image-edit workflow on KIE.
API reference: https://docs.kie.ai/market/wan/2-7-image-pro.md
Payload shape and the constraints encoded below are cross-referenced
against the production-tested proteafield integration
(megabanana/app/services/kie_client.py, model_type="wan2_7_edit").

This module:
- validates inputs
- uploads the input image(s)
- creates a task
- polls until completion (shared helper)
- downloads and decodes the resulting image (shared helper)
"""

import json
import random
import time
from typing import Any

import torch

from .auth import _validate_api_key
from .credits import _log_remaining_credits
from .http import TransientKieError, requests
from .images import _download_image, _image_bytes_to_tensor
from .jobs import _poll_task_until_complete
from .log import _log
from .results import _extract_result_urls
from .upload import _image_tensor_to_png_bytes, _truncate_url, _upload_image
from .validation import _validate_prompt


CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
MODEL_NAME = "wan/2-7-image-pro"

# Fixed KIE enum. An aspect_ratio outside this set is rejected as
# "This aspect_ratio is not within the range of allowed options" and the
# task fails with no taskId, so it must be validated strictly up front.
ASPECT_RATIO_OPTIONS = ["1:1", "16:9", "4:3", "21:9", "3:4", "9:16", "8:1", "1:8"]

# 4K is intentionally omitted: KIE only allows 4K for non-sequential
# text-to-image. An edit always carries an input image, so 4K returns
# "resolution 4K is only supported for non-sequential text-to-image".
RESOLUTION_OPTIONS = ["1K", "2K"]

PROMPT_MAX_LENGTH = 5000
MAX_INPUT_IMAGES = 9
SEED_MAX = 2_147_483_647


def _validate_options(aspect_ratio: str, resolution: str) -> None:
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        raise RuntimeError("Invalid aspect_ratio. Use the pinned enum options.")
    if resolution not in RESOLUTION_OPTIONS:
        raise RuntimeError("Invalid resolution. Use the pinned enum options.")


def _validate_image_input(images: torch.Tensor | None) -> torch.Tensor:
    if images is None:
        raise RuntimeError("images input is required.")
    if not isinstance(images, torch.Tensor):
        raise RuntimeError("images input must be a tensor batch.")
    if images.dim() != 4 or images.shape[-1] != 3:
        raise RuntimeError("images input must have shape [B, H, W, 3].")
    if images.shape[0] < 1:
        raise RuntimeError("images input batch is empty.")
    return images


def _create_wan27_edit_task(api_key: str, payload: dict[str, Any]) -> tuple[str, str]:
    try:
        response = requests.post(
            CREATE_TASK_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call createTask endpoint: {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientKieError(
            f"createTask returned HTTP {response.status_code}: {response.text}", status_code=response.status_code
        )

    try:
        payload_json: Any = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("createTask endpoint did not return valid JSON.") from exc

    if payload_json.get("code") != 200:
        message = payload_json.get("message") or payload_json.get("msg")
        raise RuntimeError(f"createTask endpoint returned error code {payload_json.get('code')}: {message}")

    task_id = (payload_json.get("data") or {}).get("taskId")
    if not task_id:
        raise RuntimeError("createTask endpoint did not return a taskId.")

    return task_id, response.text


def run_wan27_edit_pro(
    prompt: str,
    images: torch.Tensor,
    api_key: str,
    aspect_ratio: str,
    resolution: str,
    watermark: bool,
    poll_interval_s: float,
    timeout_s: int,
    log: bool,
) -> torch.Tensor:
    """Run a Wan 2.7 Image Pro image-edit job.

    Args:
        prompt: Edit instruction text (max 5000 chars).
        images: ComfyUI IMAGE tensor batch (B, H, W, 3); up to 9 used.
        api_key: KIE API key.
        aspect_ratio: Output aspect ratio (fixed KIE enum).
        resolution: Output resolution ("1K" or "2K"; 4K invalid for edit).
        watermark: Add a watermark to the output image.
        poll_interval_s: Seconds between status polls.
        timeout_s: Maximum seconds to wait for completion.
        log: Enable verbose logging.

    Returns:
        ComfyUI IMAGE tensor (1, H, W, 3) float32 in [0, 1].

    Raises:
        RuntimeError: For validation errors or non-retryable API/task failures.
        TransientKieError: For retryable API/task failures.
    """
    _validate_prompt(prompt, max_length=PROMPT_MAX_LENGTH)
    _validate_options(aspect_ratio, resolution)
    images = _validate_image_input(images)

    api_key = _validate_api_key(api_key)

    total_images = images.shape[0]
    if total_images > MAX_INPUT_IMAGES and log:
        _log(log, f"More than {MAX_INPUT_IMAGES} images provided ({total_images}); only the first {MAX_INPUT_IMAGES} will be used.")
    upload_count = min(total_images, MAX_INPUT_IMAGES)

    _log(log, f"Uploading {upload_count} edit input image(s)...")
    input_urls: list[str] = []
    for idx in range(upload_count):
        png_bytes = _image_tensor_to_png_bytes(images[idx])
        url = _upload_image(api_key, png_bytes)
        input_urls.append(url)
        _log(log, f"Image {idx + 1} upload success: {_truncate_url(url)}")

    # KIE has no native "random" flag and seed=0 is not reliably random,
    # so roll a fresh seed every call (matches proteafield's approach).
    resolved_seed = random.randint(0, SEED_MAX)

    payload = {
        "model": MODEL_NAME,
        "input": {
            "prompt": prompt,
            "input_urls": input_urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "n": 1,
            "enable_sequential": False,
            "thinking_mode": False,
            "watermark": bool(watermark),
            "seed": resolved_seed,
            "nsfw_checker": False,
        },
    }

    _log(log, f"Creating Wan 2.7 Image Pro edit task (seed={resolved_seed})...")
    start_time = time.time()
    task_id, create_response_text = _create_wan27_edit_task(api_key, payload)
    _log(log, f"createTask response (elapsed={time.time() - start_time:.1f}s): {create_response_text}")
    _log(log, f"Task created with ID {task_id}. Polling for completion...")

    record_data = _poll_task_until_complete(
        api_key,
        task_id,
        poll_interval_s,
        timeout_s,
        log,
        start_time,
    )

    result_urls = _extract_result_urls(record_data)
    _log(log, f"Result URLs: {result_urls}")
    _log(log, f"Downloading result image from {result_urls[0]}...")

    image_bytes = _download_image(result_urls[0])
    image_tensor = _image_bytes_to_tensor(image_bytes)
    _log(log, "Image downloaded and decoded.")

    _log_remaining_credits(log, record_data, api_key, _log)
    return image_tensor
