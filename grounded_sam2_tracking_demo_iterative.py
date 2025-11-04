#!/usr/bin/env python3
"""
Grounded SAM 2 iterative tracking demo.

This script combines the robustness of the continuous-ID pipeline
with the ergonomics and CLI options of the standard tracking demo.
It periodically re-runs Grounding DINO (every --detection-interval frames)
to discover new objects that appear mid-video, keeps object identities
consistent via a lightweight dictionary, and supports all of the
quality-of-life features from the iterative tracking script:
 - Frame extraction for video files (MP4/MOV/M4V) with optional reuse.
 - Optional frame down-sampling (--target-frame-rate) and hard limits (--max-frames).
 - CPU/GPU offload toggles for decoded frames and cached masks.
 - Mask caching to disk (_mask_cache/) plus final annotated frames and video.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if platform.system() == "Darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.video_utils import create_video_from_images

try:
    import decord  # type: ignore
except ImportError:  # pragma: no cover
    decord = None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded SAM 2 iterative tracking demo")
    parser.add_argument(
        "--video-source",
        required=True,
        help="Path to a video file (MP4/MOV/M4V) or a directory of JPEG frames.",
    )
    parser.add_argument(
        "--text",
        default="car.",
        help="Grounding prompt (lowercase, ending with a dot).",
    )
    parser.add_argument(
        "--output-dir",
        default="./tracking_results_iterative",
        help="Directory where annotated frames and caches are written.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional path for the output MP4. Defaults to grounded_sam2_tracking_<stem>.mp4.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate (fps) for the output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames (after any down-sampling).",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=20,
        help="Run Grounding DINO every N frames to pick up new objects.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for re-using an existing object ID when new detections arrive.",
    )
    parser.add_argument(
        "--sam2-variant",
        default="sam2.1_hiera_large",
        choices=[
            "sam2.1_hiera_large",
            "sam2.1_hiera_base_plus",
        ],
        help="Which SAM 2 checkpoint/config variant to use.",
    )
    parser.add_argument(
        "--dino-model",
        default="large",
        choices=["large", "tiny"],
        help="Grounding DINO checkpoint to use ('large' = rziga/mm_grounding_dino_large_all, 'tiny' = IDEA-Research/grounding-dino-tiny).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run Grounding DINO and SAM 2 in half precision (FP16) when supported.",
    )
    parser.add_argument(
        "--max-dino-long-edge",
        type=int,
        default=1024,
        help="Resize DINO input so the longest edge is <= this value. <=0 disables resizing.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Original FPS when supplying a folder of frames. Needed if --target-frame-rate is used.",
    )
    parser.add_argument(
        "--target-frame-rate",
        type=float,
        default=None,
        help="Down-sample the video to this FPS before tracking (drop frames).",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep decoded frames on CPU memory instead of GPU.",
    )
    parser.add_argument(
        "--offload-masks-to-disk",
        action="store_true",
        help="Persist per-frame masks to disk (tracking_results/_mask_cache) instead of RAM.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory to store (or reuse) extracted JPEG frames for video inputs.",
    )
    parser.add_argument(
        "--prompt-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO score threshold.",
    )
    parser.add_argument(
        "--prompt-text-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO text threshold.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Force computation device. 'auto' selects CUDA->MPS->CPU.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper structures
# -----------------------------------------------------------------------------


class VideoFrames:
    """Handles frame extraction/down-sampling for video files and directories."""

    def __init__(self, source: str, frames_dir: Optional[str] = None):
        self.source = source
        self.frame_ids: List[str] = []
        self.temp_dirs: List[str] = []
        self.original_kind: str
        self.source_fps: Optional[float] = None

        if os.path.isdir(source):
            self.original_kind = "folder"
            self.frame_ids = [
                p for p in os.listdir(source)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            if not self.frame_ids:
                raise RuntimeError(f"No images found in {source}")
            try:
                self.frame_ids.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                self.frame_ids.sort()
        elif os.path.isfile(source):
            if decord is None:
                raise ImportError(
                    "Decoding video inputs requires the 'decord' package. Install with `pip install decord`."
                )
            ext = os.path.splitext(source)[-1].lower()
            if ext not in {".mp4", ".mov", ".m4v"}:
                raise ValueError(f"Unsupported video extension '{ext}'.")

            target_dir = Path(frames_dir) if frames_dir else Path(tempfile.mkdtemp(prefix="sam2_frames_"))
            if frames_dir is None:
                self.temp_dirs.append(str(target_dir))
            target_dir.mkdir(parents=True, exist_ok=True)

            decord.bridge.set_bridge("native")
            reader = decord.VideoReader(source)
            if len(reader) == 0:
                raise RuntimeError(f"No frames decoded from video: {source}")

            try:
                self.source_fps = float(reader.get_avg_fps())
            except Exception:
                self.source_fps = None

            existing = sorted([p.name for p in target_dir.glob("*.jpg")])
            if not existing:
                print(f"[Iterative Tracking] Extracting frames from {source} to {target_dir}")
                for idx, frame in enumerate(reader):
                    frame_np = frame.asnumpy() if hasattr(frame, "asnumpy") else frame.numpy()
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    frame_path = target_dir / f"{idx:05d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame_bgr):
                        raise RuntimeError(f"Failed to save frame {frame_path}")
                existing = sorted([p.name for p in target_dir.glob("*.jpg")])
                print(f"[Iterative Tracking] Extracted {len(existing)} frames.")
            else:
                print(f"[Iterative Tracking] Reusing {len(existing)} extracted frames in {target_dir}")

            self.source = str(target_dir)
            self.frame_ids = existing
            self.original_kind = "video_file"
        else:
            raise FileNotFoundError(f"Video source not found: {source}")

    def limit_frames(self, max_frames: Optional[int]) -> None:
        if max_frames is None or max_frames <= 0:
            return
        if len(self.frame_ids) <= max_frames:
            return
        selected = self.frame_ids[:max_frames]
        subset_dir = Path(tempfile.mkdtemp(prefix="sam2_subset_"))
        self.temp_dirs.append(str(subset_dir))
        subset_dir.mkdir(parents=True, exist_ok=True)
        for name in selected:
            shutil.copy2(Path(self.source) / name, subset_dir / name)
        self.source = str(subset_dir)
        self.frame_ids = list(selected)

    def downsample_to_fps(self, target_fps: Optional[float], source_fps_override: Optional[float]) -> None:
        if target_fps is None or target_fps <= 0:
            return
        src_fps = self.source_fps if self.source_fps is not None else source_fps_override
        if src_fps is None:
            raise ValueError(
                "target-frame-rate provided, but source FPS is unknown. "
                "Either supply --source-fps or use a video input with detectable FPS."
            )
        if target_fps >= src_fps:
            return
        stride = max(1, int(round(src_fps / target_fps)))
        selected = [name for idx, name in enumerate(self.frame_ids) if idx % stride == 0]
        if not selected:
            selected = [self.frame_ids[0]]
        subset_dir = Path(tempfile.mkdtemp(prefix="sam2_downsampled_"))
        self.temp_dirs.append(str(subset_dir))
        for name in selected:
            shutil.copy2(Path(self.source) / name, subset_dir / name)
        self.source = str(subset_dir)
        self.frame_ids = list(selected)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def get_rgb(self, index: int) -> np.ndarray:
        path = os.path.join(self.source, self.frame_ids[index])
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_bgr(self, index: int) -> np.ndarray:
        path = os.path.join(self.source, self.frame_ids[index])
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame {path}")
        return bgr

    def cleanup(self) -> None:
        for temp_path in self.temp_dirs:
            shutil.rmtree(temp_path, ignore_errors=True)

    def __del__(self):
        self.cleanup()


def resize_longest_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max_edge is None or max_edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.BILINEAR)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


@dataclass
class ObjectState:
    mask: np.ndarray
    frame_idx: int
    class_name: str


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    video_frames = VideoFrames(args.video_source, frames_dir=args.frames_dir)
    video_frames.downsample_to_fps(args.target_frame_rate, args.source_fps)
    video_frames.limit_frames(args.max_frames)

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model_cfg, sam2_checkpoint = {
        "sam2.1_hiera_large": (
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "./checkpoints/sam2.1_hiera_large.pt",
        ),
        "sam2.1_hiera_base_plus": (
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "./checkpoints/sam2.1_hiera_base_plus.pt",
        ),
    }[args.sam2_variant]

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    if args.fp16 and device in {"cuda", "mps"}:
        sam2_image_model = sam2_image_model.half()
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    dino_id = {
        "large": "rziga/mm_grounding_dino_large_all",
        "tiny": "IDEA-Research/grounding-dino-tiny",
    }[args.dino_model]
    processor = AutoProcessor.from_pretrained(dino_id)
    dino_kwargs = {}
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_id
    ).to(device)

    autocast_kwargs = None
    if args.fp16 and device in {"cuda", "mps"}:
        autocast_device = "cuda" if device == "cuda" else "mps"
        autocast_kwargs = {"device_type": autocast_device, "dtype": torch.float16}

    inference_state = video_predictor.init_state(
        video_path=video_frames.source,
        offload_video_to_cpu=args.offload_video_to_cpu,
    )

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    mask_cache_dir = os.path.join(save_dir, "_mask_cache")
    os.makedirs(mask_cache_dir, exist_ok=True)

    mask_index: List[str] = []
    in_memory_masks: Dict[int, Dict[int, np.ndarray]] = {}
    class_name_lookup: Dict[int, str] = {}
    global_objects: Dict[int, ObjectState] = {}
    next_object_id = 0

    total_frames = len(video_frames)

    try:
        for start_idx in range(0, total_frames, args.detection_interval):
            frame_rgb = video_frames.get_rgb(start_idx)
            image = Image.fromarray(frame_rgb)
            image_for_dino = resize_longest_edge(image, args.max_dino_long_edge)

            raw_inputs = processor(
                images=image_for_dino,
                text=args.text,
                return_tensors="pt",
            )
            inputs = {}
            for key, value in raw_inputs.items():
                if isinstance(value, torch.Tensor):
                    tensor = value.to(device)
                    inputs[key] = tensor
                else:
                    inputs[key] = value
            if autocast_kwargs is not None:
                print(f"[FP16 Debug] Grounding DINO autocast kwargs: {autocast_kwargs}")
            with torch.no_grad():
                if autocast_kwargs is not None:
                    with torch.autocast(**autocast_kwargs):
                        outputs = grounding_model(**inputs)
                else:
                    outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=args.prompt_threshold,
                text_threshold=args.prompt_text_threshold,
                target_sizes=[image.size[::-1]],
            )

            detections = []
            num_candidates = len(results[0]["boxes"])
            print(f"[Iterative Tracking] DINO detections at frame {start_idx:05d}: {num_candidates}")

            if num_candidates > 0:
                image_predictor.set_image(np.array(image))
                boxes = results[0]["boxes"].cpu().numpy()
                labels = results[0]["labels"]
                masks, _, _ = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False,
                )
                if masks.ndim == 2:
                    masks = masks[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)
                for mask_np, class_name in zip(masks, labels):
                    mask_np = np.asarray(mask_np, dtype=bool)
                    if mask_np.ndim == 3:
                        mask_np = np.squeeze(mask_np, axis=0)
                    best_id = None
                    best_iou = 0.0
                    for obj_id, state in global_objects.items():
                        iou = mask_iou(mask_np, state.mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = obj_id
                    if best_id is not None and best_iou >= args.track_iou_threshold:
                        obj_id = best_id
                        print(f"[Iterative Tracking] Reusing object {obj_id} at frame {start_idx:05d} (IoU={best_iou:.3f})")
                    else:
                        obj_id = next_object_id
                        print(f"[Iterative Tracking] New object {obj_id} detected at frame {start_idx:05d}")
                        next_object_id += 1
                    global_objects[obj_id] = ObjectState(mask=mask_np, frame_idx=start_idx, class_name=class_name)
                    class_name_lookup[obj_id] = class_name
                    detections.append((obj_id, mask_np))

            # If no detections (new) and we already have objects, keep tracking using last masks.
            if not detections and global_objects:
                detections = [
                    (obj_id, state.mask)
                    for obj_id, state in global_objects.items()
                ]

            if not detections:
                print(f"[Iterative Tracking] No detections at frame {start_idx:05d}; skipping chunk.")
                continue

            video_predictor.reset_state(inference_state)
            for obj_id, mask_np in detections:
                video_predictor.add_new_mask(
                    inference_state,
                    frame_idx=start_idx,
                    obj_id=obj_id,
                    mask=mask_np,
                )

            chunk_masks: Dict[int, Dict[int, np.ndarray]] = {}
            end_idx = min(start_idx + args.detection_interval, total_frames)
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                if out_frame_idx < start_idx:
                    continue
                if out_frame_idx >= end_idx:
                    break
                frame_dict: Dict[int, np.ndarray] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_np = (out_mask_logits[i] > 0.0).cpu().numpy()
                    if mask_np.ndim > 2:
                        mask_np = np.squeeze(mask_np, axis=0)
                    mask_np = mask_np.astype(bool)
                    frame_dict[int(out_obj_id)] = mask_np
                    class_name_lookup.setdefault(int(out_obj_id), args.text.strip())
                    global_objects[int(out_obj_id)] = ObjectState(
                        mask=mask_np,
                        frame_idx=out_frame_idx,
                        class_name=class_name_lookup[int(out_obj_id)],
                    )
                chunk_masks[out_frame_idx] = frame_dict

            for frame_idx, masks_dict in chunk_masks.items():
                if args.offload_masks_to_disk:
                    cache_path = os.path.join(mask_cache_dir, f"frame_{frame_idx:05d}.npz")
                    np.savez_compressed(
                        cache_path,
                        object_ids=np.array(list(masks_dict.keys()), dtype=np.int32),
                        masks=np.stack(list(masks_dict.values()), axis=0),
                    )
                    mask_index.append(cache_path)
                else:
                    in_memory_masks[frame_idx] = masks_dict
                    mask_index.append(frame_idx)

        if args.offload_masks_to_disk:
            with open(os.path.join(mask_cache_dir, "index.json"), "w", encoding="utf-8") as f:
                json.dump(mask_index, f, indent=2)
        else:
            mask_index.sort()

        written_frames: List[int] = []
        skipped_frames: List[int] = []

        def load_masks(entry):
            if args.offload_masks_to_disk:
                data = np.load(entry)
                object_ids = data["object_ids"]
                masks = data["masks"]
                frame_idx_local = int(os.path.splitext(os.path.basename(entry))[0].split("_")[-1])
            else:
                frame_idx_local = entry
                masks_dict = in_memory_masks[frame_idx_local]
                object_ids = np.array(list(masks_dict.keys()), dtype=np.int32)
                masks = np.stack(list(masks_dict.values()), axis=0)
            return frame_idx_local, object_ids, masks

        for idx, entry in enumerate(mask_index):
            frame_idx, object_ids, masks = load_masks(entry)
            img = video_frames.get_bgr(frame_idx)
            if len(object_ids) == 0:
                skipped_frames.append(frame_idx)
                continue

            masks_bool = masks.astype(bool)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks_bool),
                mask=masks_bool,
                class_id=object_ids,
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            labels = [
                f"{class_name_lookup.get(int(obj_id), str(int(obj_id)))}"
                for obj_id in object_ids
            ]
            annotated_frame = label_annotator.annotate(
                annotated_frame,
                detections=detections,
                labels=labels,
            )
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            output_path = os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg")
            if cv2.imwrite(output_path, annotated_frame):
                written_frames.append(frame_idx)
            else:
                skipped_frames.append(frame_idx)

        print(
            f"[Iterative Tracking] Annotated {len(written_frames)} frames "
            f"({len(mask_index)} cached)."
        )
        if skipped_frames:
            print(
                "[Iterative Tracking] Skipped frames: "
                + ", ".join(f"{idx:05d}" for idx in skipped_frames)
            )

        if args.output_video:
            output_video_path = args.output_video
        else:
            stem = Path(args.video_source.rstrip(os.sep)).stem
            output_video_path = f"./grounded_sam2_tracking_iterative_{stem}.mp4"

        create_video_from_images(save_dir, output_video_path, frame_rate=args.frame_rate)

    finally:
        video_frames.cleanup()


if __name__ == "__main__":
    main(parse_args())
