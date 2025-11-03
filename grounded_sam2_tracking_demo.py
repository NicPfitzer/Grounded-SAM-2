import argparse
import os
import platform

if platform.system() == "Darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch
import numpy as np
import supervision as sv
from typing import List, Optional
from contextlib import nullcontext
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

try:
    import decord  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    decord = None


"""
Step 1: Environment settings and model initialization
"""


def _detect_device() -> str:
    override = os.environ.get("SAM2_DEVICE", "").lower()
    if override in {"cuda", "mps", "cpu"}:
        if override == "cuda" and torch.cuda.is_available():
            return "cuda"
        if override == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if override == "cpu":
            return "cpu"
        raise RuntimeError(f"SAM2_DEVICE override '{override}' is not supported on this machine.")

    if torch.cuda.is_available():
        try:
            _props = torch.cuda.get_device_properties(0)
            if _props is not None:
                return "cuda"
        except (AssertionError, RuntimeError):
            pass
    return "cpu"


COMPUTE_DEVICE = _detect_device()

if COMPUTE_DEVICE == "cuda":
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    amp_context.__enter__()
    props = torch.cuda.get_device_properties(0)
    if props is not None and props.major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif COMPUTE_DEVICE == "mps":
    amp_context = torch.autocast(device_type="mps", dtype=torch.float16)
    amp_context.__enter__()
else:
    amp_context = nullcontext()
    amp_context.__enter__()

# init sam image predictor and video predictor model
SAM2_VARIANTS = {
    "sam2.1_hiera_large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "./checkpoints/sam2.1_hiera_large.pt",
    ),
    "sam2.1_hiera_base_plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "./checkpoints/sam2.1_hiera_base_plus.pt",
    ),
}
SAM2_MODEL_VARIANT = "sam2.1_hiera_large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded SAM 2 tracking demo")
    parser.add_argument(
        "--video-source",
        required=True,
        help="Path to a directory of JPEG frames or a video file (MP4, MOV, M4V).",
    )
    parser.add_argument(
        "--text",
        default="car.",
        help="Text prompt passed to Grounding DINO (must be lowercase and end with a dot).",
    )
    parser.add_argument(
        "--output-dir",
        default="./tracking_results",
        help="Directory where annotated frames will be written.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional path for the output video. If not set, a name is derived from the source.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate (frames per second) for the output video.",
    )
    parser.add_argument(
        "--sam2-variant",
        default=SAM2_MODEL_VARIANT,
        choices=list(SAM2_VARIANTS.keys()),
        help="Which SAM 2 checkpoint/config variant to use.",
    )
    return parser.parse_args()


class VideoFrames:
    def __init__(self, source: str):
        self.source = source
        self._reader = None
        self.frame_ids: List[str]
        if os.path.isdir(source):
            self.kind = "folder"
            self.frame_ids = [
                p
                for p in os.listdir(source)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            if not self.frame_ids:
                raise RuntimeError(f"No JPEG frames found in directory: {source}")
            try:
                self.frame_ids.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                self.frame_ids.sort()
        elif os.path.isfile(source):
            if decord is None:
                raise ImportError(
                    "Reading video files requires the 'decord' package. "
                    "Install it with `pip install decord`."
                )
            ext = os.path.splitext(source)[-1].lower()
            supported_exts = {".mp4", ".mov", ".m4v"}
            if ext not in supported_exts:
                raise ValueError(
                    f"Unsupported video extension '{ext}'. Supported video formats: {supported_exts}"
                )
            self.kind = "video"
            decord.bridge.set_bridge("native")
            self._reader = decord.VideoReader(source)
            num_frames = len(self._reader)
            if num_frames == 0:
                raise RuntimeError(f"No frames decoded from video file: {source}")
            self.frame_ids = [f"frame_{idx:05d}.jpg" for idx in range(num_frames)]
        else:
            raise FileNotFoundError(f"Video source not found: {source}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def get_rgb(self, index: int) -> np.ndarray:
        if self.kind == "folder":
            frame_path = os.path.join(self.source, self.frame_ids[index])
            bgr = cv2.imread(frame_path)
            if bgr is None:
                raise RuntimeError(f"Failed to read frame from {frame_path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self._reader[index]
        rgb_frame = tensor.numpy()
        return rgb_frame

    def get_bgr(self, index: int) -> np.ndarray:
        if self.kind == "folder":
            frame_path = os.path.join(self.source, self.frame_ids[index])
            bgr = cv2.imread(frame_path)
            if bgr is None:
                raise RuntimeError(f"Failed to read frame from {frame_path}")
            return bgr
        tensor = self._reader[index]
        rgb_frame = tensor.numpy()
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    def get_frame_label(self, index: int) -> str:
        return self.frame_ids[index]

    def get_frame_path(self, index: int) -> Optional[str]:
        if self.kind == "folder":
            return os.path.join(self.source, self.frame_ids[index])
        return None


def main(args: argparse.Namespace) -> None:
    video_source = args.video_source
    text_prompt = args.text.strip().lower()
    if not text_prompt.endswith("."):
        text_prompt = f"{text_prompt}."

    video_frames = VideoFrames(video_source)

    model_cfg, sam2_checkpoint = SAM2_VARIANTS[args.sam2_variant]

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=COMPUTE_DEVICE)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=COMPUTE_DEVICE)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    model_id = "rziga/mm_grounding_dino_large_all"
    device = COMPUTE_DEVICE if COMPUTE_DEVICE in {"cuda", "mps"} else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inference_state = video_predictor.init_state(video_path=video_source)

    ann_frame_idx = 0

    frame_rgb = video_frames.get_rgb(ann_frame_idx)
    image = Image.fromarray(frame_rgb)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    image_predictor.set_image(np.array(image))

    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    PROMPT_TYPE_FOR_VIDEO = "box"

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    if PROMPT_TYPE_FOR_VIDEO == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask,
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    for frame_idx, segments in video_segments.items():
        img = video_frames.get_bgr(frame_idx)

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids]
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    if args.output_video:
        output_video_path = args.output_video
    else:
        video_stem = os.path.splitext(os.path.basename(video_source.rstrip(os.sep)))[0]
        if not video_stem:
            video_stem = "tracking_output"
        output_video_path = os.path.join(".", f"grounded_sam2_tracking_{video_stem}.mp4")

    create_video_from_images(save_dir, output_video_path, frame_rate=args.frame_rate)


if __name__ == "__main__":
    main(parse_args())
