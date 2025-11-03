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

VIDEO_SOURCE = "notebooks/videos/car"  # can be a directory of JPEG frames or a video file (e.g. MP4)


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
        return self._reader[index].asnumpy()

    def get_bgr(self, index: int) -> np.ndarray:
        if self.kind == "folder":
            frame_path = os.path.join(self.source, self.frame_ids[index])
            bgr = cv2.imread(frame_path)
            if bgr is None:
                raise RuntimeError(f"Failed to read frame from {frame_path}")
            return bgr
        return cv2.cvtColor(self._reader[index].asnumpy(), cv2.COLOR_RGB2BGR)

    def get_frame_label(self, index: int) -> str:
        return self.frame_ids[index]

    def get_frame_path(self, index: int) -> Optional[str]:
        if self.kind == "folder":
            return os.path.join(self.source, self.frame_ids[index])
        return None


video_frames = VideoFrames(VIDEO_SOURCE)

model_cfg, sam2_checkpoint = SAM2_VARIANTS[SAM2_MODEL_VARIANT]

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=COMPUTE_DEVICE)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=COMPUTE_DEVICE)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "rziga/mm_grounding_dino_large_all"
device = COMPUTE_DEVICE if COMPUTE_DEVICE in {"cuda", "mps"} else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "car."

# total frames available
num_frames = len(video_frames)

# init video predictor state (supports both frame folders and video files)
inference_state = video_predictor.init_state(video_path=VIDEO_SOURCE)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
"""

# prompt grounding dino to get the box coordinates on specific frame
frame_rgb = video_frames.get_rgb(ann_frame_idx)
image = Image.fromarray(frame_rgb)

# run Grounding DINO on the image
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.25,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image))

# process the detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
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
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = "./tracking_results"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = video_frames.get_bgr(frame_idx)

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


"""
Step 6: Convert the annotated frames to video
"""

output_video_path = "./children_tracking_demo_video.mp4"
create_video_from_images(save_dir, output_video_path)
