import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Hyper parameters
"""
TEXT_PROMPT = "trees. electricity pole. wire. person. snow."
IMG_PATH = "notebooks/images/snowy_line.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True
MULTIMASK_OUTPUT = True
BOX_SHRINK_RATIO = 1.0
MORPH_KERNEL_SIZE = 3


def shrink_boxes_xyxy(boxes: np.ndarray, ratio: float, img_width: int, img_height: int) -> np.ndarray:
    if boxes.size == 0 or not (0 < ratio < 1):
        return boxes
    shrunk = boxes.copy()
    widths = shrunk[:, 2] - shrunk[:, 0]
    heights = shrunk[:, 3] - shrunk[:, 1]
    centers_x = shrunk[:, 0] + widths * 0.5
    centers_y = shrunk[:, 1] + heights * 0.5
    half_widths = widths * ratio * 0.5
    half_heights = heights * ratio * 0.5
    shrunk[:, 0] = centers_x - half_widths
    shrunk[:, 2] = centers_x + half_widths
    shrunk[:, 1] = centers_y - half_heights
    shrunk[:, 3] = centers_y + half_heights
    shrunk[:, [0, 2]] = np.clip(shrunk[:, [0, 2]], 0, img_width)
    shrunk[:, [1, 3]] = np.clip(shrunk[:, [1, 3]], 0, img_height)
    return shrunk


def apply_morphological_opening(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if masks.size == 0 or kernel_size <= 1:
        return masks
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    processed_masks = []
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8)
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        processed_masks.append(opened.astype(bool))
    return np.stack(processed_masks, axis=0)


def ensure_masks_batch_first(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(f"Unsupported mask shape {arr.shape} for 4D masks")
    if arr.ndim == 3 and arr.shape[-1] == 1 and arr.shape[0] != 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Cannot normalize mask array with shape {arr.shape}")
    return arr

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
input_boxes = shrink_boxes_xyxy(input_boxes, BOX_SHRINK_RATIO, w, h)

if input_boxes.size == 0:
    raise RuntimeError(
        "Grounding DINO did not return any bounding boxes. "
        "Consider relaxing box/text thresholds or revising the text prompt."
    )

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=MULTIMASK_OUTPUT,
)

"""
Sample the best mask according to the score
"""
if MULTIMASK_OUTPUT:
    axis = 1 if scores.ndim > 1 else 0
    best = np.argmax(scores, axis=axis)
    if masks.ndim == 4:
        masks = masks[np.arange(masks.shape[0]), best]
    else:
        masks = masks[best]

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

masks = apply_morphological_opening(masks, MORPH_KERNEL_SIZE)
masks = ensure_masks_batch_first(masks)

confidences = confidences.numpy().tolist()
class_names = labels

class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    normalized = ensure_masks_batch_first(mask)
    if normalized.shape[0] != 1:
        raise ValueError(
            f"mask must represent exactly one instance, but got shape {normalized.shape}"
        )
    mask_2d = normalized[0]
    rle = mask_util.encode(np.array(mask_2d[np.newaxis, ...], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
