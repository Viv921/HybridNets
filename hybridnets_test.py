import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
from torch.nn import functional as F

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
parser.add_argument('--speed_test', type=boolean_string, default=False,
                    help='Measure inference latency')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (reduce if OOM)')
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')
color_list_seg = {}
# Set specific colors for road (purple) and lane (yellow) in BGR format
for seg_class in params.seg_list:
    if seg_class == 'road':
        color_list_seg[seg_class] = [128, 0, 128]  # Purple in BGR
    elif seg_class == 'lane':
        color_list_seg[seg_class] = [0, 255, 255]  # Yellow in BGR (B=0, G=255, R=255)
    else:
        # For any other classes, use a default color or random
        color_list_seg[seg_class] = [0, 0, 0]  # Black in BGR
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights
img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
# img_path = [img_path[0]]  # demo with 1 image
input_imgs = []
shapes = []
det_only_imgs = []
# Get base filenames without extension for output naming
base_names = [os.path.splitext(os.path.basename(p))[0] for p in img_path]

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales

threshold = args.conf_thresh
iou_threshold = args.iou_thresh
imshow = args.imshow
imwrite = args.imwrite
show_det = args.show_det
show_seg = args.show_seg
os.makedirs(output, exist_ok=True)

use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list

color_list = standard_to_bgr(STANDARD_COLORS)
#ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]
#ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
#print(f"FOUND {len(ori_imgs)} IMAGES")
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
#resized_shape = params.model['image_size']ori_imgs = []
# We need to keep track of valid image paths for the base_names later
#########################################################################
ori_imgs = []
valid_img_paths = [] 

print("=" * 60)
print("STEP 1/4: Loading images from disk...")
print("=" * 60)
for idx, i in enumerate(img_path, 1):
    print(f"Loading image {idx}/{len(img_path)}: {os.path.basename(i)}", end='\r')
    img = cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    
    if img is None:
        print(f"\nWARNING: Could not read image {i}. Skipping file.")
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_imgs.append(img)
    valid_img_paths.append(i)

# IMPORTANT: Update the base_names to only use the files we successfully read
base_names = [os.path.splitext(os.path.basename(p))[0] for p in valid_img_paths]

print(f"\nSuccessfully loaded {len(ori_imgs)}/{len(img_path)} images")
print()
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
resized_shape = params.model['image_size']
#############################################################################
if isinstance(resized_shape, list):
    resized_shape = max(resized_shape)
normalize = transforms.Normalize(
    mean=params.mean, std=params.std
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
print("=" * 60)
print("STEP 2/4: Preprocessing images...")
print("=" * 60)
for idx, ori_img in enumerate(ori_imgs, 1):
    print(f"Preprocessing image {idx}/{len(ori_imgs)}", end='\r')
    h0, w0 = ori_img.shape[:2]  # orig hw
    # Force resize to the exact target shape
    input_img = cv2.resize(ori_img, (640, 384), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]

    # No need for letterbox since we're forcing exact dimensions
    ratio = (h / h0, w / w0)
    pad = (0, 0)

    input_imgs.append(input_img)
    # cv2.imwrite('input.jpg', input_img * 255)
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling
print(f"\nCompleted preprocessing {len(ori_imgs)} images")
print()

# Don't load all images to GPU at once - process in batches instead
batch_size = args.batch_size
print(f"Will process images in batches of {batch_size}")
print()

# Prepare model first before processing batches
use_cuda = args.cuda
use_float16 = args.float16

checkpoint = torch.load(weight, map_location='cuda' if use_cuda else 'cpu', weights_only = False)
weight = checkpoint['model']
#new_weight = OrderedDict((k[6:], v) for k, v in weight['model'].items())
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
model.load_state_dict(weight)

model.requires_grad_(False)
model.eval()

if use_cuda:
    device = torch.device("cuda:0")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using all {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    if use_float16:
        model = model.half()

print("=" * 60)
print("STEP 3/4: Running model inference and post-processing...")
print("=" * 60)

# Process images in batches to avoid OOM
all_results = []
num_batches = (len(input_imgs) + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(input_imgs))
    batch_input_imgs = input_imgs[start_idx:end_idx]
    batch_shapes = shapes[start_idx:end_idx]
    batch_ori_imgs = ori_imgs[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx})")
    
    # Prepare batch
    if use_cuda:
        x = torch.stack([transform(fi).to(device) for fi in batch_input_imgs], 0)
    else:
        x = torch.stack([transform(fi) for fi in batch_input_imgs], 0)
    
    x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
    
    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)
        print(f"Model inference completed for batch {batch_idx + 1}")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Regression shape: {regression.shape}")
        print(f"  - Classification shape: {classification.shape}")
        print(f"  - Anchors shape (before fix): {anchors.shape}")
        print(f"  - Segmentation shape: {seg.shape}")
        
        # When using DataParallel with multiple GPUs, anchors get duplicated per GPU
        # We need to expand anchors to match the batch size
        # anchors shape is [num_gpus, num_anchors, 4], we need [batch_size, num_anchors, 4]
        if anchors.shape[0] != x.shape[0]:
            # Each GPU produces one set of anchors, but we need one per batch item
            # Simply take the first set and replicate it for all batch items (anchors are the same for all images)
            anchors = anchors[0:1].expand(x.shape[0], -1, -1).contiguous()
            print(f"  - Anchors shape (after fix): {anchors.shape}")

        # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
        seg_mask_list = []
        # (B, C, W, H) -> (B, W, H)
        if seg_mode == BINARY_MODE:
            seg_mask = torch.where(seg >= 0, 1, 0)
            seg_mask.squeeze_(1)
            seg_mask_list.append(seg_mask)
        elif seg_mode == MULTICLASS_MODE:
            _, seg_mask = torch.max(seg, 1)
            seg_mask_list.append(seg_mask)
        else:
            seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
            seg_mask_list.pop(0)
        
        print(f"Processing segmentation masks for batch {batch_idx + 1}...")
        # (B, W, H) -> (W, H)
        for i in range(seg.size(0)):
            global_idx = start_idx + i
            print(f"Processing segmentation for image {global_idx+1}/{len(ori_imgs)}", end='\r')
            
            # Add det_only_imgs once per image, not per segmentation class
            det_only_imgs.append(batch_ori_imgs[i].copy())
            
            for seg_class_index, seg_mask in enumerate(seg_mask_list):
                seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                pad_h = int(batch_shapes[i][1][1][1])
                pad_w = int(batch_shapes[i][1][1][0])
                seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
                seg_mask_ = cv2.resize(seg_mask_, dsize=batch_shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
                color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
                color_seg = color_seg[..., ::-1]  # RGB -> BGR

                color_mask = np.mean(color_seg, 2)
                seg_img = batch_ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else batch_ori_imgs[i]
                seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                seg_img = seg_img.astype(np.uint8)
                seg_filename = f'{output}/{base_names[global_idx]}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                               f'{output}/{base_names[global_idx]}_seg.jpg'
                if show_seg or seg_mode == MULTILABEL_MODE:
                    cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        
        print(f"\nCompleted segmentation processing for batch {batch_idx + 1}")

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        
        all_results.extend(out)
    
    # Clear GPU cache after each batch - be more aggressive
    if use_cuda:
        del x, features, regression, classification, anchors, seg, out, seg_mask_list
        if 'seg_mask' in locals():
            del seg_mask
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        print(f"  - GPU memory cleared for batch {batch_idx + 1}")

print("\n" + "=" * 60)
print("STEP 4/4: Processing detections and saving results...")
print("=" * 60)
for i in range(len(ori_imgs)):
    print(f"Processing detections for image {i+1}/{len(ori_imgs)}", end='\r')
    all_results[i]['rois'] = scale_coords(ori_imgs[i][:2], all_results[i]['rois'], shapes[i][0], shapes[i][1])
    for j in range(len(all_results[i]['rois'])):
        x1, y1, x2, y2 = all_results[i]['rois'][j].astype(int)
        obj = obj_list[all_results[i]['class_ids'][j]]
        score = float(all_results[i]['scores'][j])
        plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                     color=color_list[get_index_label(obj, obj_list)])
        if show_det:
            plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

    if show_det:
        cv2.imwrite(f'{output}/{base_names[i]}_det.jpg',  cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))

    if imshow:
        cv2.imshow('img', ori_imgs[i])
        cv2.waitKey(0)

    if imwrite:
        cv2.imwrite(f'{output}/{base_names[i]}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))
    
    print(f"\nCompleted processing and saved {len(ori_imgs)} images to '{output}'")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"Total images processed: {len(ori_imgs)}")
print(f"Output directory: {output}")
print("=" * 60)
print()

if not args.speed_test:
    exit(0)
print('running speed test...')

# Prepare a single batch for speed test
if use_cuda:
    x_test = torch.stack([transform(input_imgs[0]).to(device)], 0)
else:
    x_test = torch.stack([transform(input_imgs[0])], 0)
x_test = x_test.to(torch.float16 if use_cuda and use_float16 else torch.float32)

with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring 1 image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x_test)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x_test,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    print('test2: model inferring only')
    print('inferring images for batch_size 32 for 10 times...')
    t1 = time.time()
    x_test_batch = torch.cat([x_test] * 32, 0)
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x_test_batch)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
