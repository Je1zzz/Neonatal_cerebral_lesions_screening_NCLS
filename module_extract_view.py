import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
import os
import sys
import pydicom
import shutil
import json
from collections import namedtuple
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rt_detr.core import YAMLConfig
import time
from datetime import timedelta

# Anatomy configuration
neonatal_cranium_category2name = {
    4: 'COR Corpus Callosum',
    5: 'COR Anterior Horn of Lateral Ventricle',
    6: 'COR Third Ventricle',
    7: 'COR Lateral Ventricle',
    8: 'COR Thalamus and Basal Ganglia',
    9: 'SAG Corpus Callosum',
    10: 'SAG Trigeminal Nerve Area',
    11: 'SAG Lateral Ventricle',
    12: 'SAG Cerebellum',
    13: 'SAG Thalamus and Basal Ganglia',
}

neonatal_cranium_category2label = {k: i for i, k in enumerate(neonatal_cranium_category2name.keys())}
neonatal_cranium_label2category = {v: k for k, v in neonatal_cranium_category2label.items()}

# Define a named tuple for candidate frames
CandidateFrame = namedtuple('CandidateFrame', ['frame_idx', 'view_type', 'score', 'file_path', 'total_frames'])

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def save_excel_data(excel_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(excel_data, f, ensure_ascii=False, indent=4)
    print(f"Saved excel data to {output_file}")

# Function to load excel data from a text file (if exists)
def load_excel_data(input_file):
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            return json.load(f)
    return []

def process_frame(image_data, photometric_interpretation):
    if photometric_interpretation == 'MONOCHROME1':
        image_data = np.invert(image_data)
        image = Image.fromarray(image_data).convert('L')
    elif photometric_interpretation == 'MONOCHROME2':
        image = Image.fromarray(image_data).convert('RGB')
    elif photometric_interpretation == 'RGB':
        image = Image.fromarray(image_data).convert('RGB')
    elif photometric_interpretation in ['YBR_FULL_422', 'YBR_FULL']:
        image = Image.fromarray(image_data, 'YCbCr').convert('RGB')
    else:
        raise ValueError(f"Unsupported photometric interpretation: {photometric_interpretation}")
    return image


def filter_cor_frames(candidates):
    """Filter COR frames based on the relationship between COR2 and COR3 using mean frame sequence"""
    if not candidates:
        return candidates

    # Check if we have enough frames in each view
    cor3_frames = candidates.get("COR3", [])
    cor2_frames = candidates.get("COR2", [])
    cor1_frames = candidates.get("COR1", [])

    if (len(cor3_frames) < 5 or len(cor2_frames) == 0 or len(cor1_frames) == 0):
        return candidates

    # Calculate mean frame sequence for COR2 and COR3
    mean_cor2 = np.mean([c.frame_idx for c in cor2_frames])
    mean_cor3 = np.mean([c.frame_idx for c in cor3_frames])

    if mean_cor2 < mean_cor3:
        # COR1 → COR2 → COR3 order
        # Find max frame in COR2
        max_cor2_frame = max(c.frame_idx for c in cor2_frames)
        # Remove COR1 frames that have higher frame numbers than COR2 max frame
        candidates["COR1"] = [c for c in cor1_frames if c.frame_idx < max_cor2_frame]
    else:
        # COR3 → COR2 → COR1 order
        # Find min frame in COR2
        min_cor2_frame = min(c.frame_idx for c in cor2_frames)
        # Remove COR1 frames that have lower frame numbers than COR2 min frame
        candidates["COR1"] = [c for c in cor1_frames if c.frame_idx > min_cor2_frame]

    return candidates

def adjust_sag_frames(candidates):
    """Adjust SAG frames by potentially converting some SAG2 to SAG3"""
    if not candidates:
        return candidates

    sag1_frames = candidates.get("SAG1", [])
    sag2_frames = candidates.get("SAG2", [])

    if (len(sag1_frames) < 5 or len(sag2_frames) < 5 or
        not any(s.frame_idx > max(s1.frame_idx for s1 in sag1_frames) for s in sag2_frames) or
        not any(s.frame_idx < min(s1.frame_idx for s1 in sag1_frames) for s in sag2_frames)):
        return candidates

    # Find median frame of SAG1 as split point
    sag1_median = np.median([s.frame_idx for s in sag1_frames])

    # Convert SAG2 frames that are after SAG1 median to SAG3
    new_sag2 = []
    new_sag3 = candidates.get("SAG3", [])

    for s in sag2_frames:
        if s.frame_idx > sag1_median:
            # Convert to SAG3
            new_sag3.append(CandidateFrame(
                frame_idx=s.frame_idx,
                view_type="SAG3",
                score=s.score,
                file_path=s.file_path,
                total_frames=s.total_frames
            ))
        else:
            new_sag2.append(s)

    candidates["SAG2"] = new_sag2
    candidates["SAG3"] = new_sag3

    return candidates

def evaluate_candidate_queues(all_results):
    """Evaluate and select the best COR and SAG queues from all results"""
    # Separate COR and SAG queues
    cor_queues = []
    sag_queues = []

    for result in all_results:
        if result['primary_queue'] == 'COR':
            cor_queues.append(result)
        else:
            sag_queues.append(result)

    # Function to calculate queue score
    def calculate_queue_score(candidates):
        score = 0.0
        for view in candidates:
            if candidates[view]:  # If view has frames
                top_score = max(c.score for c in candidates[view])
                score += top_score
        return score

    # Find best COR queue
    best_cor = None
    max_cor_score = -1
    for queue in cor_queues:
        current_score = calculate_queue_score(queue['candidates'])
        if current_score > max_cor_score:
            max_cor_score = current_score
            best_cor = queue

    # Find best SAG queue
    best_sag = None
    max_sag_score = -1
    for queue in sag_queues:
        current_score = calculate_queue_score(queue['candidates'])
        if current_score > max_sag_score:
            max_sag_score = current_score
            best_sag = queue

    return {
        'best_cor': best_cor,
        'best_sag': best_sag,
        'cor_score': max_cor_score,
        'sag_score': max_sag_score
    }


def initialize_candidates():
    return {"COR1": [], "COR2": [], "COR3": [], "SAG1": [], "SAG2": [], "SAG3": []}

def classify_frame(labels_list):
    possible_views = []

    if labels_list.count(6) >= 1:
        possible_views.append("COR2")
    elif labels_list.count(4) >= 1:
        possible_views.append("COR1")
    if labels_list.count(7) >= 1:
        possible_views.append("COR3")
    if (labels_list.count(9) >= 1 and labels_list.count(10) >= 1) or \
       (labels_list.count(9) >= 1 and labels_list.count(12) >= 1) or \
       (labels_list.count(10) >= 1 and labels_list.count(12) >= 1):
        possible_views.append("SAG1")
    if labels_list.count(11) >= 1 or labels_list.count(13) >= 1:
        possible_views.append("SAG2")
    if labels_list.count(12) >= 2:
        possible_views.append("SAG3")

    return possible_views

def calculate_score(labels, boxes, scores, view_type, pil_image):
    score = 0.0
    anatomy_scores = {
        "COR1": {4: 0.3, 5: 0.2, 8: 0.15},
        "COR2": {4: 0.2, 5: 0.1, 8: 0.1, 6: 0.4},
        "COR3": {7: 0.5},
        "SAG1": {9: 0.33, 10: 0.33, 12: 0.33},
        "SAG2": {11: 0.3, 13: 0.7},
        "SAG3": {12: 0.8}
    }

    anatomy_limits = {
        "COR1": {4: 1, 5: 2, 8: 2},
        "COR2": {4: 1, 5: 2, 8: 2, 6: 1},
        "COR3": {7: 2},
        "SAG1": {9: 1, 10: 1, 12: 1},
        "SAG2": {11: 1, 13: 1},
        "SAG3": {12: 2}
    }

    base_score = anatomy_scores.get(view_type, {})
    max_limit = anatomy_limits.get(view_type, {})
    label_counts = {}

    for i, label in enumerate(labels):
        confidence = scores[i].item()
        if label in base_score:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] <= max_limit.get(label, float('inf')):
                score += base_score[label] * confidence
        else:
            score -= 0.2 * confidence

    return score

def determine_primary_queue(candidates):
    """Determine if the primary queue should be COR or SAG based on frame counts"""
    cor_count = sum(len(candidates[view]) for view in ["COR1", "COR2", "COR3"])
    sag_count = sum(len(candidates[view]) for view in ["SAG1", "SAG2", "SAG3"])
    if cor_count > sag_count:
        # Remove all SAG views
        for view in ["SAG1", "SAG2", "SAG3"]:
            candidates[view] = []
        return "COR"
    else:
        # Remove all COR views
        for view in ["COR1", "COR2", "COR3"]:
            candidates[view] = []
        return "SAG"

def filter_invalid_candidates(candidates, min_frame_ratio=0.2):
    """Remove entire candidate set if total candidate frames < min_frame_ratio*total_frames"""
    if not candidates or not any(candidates.values()):
        return None  # No candidates at all

    # Get total frames from the first candidate
    total_frames = None
    for view_candidates in candidates.values():
        if view_candidates:  # 找到第一个有候选帧的视图类型
            total_frames = view_candidates[0].total_frames
            break
    min_frames_required = total_frames * min_frame_ratio

    # Calculate total candidate frames across all view types
    total_candidate_frames = sum(len(view_candidates) for view_candidates in candidates.values())

    if total_candidate_frames >= min_frames_required:
        return candidates  # Keep all candidates
    else:
        print(f"Removing entire candidate set: only {total_candidate_frames} candidate frames "
              f"(needs at least {min_frames_required:.0f})")
        return None  # Remove all candidates for this file

def process_mp4_file(video_path, model, transforms, device, thrh):
    try:
        start_time = time.time()
        print(f"\n=== Processing {video_path} ===")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return None

        # Get video properties
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames <= 1:
            return None

        candidates = initialize_candidates()
        print(f"Total frames to process: {num_frames}")

        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:  # Print progress every 10 frames
                print(f"Reading frame {frame_idx}/{num_frames} ({(frame_idx/num_frames*100):.1f}%)", end='\r')

            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame...
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            w, h = pil_frame.size
            orig_size = torch.tensor([[w, h]]).to(device)
            im_data = transforms(pil_frame)[None].to(device)
            output = model(im_data, orig_size)

            labels, boxes, scores = output
            lab = labels[scores > thrh]
            box = boxes[scores > thrh]
            scrs = scores[scores > thrh]
            labels_list = [neonatal_cranium_label2category.get(label.item(), -1)
                    for label in lab]
            labels_list = [label for label in labels_list if label != -1]

            if not labels_list:
                continue

            possible_views = classify_frame(labels_list)

            for view in possible_views:
                score = calculate_score(labels_list, box, scrs, view, pil_frame)
                candidate = CandidateFrame(
                    frame_idx=frame_idx,
                    view_type=view,
                    score=score,
                    file_path=video_path,
                    total_frames=num_frames
                )
                candidates[view].append(candidate)

        cap.release()
        print("\nFrame processing completed")

        # Filter invalid candidates
        print("Filtering invalid candidates...")
        candidates = filter_invalid_candidates(candidates)
        if candidates is None:
            print("No valid candidates found")
            return None

        # Determine primary queue
        print("Determining primary queue...")
        primary_queue = determine_primary_queue(candidates)
        print(f"Selected primary queue: {primary_queue}")

        print("Filtering frames...")
        if primary_queue == 'COR':
            candidates = filter_cor_frames(candidates)
        else:  # SAG
            candidates = adjust_sag_frames(candidates)

        processing_time = time.time() - start_time
        print(f"Processing completed in {timedelta(seconds=int(processing_time))}")

        return {
            'file_path': video_path,
            'candidates': candidates,
            'primary_queue': primary_queue,
            'total_frames': num_frames
        }

    except Exception as e:
        print(f"Error processing video file {video_path}: {str(e)}")
        return None

def save_best_frames(best_queues, output_dir, id, date):
    """
    Save the best frames from video as PNG images and copy the original video file
    """
    print("\n=== Saving best frames ===")
    case_output_dir = os.path.join(output_dir,'StandardViews', id, date)
    os.makedirs(case_output_dir, exist_ok=True)
    print(f"Output directory: {case_output_dir}")

    def save_view_frames(queue, queue_type):
        if not queue:
            return

        print(f"\nProcessing {queue_type} queue...")
        video_opened = False
        cap = None

        for view_type, frames in queue['candidates'].items():
            if not frames:
                continue

            # Find frame with highest score
            best_frame = max(frames, key=lambda x: x.score)

            if not video_opened:
                # Open video for frame extraction
                cap = cv2.VideoCapture(best_frame.file_path)
                video_opened = True

            try:
                # Extract the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame.frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_path = os.path.join(case_output_dir, f"{view_type}.png")
                    img.save(img_path)
                    print(f"Saved {view_type} frame (score: {best_frame.score:.3f}) to: {img_path}")

            except Exception as e:
                print(f"Error saving {queue_type} {view_type} frame: {str(e)}")

        if cap is not None:
            cap.release()

    # Save COR and SAG queue frames
    save_view_frames(best_queues['best_cor'], "COR")
    save_view_frames(best_queues['best_sag'], "SAG")

def extract_views(args):
    total_start_time = time.time()
    print("\n=== Starting view extraction ===")
    print(f"Input directory: {args.dicom_dir}")
    print(f"Output directory: {args.output_dir}")

    # Dictionary to store processing times for each case
    case_processing_times = {}

    print("Loading model...")

    cfg = YAMLConfig(args.cfg_detection, resume=args.weight_detection)
    checkpoint = torch.load(args.weight_detection, map_location=args.device)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(args.device)

    print("Warming up model...")
    blank_image = torch.zeros((1, 3, 640, 640)).to(args.device)
    blank_size = torch.tensor([[640, 640]]).to(args.device)
    _ = model(blank_image, blank_size)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    total_videos = 0
    processed_videos = 0

    # First count total videos
    for root, _, files in os.walk(args.dicom_dir):
        total_videos += len([f for f in files if f.lower().endswith('.mp4')])

    print(f"\nFound {total_videos} video files to process")

    for root, _, files in os.walk(args.dicom_dir):
        video_files = [f for f in files if f.lower().endswith('.mp4')]

        if video_files:
            case_start_time = time.time()
            all_results = []
            id, date = os.path.basename(os.path.dirname(root)), os.path.basename(root)
            case_id = f"{id}_{date}"
            print(f"\nProcessing case ID: {case_id}")

            for file in video_files:
                processed_videos += 1
                print(f"\nProcessing video {processed_videos}/{total_videos}")

                file_path = os.path.join(root, file)
                result = process_mp4_file(file_path, model, transforms, args.device, thrh=0.6)

                if result is not None:
                    all_results.append(result)

            if all_results:
                print("\nEvaluating candidate queues...")
                best_queues = evaluate_candidate_queues(all_results)
                save_best_frames(best_queues, args.output_dir, id, date)
                all_results.clear()

                # Record processing time for this case
                case_time = time.time() - case_start_time
                case_processing_times[case_id] = case_time
                print(f"Case {case_id} processing time: {timedelta(seconds=int(case_time))}")

    total_time = time.time() - total_start_time
    print(f"\n=== Processing completed ===")
    print(f"Total processing time: {timedelta(seconds=int(total_time))}")

    # Convert timedelta objects to seconds for easier handling in the diagnosis module
    case_processing_times_seconds = {}
    for case_id, time_value in case_processing_times.items():
        case_processing_times_seconds[case_id] = time_value

    return case_processing_times_seconds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_detection', type=str, default="./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml")
    parser.add_argument('-r', '--weight_detection', type=str, default="log/detection_weight/detection_weight.pth")
    parser.add_argument('-d', '--dicom-dir', type=str, default='/data1/zhm/neonatal_cerebral_lesion/source_data',
                       help='Root dir for mp4 files(root/ID/Date/videos)')
    parser.add_argument('-o', '--output-dir', type=str, default='/data1/zhm/neonatal_cerebral_lesion/selected_data')
    parser.add_argument('--thrh', type=float, default=0.6)
    parser.add_argument('-de', '--device', type=str, default='cuda')
    args = parser.parse_args()
    extract_views(args)
