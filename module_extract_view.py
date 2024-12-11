import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import cv2
import os
import time
import sys
import pydicom
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import colorsys
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

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
    # 14: 'Choroid Plexus Cyst',            
    # 15: 'Hemorrhagic Focus',             
    # 16: 'Softening Focus',               
    # 17: 'Hydrocephalus',                 
    # 18: 'Hemorrhagic Focus Liquefaction',  
}

neonatal_cranium_category2label = {k: i for i, k in enumerate(neonatal_cranium_category2name.keys())}
neonatal_cranium_label2category = {v: k for k, v in neonatal_cranium_category2label.items()}

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)    # (x1,y1,x2,y2,theta)
        return outputs

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

def extract_views(args):
    cfg = YAMLConfig(args.cfg_detection, resume=args.weight_detection)

    if args.weight_detection:
        checkpoint = torch.load(args.weight_detection, map_location=args.device)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)
    model = Model(cfg).to(args.device)

    print("Warming up the model...")
    blank_image = torch.zeros((1, 3, 640, 640)).to(args.device)
    blank_size = torch.tensor([[640, 640]]).to(args.device)
    _ = model(blank_image, blank_size)

    return process_files(model, args.dicom_dir, args.output_dir, device=args.device, thrh=0.6)

def initialize_candidates():
    return {"COR1": {}, "COR2": {}, "COR3": {}, "SAG1": {}, "SAG2": {}}

def classify_frame(labels):
    # Candidate frame selection rules
    possible_views = []
    labels_list = labels.detach().cpu().numpy().tolist()
    
    ### Coronal Plane
    # COR1 Anterior Horn View contains: 
    #   - Corpus Callosum(1),  
    #   - Anterior Horn of Lateral Ventricle(2), 
    #   - Thalamus and Basal Ganglia(2)
    # Selection rule: appear Corpus Callosum
    if labels_list.count(4) >= 1 :
        possible_views.append("COR1")
    
    # COR2 Third Ventricle View contains contains 
    #   - Corpus Callosum(1),  
    #   - Anterior Horn of Lateral Ventricle(2), 
    #   - Thalamus and Basal Ganglia(2)
    #   - Third Ventricle(1)
    # Selection rule: appear Third Ventricle
    if  labels_list.count(2) >= 1:
        possible_views.append("COR2")
    
    # COR3 Body View contains: 
    #   - Lateral Ventricle(2)
    # Selection rule: appear Lateral Ventricle
    if labels_list.count(3) >= 1:
        possible_views.append("COR3")
    
    # SAG1 Midsagittal View contains:
    #   - Corpus Callosum(1),
    #   - Trigeminal Nerve Area(1),
    #   - Cerebellum(1),
    # Selection rule: appear 2 of 3 structures above
    if (labels_list.count(5) >= 1 and labels_list.count(6) >= 1) or\
        (labels_list.count(5)>=1 and  labels_list.count(8) >= 1) or\
        (labels_list.count(6)>=1 and labels_list.count(8)>=1):
        possible_views.append("SAG1")
    
    # SAG2 Parasagittal View: 
    #   - Lateral Ventricle(1),
    #   - Thalamus and Basal Ganglia(1)
    # Selection rule: Any of the structures appear
    if labels_list.count(7) >= 1 or labels_list.count(9) >= 1:
        possible_views.append("SAG2")
    
    return possible_views


def calculate_score(labels, boxes, scores, view_type, pil_image):
    score = 0.0
    log = []

    # Base score
    anatomy_scores = {
        "COR1": {0: 0.3, 1: 0.2, 4: 0.15},
        "COR2": {0: 0.2, 1: 0.1, 4: 0.1, 2: 0.4},
        "COR3": {3: 0.5},
        "SAG1": {5: 0.33, 6: 0.33, 8: 0.33},  
        "SAG2": {7: 0.3, 9: 0.7}
    }

    # Maximum number of each anatomy structure allowed in each view
    anatomy_limits = {
        "COR1": {0: 1, 1: 2, 4: 2},
        "COR2": {0: 1, 1: 2, 4: 2, 2: 1},
        "COR3": {3: 2},
        "SAG1": {5: 1, 6: 1, 8: 1},
        "SAG2": {7: 1, 9: 1}
    }

    base_score = anatomy_scores.get(view_type, {})
    max_limit = anatomy_limits.get(view_type, {})
    center_points = {}
    label_counts = {}

    # Iterate through detected labels and adjust the score
    for i, label in enumerate(labels.detach().cpu().numpy()):
        confidence = scores[i].item()
        box = boxes[i].detach().cpu().numpy()
        x1, y1, x2, y2, theta = box
        theta = (theta - 0.5) * np.pi
        w = x2 - x1
        h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if label not in center_points:
            center_points[label] = []
        center_points[label].append((center_x, center_y, theta))
        
        if label in base_score:
            # Check if we've reached the limit for this label
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] <= max_limit.get(label, float('inf')):
                # Add score based on confidence and the relevant anatomy score
                score += base_score[label] * confidence
                score_change = base_score[label] * confidence
                log.append(f"Detect {label},confidence {confidence:.4f},increase {score_change:.4f},current scores {score:.4f}")
        else:
            # Subtract score for anatomy that should not be present
            score -= 0.2 * confidence
            score_change = 0.2 * confidence
            log.append(f"Detect {label},decrease {score_change:.4f},current scores {score:.4f}")
    
    # Extra score calculation
    if view_type == "COR2":
        if 2 in center_points and 4 in center_points:
            third_ventricle_center = center_points[2][0]
            thalamus_centers = [center[0] for center in center_points[4]]
            if len(thalamus_centers) >= 2:
                min_thalamus = min(thalamus_centers)
                max_thalamus = max(thalamus_centers)
                if not (min_thalamus < third_ventricle_center[0] < max_thalamus):
                    score -= 0.2    
                    log.append(f"Third ventricle not between two thalamic,decrease 0.2,current scores {score:.4f}")

        left_region = pil_image.crop((x1, y1, x1 + w // 3, y2))
        right_region = pil_image.crop((x1 + 2 * w // 3, y1, x2, y2))
        center_region = pil_image.crop((x1 + w // 3, y1, x1 + 2 * w // 3, y2))
        left_pixels = np.mean(np.array(left_region))
        right_pixels = np.mean(np.array(right_region))
        center_pixels = np.mean(np.array(center_region))
        if not center_pixels < min(left_pixels, right_pixels):
            score -= 0.2 
            log.append(f"Pixel values in the center of the third ventricle are higher than on both sides,decrease 0.2,current scores {score:.4f}")

    return score , log

def filter_frame_scores(frame_scores,log_recore):
    # Determine the plane type based on the number of frames (COR or SAG)
    cor_count = sum(len(frame_scores.get(f'COR{i}', {})) for i in range(1, 4))
    sag_count = sum(len(frame_scores.get(f'SAG{i}', {})) for i in range(1, 3))
    plane_type = None
    if cor_count > sag_count:
        frame_scores.pop('SAG1', None)
        frame_scores.pop('SAG2', None)

        log_recore.pop('SAG1',None)
        log_recore.pop('SAG2',None)

        plane_type = 'COR'
        print("The video is determined to be a coronal plane (COR)")
    elif sag_count > cor_count:
        frame_scores.pop('COR1', None)
        frame_scores.pop('COR2', None)
        frame_scores.pop('COR3', None)
        log_recore.pop('COR1',None)
        log_recore.pop('COR2',None)
        log_recore.pop('COR3',None)
        plane_type = 'SAG'
        print("The video is determined to be a sagittal plane (SAG)")
    else:
        print("The video contains an equal number of coronal and sagittal plane data, and the data has not been deleted")
    
    return frame_scores, plane_type,log_recore

def candidate_selection(frame_data, frame_idx, frame_candidates,log_candidates, model, transforms, device, thrh, photometric_interpretation=None):
    pil_frame = process_frame(frame_data, photometric_interpretation) if photometric_interpretation else frame_data
    w, h = pil_frame.size
    orig_size = torch.tensor([[w, h]]).to(device)
    im_data = transforms(pil_frame)[None].to(device)
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    lab = labels[scores > thrh]
    box = boxes[scores > thrh]
    scrs = scores[scores > thrh]
    possible_views = classify_frame(lab)    
    for view in possible_views:             
        frame_candidates[view][frame_idx],log_candidates[view][frame_idx] = calculate_score(lab, box, scrs, view, pil_frame)


def process_mp4_file(video_path, model, transforms, device, thrh):
    cap = cv2.VideoCapture(video_path)
    frame_candidates = initialize_candidates()
    log_candidates = initialize_candidates()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time_saving= []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        start_time = time.time()
        candidate_selection(pil_frame, frame_idx, frame_candidates,log_candidates, model, transforms, device, thrh)
        print(f"MP4 file {video_path}, frame {frame_idx} processed in {(time.time() - start_time) * 1000:.2f} ms")
        time_saving.append((time.time() - start_time) * 1000)
    fps = (frame_count-1) / (sum(time_saving[1:]) / 1000)

    return frame_candidates, log_candidates,frame_count, time_saving, fps

def process_dicom_file(dicom_path, model, transforms, device, thrh):
    ds = pydicom.dcmread(dicom_path, force=True)
    photometric_interpretation = ds.PhotometricInterpretation
    image_data = ds.pixel_array
    num_frames = image_data.shape[0] if len(image_data.shape) >= 3 else 1
    frame_candidates = initialize_candidates()
    log_candidates = initialize_candidates()
    time_saving = []

    for frame_idx in range(num_frames):     
        frame_data = image_data[frame_idx] if num_frames > 1 else image_data
        start_time = time.time()
        candidate_selection(frame_data, frame_idx, frame_candidates,log_candidates, model, transforms, device, thrh, photometric_interpretation)
        print(f"DICOM file {dicom_path}, frame {frame_idx} processed in {(time.time() - start_time) * 1000:.2f} ms")
        time_saving.append((time.time() - start_time) * 1000)
    fps = (num_frames-1) / (sum(time_saving[1:]) / 1000)

    return frame_candidates, log_candidates,num_frames,time_saving, fps

def process_clusters(frame_candidates, plane_type,num_frames,log_candidates):
    # Delete invaild frame based on the DBSCAN cluster
    cluster = {}
    if plane_type == "COR":
        cor2_frames = list(frame_candidates.get('COR2', {}).keys())
        cor2_scores = list(frame_candidates.get('COR2', {}).values())
        cor3_frames = list(frame_candidates.get('COR3', {}).keys())
        cor3_scores = list(frame_candidates.get('COR3', {}).values())
        cor1_frames = list(frame_candidates.get('COR1', {}).keys())
        cor1_scores = list(frame_candidates.get('COR1', {}).values())
        cor1_clusters = None
        cor2_clusters = None
        cor3_clusters = None

        if cor2_frames:
            cor2_data = np.array(list(zip(cor2_frames, cor2_scores)))
            cor2_clusters = DBSCAN(eps=4.0, min_samples=3).fit(cor2_data).labels_
            cor2_max_cluster_frames = np.array(cor2_frames)[cor2_clusters != -1] # delete noise frame
            if len(cor2_max_cluster_frames) > 0:   
                cor2_avg_frame = np.mean(cor2_max_cluster_frames)   # calculate this for determine scanning direction
            else:
                cor2_avg_frame = np.mean(cor2_frames)
        if cor3_frames:
            cor3_data = np.array(list(zip(cor3_frames, cor3_scores)))
            cor3_clusters = DBSCAN(eps=4.0, min_samples=3).fit(cor3_data).labels_
            cor3_avg_frame = np.mean(cor3_frames)  
         
        if cor1_frames:
            cor1_data = np.array(list(zip(cor1_frames, cor1_scores)))
            cor1_clusters = DBSCAN(eps=4.0, min_samples=3).fit(cor1_data).labels_

            if cor2_frames and cor3_frames and cor2_avg_frame < cor3_avg_frame:
                # COR scanning direction is COR1 -> COR2 -> COR3
                min_cor2_frame = min(cor2_max_cluster_frames) if len(cor2_max_cluster_frames) > 0 else min(cor2_frames)
                frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame <= min_cor2_frame - 4}

            elif cor2_frames and cor3_frames and cor2_avg_frame > cor3_avg_frame:
                # COR scanning direction is COR3 -> COR2 -> COR1
                max_cor2_frame = max(cor2_max_cluster_frames) if len(cor2_max_cluster_frames) > 0 else max(cor2_frames)
                frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame >= max_cor2_frame + 4}
            elif cor2_frames: # if there is not cor3
                min_cor2_frame = min(cor2_max_cluster_frames) if len(cor2_max_cluster_frames) > 0 else min(cor2_frames)
                max_cor2_frame = max(cor2_max_cluster_frames) if len(cor2_max_cluster_frames) > 0 else max(cor2_frames)
                
                # filter the frame of cor1 
                left_cor1_frames = [frame for frame in cor1_frames if frame < min_cor2_frame]
                right_cor1_frames = [frame for frame in cor1_frames if frame > max_cor2_frame]
                
                if left_cor1_frames and right_cor1_frames:
                    # calculate distance
                    start_to_cor1 = min(left_cor1_frames)
                    cor1_to_end =  num_frames - max(right_cor1_frames)
                    
                    if start_to_cor1 > cor1_to_end:
                        # if cor1 is closer to the start, the scanning direction is COR1 -> COR2
                        frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame <= min_cor2_frame - 4}
                    else:
                        # if cor1 is closer to the end, the scanning direction is COR2 -> COR1
                        frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame >= max_cor2_frame + 4}
                elif left_cor1_frames:
                    # Only left cor1 exists, the scanning direction is COR1 -> COR2
                    frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame <= min_cor2_frame - 4}
                elif right_cor1_frames:
                    # Only right cor1 exists, the scanning direction is COR2 -> COR1
                    frame_candidates['COR1'] = {frame: score for frame, score in frame_candidates['COR1'].items() if frame >= max_cor2_frame + 4}
                else:
                    pass
        
        cluster['COR1'] = cor1_clusters if cor1_clusters is not None else []
        cluster['COR2'] = cor2_clusters if cor2_clusters is not None else []    
        cluster['COR3'] = cor3_clusters if cor3_clusters is not None else []


    elif plane_type == "SAG":
        sag2_frames = list(frame_candidates.get('SAG2', {}).keys())
        sag2_scores = list(frame_candidates.get('SAG2', {}).values())

        sag1_frames = list(frame_candidates.get('SAG1', {}).keys())
        sag1_scores = list(frame_candidates.get('SAG1', {}).values())

        sag1_clusters = None
        sag2_clusters = None
        sag3_clusters = None  # new

        if sag2_frames:
            sag2_data = np.array(list(zip(sag2_frames, sag2_scores)))
            sag2_clusters = DBSCAN(eps=5.0, min_samples=3).fit(sag2_data).labels_
        if sag1_frames:
            sag1_data = np.array(list(zip(sag1_frames, sag1_scores)))
            sag1_clusters = DBSCAN(eps=5.0, min_samples=3).fit(sag1_data).labels_

        if sag2_frames and sag1_frames:
            unique_clusters = set(sag2_clusters)
            if len(unique_clusters - {-1}) > 1:
                # if there are more than one valid cluster in SAG2 
                cluster_labels = sorted(list(unique_clusters))
                left_sag2_cluster = min(np.array(sag2_frames)[sag2_clusters == cluster_labels[0]])
                right_sag2_cluster = max(np.array(sag2_frames)[sag2_clusters == cluster_labels[-1]])

                if left_sag2_cluster < min(sag1_frames) and right_sag2_cluster > max(sag1_frames):
                    # there are left and right parasagittal views in one video.
                    frame_candidates['SAG3'] = {frame: frame_candidates['SAG2'][frame] for frame in frame_candidates['SAG2'] if frame < min(sag1_frames)}
                    frame_candidates['SAG2'] = {frame: score for frame, score in frame_candidates['SAG2'].items() if frame >= max(sag1_frames)}
                    log_candidates['SAG3'] = {frame: log_candidates['SAG2'][frame] for frame in log_candidates['SAG2'] if frame < min(sag1_frames)}
                    log_candidates['SAG2'] = {frame: log for frame, log in log_candidates['SAG2'].items() if frame >= max(sag1_frames)}
                    sag3_frames = list(frame_candidates.get('SAG3', {}).keys())
                    sag3_scores = list(frame_candidates.get('SAG3', {}).values())
                    if sag3_frames:
                        sag3_data = np.array(list(zip(sag3_frames, sag3_scores)))
                        sag3_clusters = DBSCAN(eps=5.0, min_samples=3).fit(sag3_data).labels_
              
        cluster['SAG1'] = sag1_clusters if sag1_clusters is not None else []
        cluster['SAG2'] = sag2_clusters if sag2_clusters is not None else []
        cluster['SAG3'] = sag3_clusters if sag3_clusters is not None else []  

    return frame_candidates, cluster, log_candidates

def select_best_frame(key, frame_scores, log_record, best_log_candidates, cluster, best_frames, overall_best_frames):
    original_key = key
    if key == 'SAG2':
        if 'SAG2' in overall_best_frames.keys():
            key = 'SAG3'

    current_cluster = cluster.get(original_key, [])
    
    if len(current_cluster) > 0:
        frame_indices = list(frame_scores.keys())
        current_cluster = current_cluster[:len(frame_indices)]
        
        # valid cluster
        unique_clusters, counts = np.unique(current_cluster, return_counts=True)
        valid_clusters = unique_clusters[unique_clusters != -1]
        
        if len(valid_clusters) > 0:
            largest_cluster = valid_clusters[np.argmax(counts[unique_clusters != -1])]
            largest_cluster_size = counts[unique_clusters == largest_cluster][0]
            
            if largest_cluster_size >= 6:
                valid_frames = np.array(frame_indices)[current_cluster == largest_cluster]
                cluster_center = np.mean(valid_frames)
                cluster_std = np.std(valid_frames)
                
                # Gaussian distribution
                gaussian = norm(loc=cluster_center, scale=cluster_std)
                
                weighted_scores = {}
                for frame, score in frame_scores.items():
                    if frame in valid_frames:
                        gaussian_weight = gaussian.pdf(frame) / gaussian.pdf(cluster_center)
                        weighted_scores[frame] = score * (1 + gaussian_weight)            
                    else:
                        weighted_scores[frame] = score
 
                best_frame_idx = max(weighted_scores, key=weighted_scores.get)
                best_frames[key] = (best_frame_idx, frame_scores[best_frame_idx])
                best_log_candidates[key] = log_record[best_frame_idx]
    
                print(f"Select best frame {best_frame_idx}, Original score: {frame_scores[best_frame_idx]:.4f}, Weighted score: {weighted_scores[best_frame_idx]:.4f}")
                return best_frame_idx, frame_scores[best_frame_idx], key,  best_frames, best_log_candidates
        
    # original score
    best_frame_idx = max(frame_scores, key=lambda x: frame_scores[x][0] if isinstance(frame_scores[x], list) else frame_scores[x])
    print(f"Select best frame: {best_frame_idx}, score: {frame_scores[best_frame_idx]:.4f}")
    best_frames[key] = (best_frame_idx, frame_scores[best_frame_idx])
    best_log_candidates[key] = log_record[best_frame_idx]
    return best_frame_idx, frame_scores[best_frame_idx], key,  best_frames, best_log_candidates

def save_best_frames(frame_candidates, output_dir, check_id, check_date):
    extract_dir = os.path.join(output_dir, 'Standard View',check_id, check_date)
    os.makedirs(extract_dir, exist_ok=True)

    for key, (frame_idx, score, video_path) in frame_candidates.items():
        if '.DCM' in video_path:
            ds = pydicom.dcmread(video_path, force=True)
            image_data = ds.pixel_array
            photometric_interpretation = ds.PhotometricInterpretation
            frame_data = image_data[frame_idx] if len(image_data.shape) >= 3 else image_data
            pil_frame = process_frame(frame_data, photometric_interpretation)
        
        elif '.mp4' in video_path:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_path}")
                continue
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        
        save_path = os.path.join(extract_dir, f"{key}.png")
        pil_frame.save(save_path)
        print(f"Saved best frame {key} to {save_path}")

def process_files(model, video_dir, output_dir, device='cpu',thrh=0.75):
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for root, _, files in os.walk(video_dir):
        overall_best_candidate = {}
        overall_best_frames = {}
        best_frames = {}
        best_log_candidates = {}
        video_files = [f for f in files if f.lower().endswith(('.dcm', '.mp4'))]
        if len(video_files) > 0 :
            check_id,check_date = root.split('/')[-2],  root.split('/')[-1] 
            total_time = [] 
            total_fps = []
            for file in video_files:  
                video_path = os.path.join(root, file)
                if file.lower().endswith('.dcm'):
                    frame_candidates, log_candidates,total_frames, time_saving, fps = process_dicom_file(video_path, model, transforms, device, thrh)

                elif file.lower().endswith('.mp4'):
                    frame_candidates, log_candidates,total_frames, time_saving, fps = process_mp4_file(video_path, model, transforms, device, thrh)

                start_time = time.time()
                frame_candidates, plane_type, log_candidates = filter_frame_scores(frame_candidates,log_candidates)
                frame_candidates, cluster, log_candidates = process_clusters(frame_candidates, plane_type,total_frames,log_candidates)
                for key in ['COR1', 'COR2', 'COR3', 'SAG1', 'SAG2','SAG3']:
                    frame_scores = frame_candidates.get(key, {})
                    log_record = log_candidates.get(key, {})
                    if frame_scores:
                        best_frame_idx, best_frame_score, key1, best_frames, best_log_candidates = select_best_frame(key,frame_scores,log_record,best_log_candidates, cluster,best_frames,overall_best_frames)
                        print(f"Best frame for {key1} in {video_path}: {best_frame_idx} with score {best_frame_score}")
                for key, (frame_idx, score) in best_frames.items():
                    if key not in overall_best_frames or score > overall_best_frames[key][1]:
                        overall_best_frames[key] = (frame_idx, score, video_path)
                        overall_best_candidate[key] = best_log_candidates[key]
                
                end_time = time.time()
                time_saving.append((end_time - start_time) * 1000)  

                print(f"Aver Time for 1 frame: {np.mean(time_saving[1:-1]):.2f} ms")
                print(f"Max Time for 1 frame: {np.max(time_saving[1:-1]):.2f} ms")
                print(f"Min Time for 1 frame: {np.min(time_saving[1:-1]):.2f} ms")
                print(f"Total Time: {np.sum(time_saving[1:])/1000:.2f} s")
                total_time.append(np.sum(time_saving[1:])/1000) 
                total_fps.append(fps)
            
            save_best_frames(overall_best_frames, output_dir, check_id, check_date)
            os.makedirs(os.path.join(output_dir,'Diagnostic result',check_id,check_date),exist_ok=True)
            diagnosis_file = os.path.join(output_dir,'Diagnostic result',check_id,check_date, 'diagnosis_result.txt')
            with open(diagnosis_file, 'w', encoding='utf-8') as f:
                f.write(f"Diagnostic time：{(np.sum(total_time)):.2f} 秒\n")
                f.write(f'Candidate frame scoring process:\n')
                for key, (frame_idx, score, video_path) in overall_best_frames.items():
                    f.write(f"View {key}:\n")
                    log_record = overall_best_candidate.get(key, {})
                    for line in log_record:
                        f.write(f"  {line}\n")
                        #print(f"{key}  {line}")
                #f.write(f'FPS:{np.round(np.mean(total_fps))}\n')
            print(f"Diagnosis time written to {diagnosis_file}") 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_detection', type=str, default="./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml")
    parser.add_argument('-r', '--weight_detection', type=str, default="log/detection_weight/detection_weight.pth")
    parser.add_argument('-d', '--dicom-dir', type=str, default='./Example_', help='Path to directory containing DICOM/MP4 files')
    parser.add_argument('-o', '--output-dir', type=str, default='./output', help='Directory to save output AVI files')
    parser.add_argument('-de', '--device', type=str, default='cuda')
    args = parser.parse_args()
    extract_views(args)
