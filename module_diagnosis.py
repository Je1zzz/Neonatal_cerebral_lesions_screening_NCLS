import os
import re
import yaml
import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from collections import  OrderedDict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.function_utils import init_model
from utils.tool_utils import update
import argparse
import time
import cv2
from module_extract_view import extract_views
from collections import defaultdict

def remove_black_borders(img):
        image = img.copy()
        area_threshold = 20000
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim == 2: 
            gray = image
        elif image.ndim == 3: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid image dimensions")
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.5 and cv2.contourArea(contour) > area_threshold:
                valid_contours.append(contour)
        if valid_contours:
            c_max = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_max)
            for c in valid_contours:
                if c is not c_max:
                    cv2.drawContours(image, [c], -1, 0, thickness=cv2.FILLED)
            cropped_image = image[y:y+h, x:x+w]
        else:
            cropped_image = image
        if cv2.contourArea(c_max) > area_threshold:
            return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Invalid image")

def set_seed(config):
    if config['training_opt']['seed'] is not None:
        seed = config['training_opt']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"set seed {seed}")

class TTATransform:
    def __init__(self):
        rgb_mean, rgb_std = [0.1798, 0.1799, 0.1799], [0.2080, 0.2080, 0.2081]
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.tta_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.6, contrast=0.6)], p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
            transforms.Normalize(self.rgb_mean, self.rgb_std)
        ]
        self.transform = transforms.Compose(self.tta_transforms)
    def __call__(self, img):
        return self.transform(img)

class SimpleTransform:
    def __init__(self):
        rgb_mean, rgb_std = [0.1798, 0.1799, 0.1799], [0.2080, 0.2080, 0.2081]
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        transform_list = [
                            transforms.Resize(300),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            #transforms.Normalize(self.rgb_mean, self.rgb_std)
                        ]
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self,img):
        img = remove_black_borders(img)
        return self.transform(img)

class NeonatalCranialBags(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.bags = {}
        self.transform = SimpleTransform()
        self._create_bags()

    def _create_bags(self):
        for patient_id in os.listdir(self.base_dir):
            patient_dir = os.path.join(self.base_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            for examine_date in os.listdir(patient_dir):
                examine_dir = os.path.join(patient_dir, examine_date)
                if not os.path.isdir(examine_dir):
                    continue

                bag = []
                for img_file in os.listdir(examine_dir):
                    if img_file.endswith('.png') and '聚类分布' not in img_file:  
                        img_path = os.path.join(examine_dir, img_file)
                        bag.append(img_path)

                if bag:
                    pid = f"{patient_id}_{examine_date}"
                    self.bags[pid] = bag
        self.bags_list = list(self.bags.values())


    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        parts = bag[0].split('/')
        patient_id = f'{parts[-3]}_{parts[-2]}'
        img_idx = [os.path.basename(img_path) for img_path in bag] 
        bag_imgs = [self.transform(Image.open(img_path).convert('RGB')) for img_path in bag]
        bag_lengths = torch.tensor([len(bag_imgs)])
        return bag_imgs, bag_lengths, patient_id, bag, img_idx

    def extract_patient_id(self, path):
        patient_id = path.split('/')[-3]
        examine_date = path.split('/')[-2]
        match = re.search(r'\d+', patient_id)
        if match:
            return f"{match.group(0)}_{examine_date}"
        return None
    
    @staticmethod
    def collate_fn(batch):
        bag_imgs, bag_lengths, patienti_ids, bag, img_idx = zip(*batch)
        images = torch.cat([torch.stack(bag) for bag in bag_imgs], dim=0)
        bag_lengths = torch.tensor([len(imgs) for imgs in bag_imgs])
        return images, bag_lengths, patienti_ids, bag, img_idx
    
def apply_tta_and_predict(model, imgs, bag_lengths, device, tta_transform, num_tta=10):
    # Aapply {num_tta} times TTA and obtain the average prediction
    tta_patient_pred = []
    tta_img_pred = []
    for _ in range(num_tta):
        augmented_imgs = torch.stack([tta_transform(img) for img in imgs])
        augmented_imgs = augmented_imgs.to(device)
        patient_pred, img_pred, _ = model(augmented_imgs, bag_lengths)
        tta_patient_pred.append(patient_pred)
        tta_img_pred.append(img_pred)
    avg_patient_pred = torch.mean(torch.stack(tta_patient_pred), dim=0)
    avg_img_pred = torch.mean(torch.stack(tta_img_pred), dim=0)
    return avg_patient_pred, avg_img_pred

def load_model(config,i):
    device = torch.device(config['device'])
    path = os.path.join(config['weight_classfication'])
    checkpoint = torch.load(path, map_location='cpu')
    model = init_model(config, pretrain=False)
    model = model.to(device)
    model.eval()
    weight_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        new_state_dict[name] = v
    print(model.load_state_dict(new_state_dict, strict=False))
    return model

def predict_analysis(cfg, model_list, dataloader, tta_transform):
    device = torch.device(cfg['device'])

    severe_img = {1, 2}  
    all_patient_preds = defaultdict(list)
    all_img_preds = defaultdict(list)
    all_bag_length = []
    all_patient_ids = []

    device = config['device']
    patient_preds_softmax_roc = {}
    start_time =time.time()
    with torch.no_grad():
        for fold,model in enumerate(model_list):
            model.eval()
            for data in tqdm(dataloader, desc='testing...', leave=False):
                imgs, bag_lengths, patient_id, data_path, img_idx = data
                imgs = imgs.to(device)

                if fold == 0:
                    name = [f'{path[0].split("/")[-3]} {path[0].split("/")[-2]}' for path in data_path]
                    all_patient_ids.extend(data_path)
                    all_bag_length.extend(bag_lengths)
                
                patient_pred, img_pred = apply_tta_and_predict(model, imgs, bag_lengths, device, tta_transform, num_tta=10)
                all_patient_preds[fold].extend(patient_pred.detach().cpu().numpy())
                all_img_preds[fold].extend(img_pred.detach().cpu().numpy())
        all_patient_preds_np = np.array(list(all_patient_preds.values()))
        all_img_preds_np = np.array(list(all_img_preds.values()))

        ensemble_preds = torch.tensor(np.mean(all_patient_preds_np, axis=0))
        ensemble_img_preds = torch.tensor(np.mean(all_img_preds_np, axis=0))

        ensemble_preds_softmax = F.softmax(ensemble_preds, dim=1)
        patient_probs, patient_preds = torch.max(ensemble_preds_softmax, 1)
        threshold_class_1 = 0.7     # Non-severe
        threshold_class_2 = 0.3     # Severe
        threshold = 0.5

        patient_preds = torch.where(ensemble_preds_softmax[:, 0] > threshold_class_1, 0, 1)  # Non-severe
        patient_preds = torch.where(ensemble_preds_softmax[:, 1] > threshold_class_2, 1, patient_preds)  # Severe

        img_preds = torch.sigmoid(ensemble_img_preds)
        img_preds = (img_preds > threshold).int()   # the threshold for multi-label classification should be adjust.

        for i, pid in enumerate(name):
            patient_preds_softmax_roc[pid]=ensemble_preds_softmax[i, 1].detach().cpu().numpy()


        predicted_needs_severe = torch.zeros(len(all_bag_length), dtype=torch.int)
        img_predicted_needs_severe = torch.zeros(len(all_bag_length), dtype=torch.int)
        total_predicted_severe = torch.zeros(len(all_bag_length), dtype=torch.int)

        start_idx = 0
        for i, length in enumerate(all_bag_length):
            end_idx = start_idx + length
            check_number, check_date = name[i].split(' ')
            
            txt_path = os.path.join(cfg['output_dir'],'Diagnostic result',check_number,check_date,'diagnosis_result.txt')
            patient_img_preds = img_preds[start_idx:end_idx]

            # case predicted head
            if patient_preds[i] == 1:
                predicted_needs_severe[i] = 1
            
            # image predicted head
            img_flag_3 = False
            img_flag_5 = False
            img_flag_use = True
            itv_img_count = 0
            itv_img_normal_count = 0

            for vector in patient_img_preds:
                img_same_3 = False
                img_same_5 = False
                if vector[0] == 1:
                    itv_img_normal_count += 1
                if any(vector[idx] == 1 for idx in severe_img):  
                    # if there is a image predict as PVL or Hydrocephalus directly, the case predict as severe.
                    img_predicted_needs_severe[i] = 1
                    itv_img_count += 1
                if vector[3] == 1:
                    img_flag_3 = True
                    img_same_3 = True
                if vector[5] == 1:
                    img_flag_5 = True
                    img_same_5 = True
                if img_same_3 and img_same_5:
                    # if IVH and dilatation exist at the same time, the case should be grade III or IV IVH and predict as severe.
                    itv_img_count += 1
                    img_flag_use = False

            if img_flag_3 and img_flag_5:
                img_predicted_needs_severe[i] = 1
                if img_flag_use:
                    itv_img_count += 1

            img_flag_3 = False
            img_flag_5 = False
            
            # Combine the case and image prediction
            if predicted_needs_severe[i] == 0 and img_predicted_needs_severe[i] == 1:
                if itv_img_count >= 2 or (itv_img_count == 1 and itv_img_normal_count < 3):
                    total_predicted_severe[i] = 1
                else:
                    total_predicted_severe[i] = 0
            elif predicted_needs_severe[i] == 1 and img_predicted_needs_severe[i] == 0:
                if patient_probs[i] > 0.95:
                    total_predicted_severe[i] = 1
                else:
                    total_predicted_severe[i] = 0
            else:
                total_predicted_severe[i] = predicted_needs_severe[i]
            
            #total_predicted_severe[i] = predicted_needs_severe[i]
            final_result = "Severe" if total_predicted_severe[i] == 1 else "Non-severe"
            
            start_idx = end_idx
            end_time = time.time()
            time_in_sec = 0.0  
            diagnosis_process = []
            if os.path.exists(txt_path):    
                with open(txt_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()  
                    
                    for line in lines:
                        if line.startswith('Diagnostic time'):
                            time_in_sec = float(line.strip().replace("Diagnostic time：", "").replace("Diagnostic time:", "").replace("秒", "").strip())  
                        elif line.startswith('Candidate frame scoring process'):
                            diagnosis_process.append(line.strip())
                            for next_line in lines[lines.index(line)+1:]:
                                diagnosis_process.append(next_line.strip())

            now_time = time_in_sec + (end_time - start_time)

            if not os.path.exists(os.path.dirname(txt_path)):
                os.makedirs(os.path.dirname(txt_path),exist_ok=True)

            with open(txt_path, 'w', encoding='utf-8') as file:
                file.write(f'Diagnostic time: {now_time:.2f}秒\n')
                file.write(f'Identity: AI\n')
                file.write(f'Diagnostic result: {final_result}\n\n')
                if diagnosis_process:
                    file.write('\n'.join(diagnosis_process))
            start_time = time.time()
            print(f'ID: {check_number}, Date: {check_date}, Final Result: {final_result}')
            print(f'Save txt to {txt_path}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_classfication', type=str, default='configs/convnext.yaml',help='Config file for classification')
    parser.add_argument('--weight_classfication', type=str, default='log/diagnostic_weight/model_35.pth',help='Weight file for classification')
    
    parser.add_argument('--cfg_detection', type=str, default="configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",help='Config file for detection')
    parser.add_argument('--weight_detection', type=str, default="log/detection_weight/detection_weight.pth",help='Weight file for detection')
    parser.add_argument('--dicom-dir', type=str, default='Example_', help='Root dir for dicom files(root/ID/Date/*.dcm)')
    parser.add_argument('--output-dir', type=str, default='output', help='Store results')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open(args.cfg_classfication) as f:
        config = yaml.safe_load(f)
    config = update(config, args)
    set_seed(config)
    model_list = []
    for i in range(1,6):
        model = load_model(config,i)
        model_list.append(model)
    extract_views(args)
    bag = NeonatalCranialBags(os.path.join(config['output_dir'],'Standard View'))
    dataloader = DataLoader(bag, batch_size=32, shuffle=False, collate_fn=bag.collate_fn)
    tta_transform = TTATransform()
    predict_analysis(config, model_list,dataloader, tta_transform)



    