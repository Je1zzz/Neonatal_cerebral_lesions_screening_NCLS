import sys
from utils.function_utils import init_criterion, ConfusionMatrix
from tqdm import tqdm
import torch
from utils.dist_utils import reduce_value, is_main_process, is_dist_avail_and_initialized
import torch.nn.functional as F
import torch.distributed as dist
from utils.tool_utils import check_tensor_device, get_grad_norm 
import numpy as np
import os

def train_one_epoch(model, optimizer, data_loader, device, epoch, config, save_path):
    model.train()
    wk_path = save_path
    loss_function, loss_weight = init_criterion(config)
    accu_loss = torch.zeros(1).to(device)
    accu_patient_loss = torch.zeros(1).to(device)
    accu_img_loss = torch.zeros(1).to(device)
    accu_plane_loss = torch.zeros(1).to(device)
    train_instance = any(param.requires_grad for name, param in model.named_parameters() if 'instance_classifier' in name)
    train_plane = any(param.requires_grad for name, param in model.named_parameters() if 'plane_classifier' in name)
    all_patient_labels = []
    all_patient_preds = []
    labels_name = ['No Intervention', 'Intervention']
    intervention_confusion_matrix = ConfusionMatrix(len(labels_name), labels_name)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    torch.autograd.set_detect_anomaly(True)
    for step, data in enumerate(data_loader):

        optimizer.zero_grad()
        imgs, patient_labels, img_labels, plane_labels, bag_lengths, _,_ = data

        imgs = imgs.to(device)
        patient_labels = patient_labels.to(device)
        img_labels = img_labels.to(device)
        plane_labels = plane_labels.to(device)

        patient_pred, img_pred, plane_pred = model(imgs,bag_lengths)

        patient_loss = loss_function['Loss1'](patient_pred, patient_labels) * loss_weight['Loss1']
        img_loss = loss_function['Loss2'](img_pred, img_labels) * loss_weight['Loss2']
        plane_loss = loss_function['Loss3'](plane_pred, plane_labels) * loss_weight['Loss3']
        
        if train_instance and train_plane:
            total_loss = patient_loss + img_loss + plane_loss
        else:
            total_loss = patient_loss

        total_loss.backward()
        #grad_norm = get_grad_norm(model)
        #print(f"Gradient Norm: {grad_norm}")
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accu_loss += total_loss.detach()
        accu_patient_loss += patient_loss.detach()
        accu_img_loss += img_loss.detach()
        accu_plane_loss += plane_loss.detach()

        threshold = 0.5
        patient_preds_softmax = F.softmax(patient_pred,dim=1)
        patient_probs,patient_preds = torch.max(patient_preds_softmax,1)
        img_preds = (torch.sigmoid(img_pred) > threshold).int()
        plane_preds_softmax = F.softmax(plane_pred, dim=1)
        plane_probs, plane_preds = torch.max(plane_preds_softmax, 1)
        
        accu_loss = reduce_value(accu_loss, average=True)
        accu_patient_loss = reduce_value(accu_patient_loss, average=True)
        accu_img_loss = reduce_value(accu_img_loss, average=True)
        accu_plane_loss = reduce_value(accu_plane_loss, average=True)

        needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)
        predicted_needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)

        for i, (label_vector, patient_label_vector) in enumerate(zip(patient_preds, patient_labels)):
            if label_vector == 1:
                predicted_needs_intervention[i] = 1
            if patient_label_vector == 1:
                needs_intervention[i] = 1

        intervention_confusion_matrix.update(check_tensor_device(predicted_needs_intervention), check_tensor_device(needs_intervention))

        if is_main_process():
            data_loader.desc = "[train epoch {}] loss: {:.3f}  lr: {:.6f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                optimizer.param_groups[0]["lr"]
            )
        if is_dist_avail_and_initialized():
            all_preds = [torch.zeros_like(patient_preds) for _ in range(dist.get_world_size())]
            all_labels = [torch.zeros_like(patient_labels) for _ in range(dist.get_world_size())]        
            dist.all_gather(all_preds, patient_preds)
            dist.all_gather(all_labels, patient_labels)
            patient_preds = torch.cat(all_preds)
            patient_labels = torch.cat(all_labels)
        all_patient_labels.extend(patient_labels.detach().cpu().numpy())
        all_patient_preds.extend(patient_preds.detach().cpu().numpy())

    all_patient_labels_np = np.array(all_patient_labels)
    all_patient_preds_np = np.array(all_patient_preds)
    intervention_acc, intervention_precision, intervention_recall, intervention_f1, intervention_specificity = intervention_confusion_matrix.summary()
    intervention_confusion_matrix.plot('intervention_train', wk_path)

    loss = {}
    loss['train_loss'] = accu_loss.item() / (step + 1)
    loss['train_img_loss'] = accu_img_loss / (step + 1)
    loss['train_plane_loss'] = accu_plane_loss / (step + 1)
    
    return loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, config, save_path):
    model.eval()
    wk_path = save_path
    loss_function, loss_weight = init_criterion(config)
    accu_loss = torch.zeros(1).to(device)
    accu_patient_loss = torch.zeros(1).to(device)
    accu_img_loss = torch.zeros(1).to(device)
    accu_plane_loss = torch.zeros(1).to(device)
    train_instance = any(param.requires_grad for name, param in model.named_parameters() if 'instance_classifier' in name)
    train_plane = any(param.requires_grad for name, param in model.named_parameters() if 'plane_classifier' in name)
    all_patient_labels = []
    all_patient_preds = []
    labels_name = ['No Intervention', 'Intervention']
    intervention_confusion_matrix = ConfusionMatrix(len(labels_name), labels_name)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    torch.autograd.set_detect_anomaly(True)
    for step, data in enumerate(data_loader):

        imgs, patient_labels, img_labels, plane_labels, bag_lengths, _,_ = data

        imgs = imgs.to(device)
        patient_labels = patient_labels.to(device)
        img_labels = img_labels.to(device)
        plane_labels = plane_labels.to(device)

        patient_pred, img_pred, plane_pred = model(imgs,bag_lengths)

        patient_loss = loss_function['Loss1'](patient_pred, patient_labels) * loss_weight['Loss1']
        img_loss = loss_function['Loss2'](img_pred, img_labels) * loss_weight['Loss2']
        plane_pred = plane_pred.view(-1, plane_pred.size(-1))
        plane_labels = plane_labels.view(-1)
        plane_loss = loss_function['Loss3'](plane_pred, plane_labels) * loss_weight['Loss3']
        
        if train_instance and train_plane:
            total_loss = patient_loss + img_loss + plane_loss
        else:
            total_loss = patient_loss

        accu_loss += total_loss.detach()
        accu_patient_loss += patient_loss.detach()
        accu_img_loss += img_loss.detach()
        accu_plane_loss += plane_loss.detach()

        threshold = 0.5
        patient_preds_softmax = F.softmax(patient_pred,dim=1)
        patient_probs,patient_preds = torch.max(patient_preds_softmax,1)
        img_preds = (torch.sigmoid(img_pred) > threshold).int()
        plane_preds_softmax = F.softmax(plane_pred, dim=1)
        plane_probs, plane_preds = torch.max(plane_preds_softmax, 1)
        
        accu_loss = reduce_value(accu_loss, average=True)
        accu_patient_loss = reduce_value(accu_patient_loss, average=True)
        accu_img_loss = reduce_value(accu_img_loss, average=True)
        accu_plane_loss = reduce_value(accu_plane_loss, average=True)

        needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)
        predicted_needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)

        for i, (label_vector, patient_label_vector) in enumerate(zip(patient_preds, patient_labels)):
            if label_vector == 1:
                predicted_needs_intervention[i] = 1
            if patient_label_vector == 1:
                needs_intervention[i] = 1

        intervention_confusion_matrix.update(check_tensor_device(predicted_needs_intervention), check_tensor_device(needs_intervention))

        if is_main_process():
            data_loader.desc = "[val epoch {}] loss: {:.3f} ".format(
                epoch,
                accu_loss.item() / (step + 1),
            )
        if is_dist_avail_and_initialized():
            all_preds = [torch.zeros_like(patient_preds) for _ in range(dist.get_world_size())]
            all_labels = [torch.zeros_like(patient_labels) for _ in range(dist.get_world_size())]        
            dist.all_gather(all_preds, patient_preds)
            dist.all_gather(all_labels, patient_labels)
            patient_preds = torch.cat(all_preds)
            patient_labels = torch.cat(all_labels)
        all_patient_labels.extend(patient_labels.detach().cpu().numpy())
        all_patient_preds.extend(patient_preds.detach().cpu().numpy())

    all_patient_labels_np = np.array(all_patient_labels)
    all_patient_preds_np = np.array(all_patient_preds)
    intervention_acc, intervention_precision, intervention_recall, intervention_f1, intervention_specificity = intervention_confusion_matrix.summary()
    intervention_confusion_matrix.plot('intervention_train', wk_path)

    loss = {}
    loss['val_loss'] = accu_loss.item() / (step + 1)
    loss['val_img_loss'] = accu_img_loss / (step + 1)
    loss['val_plane_loss'] = accu_plane_loss / (step + 1)
    
    return loss


@torch.no_grad()
def evaluate_test(model, data_loader, device, save_path):
    model.eval()
    wk_path = save_path
    all_patient_labels = []
    all_patient_preds = []
    labels_name = ['No Intervention', 'Intervention']
    intervention_confusion_matrix = ConfusionMatrix(len(labels_name), labels_name)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    torch.autograd.set_detect_anomaly(True)
    for step, data in enumerate(data_loader):

        imgs, patient_labels, img_labels, plane_labels, bag_lengths, _,_ = data

        imgs = imgs.to(device)
        patient_labels = patient_labels.to(device)
        img_labels = img_labels.to(device)
        plane_labels = plane_labels.to(device)

        patient_pred, img_pred, plane_pred = model(imgs,bag_lengths)


        threshold = 0.5
        patient_preds_softmax = F.softmax(patient_pred,dim=1)
        patient_probs,patient_preds = torch.max(patient_preds_softmax,1)
        img_preds = (torch.sigmoid(img_pred) > threshold).int()
        plane_preds_softmax = F.softmax(plane_pred, dim=1)
        plane_probs, plane_preds = torch.max(plane_preds_softmax, 1)
        

        needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)
        predicted_needs_intervention = torch.zeros(patient_labels.size(0), dtype=torch.int)

        for i, (label_vector, patient_label_vector) in enumerate(zip(patient_preds, patient_labels)):
            if label_vector == 1:
                predicted_needs_intervention[i] = 1
            if patient_label_vector == 1:
                needs_intervention[i] = 1

        intervention_confusion_matrix.update(check_tensor_device(predicted_needs_intervention), check_tensor_device(needs_intervention))

        if is_main_process():
            data_loader.desc = "final testing ..."
            
        if is_dist_avail_and_initialized():
            all_preds = [torch.zeros_like(patient_preds) for _ in range(dist.get_world_size())]
            all_labels = [torch.zeros_like(patient_labels) for _ in range(dist.get_world_size())]        
            dist.all_gather(all_preds, patient_preds)
            dist.all_gather(all_labels, patient_labels)
            patient_preds = torch.cat(all_preds)
            patient_labels = torch.cat(all_labels)
        all_patient_labels.extend(patient_labels.detach().cpu().numpy())
        all_patient_preds.extend(patient_preds.detach().cpu().numpy())

    all_patient_labels_np = np.array(all_patient_labels)
    all_patient_preds_np = np.array(all_patient_preds)
    intervention_acc, intervention_precision, intervention_recall, intervention_f1, intervention_specificity = intervention_confusion_matrix.summary()
    intervention_confusion_matrix.plot('intervention_train', wk_path)
    
