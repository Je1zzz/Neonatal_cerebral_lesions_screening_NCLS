from utils.tool_utils import source_import
import torch
import tempfile
from bisect import bisect_right
from utils.dist_utils import dist
from utils.dist_utils import is_main_process,is_dist_avail_and_initialized
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os
import math
import warnings
from collections import OrderedDict
from torch.utils.data import  DataLoader



def init_model(config,pretrain=True):
    weights_path = config['training_opt']['pretrain_path']
    use_my_pth = config['training_opt']['use_my_pth']
    device = torch.device(config['device'])
    networks_defs = config['network']
    networks = {}  
    def_file = networks_defs['def_file'] 
    model_args = networks_defs['params']  
    networks= source_import(def_file).create_model(**model_args)

    if weights_path is not None and pretrain:
        if os.path.exists(weights_path) and not use_my_pth:
            if 'convnext' in weights_path or 'swin' in weights_path:
                weights_dict = torch.load(weights_path)['model']   # convnext, swin-transformer need to add ['model']
            else:
                weights_dict = torch.load(weights_path)

            model_dict = networks.state_dict()
            # if rank == 0:
            #     print(weights_dict.keys())
            #     print(model_dict.keys())
            matched_weights = {}
            for k, v in weights_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    matched_weights[k] = v

            missing,unexpected = networks.load_state_dict(matched_weights, strict=False)
            if is_main_process() :
                print(f"missing:{missing}\nunexpected:{unexpected}")
        
        elif os.path.exists(weights_path) and use_my_pth:
            checkpoint = torch.load(weights_path, map_location='cpu')
            weight_dict = checkpoint['model']
            new_state_dict = OrderedDict()
            for k, v in weight_dict.items():
                name = k[7:] if k.startswith('module.') else k  # 删除 'module.' 前缀
                if not ( 
                name.startswith('disease_classifier') or 
                name.startswith('self_attention') or
                name.startswith('cls_token')):
                    new_state_dict[name] = v
            # print(weight_dict.keys())
            # print(model.state_dict().keys())
            print(networks.load_state_dict(new_state_dict,strict=False))
        else:
            raise FileNotFoundError('not found weights file: {}'.format(weights_path))
    else:
        if is_dist_avail_and_initialized():
            checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
            # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
            if is_main_process():
                torch.save(networks.state_dict(), checkpoint_path)

            dist.barrier()
            # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
            networks.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return networks


def init_criterion(config,hsm=False):
    criterion_defs = config['criterions']
    criterions = {}
    criterion_weights = {}
    for loss_name, loss_info in criterion_defs.items():
        def_file = loss_info['def_file']  # 损失函数的定义文件
        loss_args = loss_info['loss_params']  # 损失函数的参数
        weight = loss_info['weight']  # 损失函数的权重
        if 'CBLoss' in def_file :
            loss_args['patient_per_cls'] = config['patient_per_cls'] if loss_args['use_patient_per_cls'] else None
            loss_args['image_per_cls'] = config['image_per_cls'] if not loss_args['use_patient_per_cls'] else None
        if 'BCELoss' in def_file and loss_args['use_posweight'] and not hsm:
            loss_args['pos_weight'] = config['pos_weight']

        criterions[loss_name] = source_import(def_file).create_loss(**loss_args).cuda()
        criterion_weights[loss_name] = weight # 权重
        
    return criterions, criterion_weights


def init_optimizer(config, op):
    training_opt = config['training_opt']

    if 'optimizer' in training_opt and training_opt['optimizer'] == 'adam':
        if is_main_process():
            print('using Adam optimizer\n')
        optimizer = torch.optim.Adam(params=op['params'],lr=float(op['lr']),weight_decay=op['weight_decay'])
    elif 'optimizer' in training_opt and training_opt['optimizer'] == 'sgd':
        if is_main_process():
            print('using SGD optimizer\n')
        optimizer = torch.optim.SGD(params=op['params'],lr=float(op['lr']),weight_decay=op['weight_decay'],momentum=op['momentum'])
    elif 'optimizer' in training_opt and training_opt['optimizer'] == 'adamw':
        if is_main_process():
            print('using AdamW optimizer\n')
        optimizer = torch.optim.AdamW(params=op['params'], lr=float(op['lr']), weight_decay=op['weight_decay'])

    if training_opt['coslr']:  
        schedule= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_opt['epochs'], eta_min=training_opt['endlr'])  
        if is_main_process():
            print('using coslr scheduler')
    elif training_opt['warmup']:  
        schedule = WarmupMultiStepLR(optimizer, training_opt['lr_step'],gamma=training_opt['lr_factor'], warmup_epochs=training_opt['warm_epoch'],decay_method=
                                     training_opt['decay_method'],total_epoch=training_opt['epochs'])
        if is_main_process():
            print('using warmup scheduler')
    else:    
        schedule=  torch.optim.lr_scheduler.StepLR(optimizer,step_size=training_opt['step_size'],gamma=training_opt['gamma']) 
        if is_main_process():
            print('using steplr scheduler')
    return optimizer, schedule

def extract_params(config,networks):
    training_opt = config['training_opt']
    model_params = [p for p in networks.parameters() if p.requires_grad]
    model_optim_params_dict = {'params': model_params,
                                'lr': training_opt['lr'],
                                'momentum': training_opt['momentum'],
                                'weight_decay': training_opt['weight_decay']}
    return model_optim_params_dict



def load_my_model(config):
    device = torch.device('cuda')
    checkpoint = torch.load(config['weight_path'], map_location='cpu')
    model = init_model(config,pretrain=False)
    model = model.to(device)
    model.eval()

    weight_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 删除 'module.' 前缀
        new_state_dict[name] = v
    # print(weight_dict.keys())
    # print(model.state_dict().keys())
    print(model.load_state_dict(new_state_dict,strict=True))
    return model


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler): # 继承torch.optim.lr_scheduler._LRScheduler，重写get_lr
    def __init__(
        self,
        optimizer,
        milestones, # 表明在哪个epochs时进行学习率降低的列表
        gamma=0.1, # 学习率衰减因子
        warmup_factor=1.0 / 3, # warmup默认因子
        warmup_epochs=5, # warmup的epochs
        warmup_method="linear", # warmup的方式，选择constant或linear
        decay_method = "coslr",
        last_epoch=-1,
        total_epoch= 100,
        eta_min = 0.
    ):
        if not list(milestones) == sorted(milestones): # 如果不是升序的列表，不符合要求
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"): # 只能是这两种方法
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        
        if decay_method not in ("coslr", "steplr"): # 只能是这两种方法
            raise ValueError(
                "Only 'coslr' or 'steplr' warmup_method accepted"
                "got {}".format(decay_method)
            )
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.decay_method = decay_method
        self.total_epochs = total_epoch
        self.eta_min = eta_min
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            warmup_factor = 1

        if self.decay_method == "step":
            lr_scale = warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
        elif self.decay_method == "coslr" and self.last_epoch >= self.warmup_epochs:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            post_warmup_epochs = self.total_epochs - self.warmup_epochs
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / post_warmup_epochs))
            lr_scale = warmup_factor * (self.eta_min + (1 - self.eta_min) * cosine_factor)
        else:
            lr_scale = warmup_factor

        return [base_lr * lr_scale for base_lr in self.base_lrs]
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.""" 
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels_name = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            if isinstance(p,int):
                self.matrix[p,t] += 1
            else:
                self.matrix[p.cpu().numpy().astype(int), t.cpu().numpy().astype(int)] += 1

    def summary(self):
        TP = np.diag(self.matrix)
        FP = np.sum(self.matrix, axis=1) - TP
        FN = np.sum(self.matrix, axis=0) - TP
        TN = np.sum(self.matrix) - (FP + FN + TP)

        # Calculate precision, recall, F1 score, and specificity, replacing NaNs and infinities
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            precision = np.nan_to_num(np.where(TP + FP == 0, 0, TP / (TP + FP)))
            recall = np.nan_to_num(np.where(TP + FN == 0, 0, TP / (TP + FN)))
            f1 = np.nan_to_num(np.where((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)))
            specificity = np.nan_to_num(np.where(TN + FP == 0, 0, TN / (TN + FP)))

        # 打印每个类别的性能指标
        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Specificity"]
        for i in range(self.num_classes):
            table.add_row([
                self.labels_name[i],
                f"{precision[i]:.4f}",
                f"{recall[i]:.4f}",
                f"{f1[i]:.4f}",
                f"{specificity[i]:.4f}"
            ])
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        macro_specificity = np.mean(specificity)
        overall_accuracy = np.sum(TP) / np.sum(self.matrix)

        # 打印整体性能指标
        print(table)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Macro Specificity: {macro_specificity:.4f}")

        return overall_accuracy, macro_precision, macro_recall, macro_f1, macro_specificity

    def plot(self, name, save_path):
        matrix = self.matrix
        if is_main_process():
            print(matrix)
            print('\n')
            plt.imshow(matrix, cmap=plt.cm.Blues)  # 使用蓝色色谱
            plt.xticks(range(self.num_classes), self.labels_name, rotation=45)
            plt.yticks(range(self.num_classes), self.labels_name)
            plt.colorbar()
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            plt.title('Confusion matrix')

            # 在图中标注数量/概率信息
            thresh = matrix.min() + (matrix.max()-matrix.min()) / 2
            for x in range(self.num_classes):
                for y in range(self.num_classes):
                    info = int(matrix[y, x])
                    plt.text(x, y, info,
                            verticalalignment='center',
                            horizontalalignment='center',
                            fontsize=12,  # 增加字体大小
                            color="white" if matrix[y, x] > thresh else "black")  # 调整文本颜色
            plt.tight_layout()
            plt.savefig(os.path.join(save_path,f"confusion_matrix_{name}.png"))
            plt.close()  


def load_model(config,device):
    checkpoint = torch.load(config['training_opt']['weight_path'], map_location='cpu')
    model = init_model(config,pretrain=False)
    model = model.to(device)
    model.eval()

    weight_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 删除 'module.' 前缀
        new_state_dict[name] = v
    # print(weight_dict.keys())
    # print(model.state_dict().keys())
    print(model.load_state_dict(new_state_dict,strict=True))
    return model

def update_counters(id_number, id_date, is_incorrect, conditions, counters, total_counters):
    for condition in conditions:
        found = False
        condition_path = os.path.join('./data/cases', f'{condition}.txt')
        if os.path.exists(condition_path):
            with open(condition_path, 'r') as cond_file:
                for line in cond_file:
                    line = line.replace(' ', '').split(';')
                    if id_number == line[0] and id_date == line[1]:
                        total_counters[condition] += 1  # Increment total count
                        if is_incorrect:
                            counters[condition] += 1
                        found = True
                        break
            if found:
                break
        else:
            total_counters[condition] += 1
            if is_incorrect:
                counters[condition] += 1
            break

def hard_sample_mining(config, data_loader, device, new_batch_size=32):
    difficult_samples = []
    loss_function, loss_weight = init_criterion(config,hsm=True)
    model = load_model(config, device)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            model.eval()
            imgs, patient_labels, img_labels, slice_labels, patient_lengths, _,_ = data
            imgs = imgs.to(device)
            patient_labels = patient_labels.to(device)

            patient_pred, _, _ = model(imgs, patient_lengths)
            patient_loss = loss_function['Loss_hsm'](patient_pred, patient_labels) * loss_weight['Loss1']
            if patient_labels == 1:
                difficult_samples.append((patient_loss.item(), data))

    difficult_samples.sort(key=lambda x: x[0], reverse=True)

    top_difficult_samples = difficult_samples[:int(0.2 * len(difficult_samples))]
    new_dataset = data_loader.dataset
    if new_batch_size > 1:
        new_dataset.use_augmentation = True
    
    count_patient_label_tds = [0] * 2  # Initialize counters for labels 0-7
    count_patient_label_ac = [0] * 2
    count_change = 0

    conditions = ['normal','intervention','nonintervention']
    counters = {cond: 0 for cond in conditions}
    total_counters = {cond: 0 for cond in conditions} 
    
    # Open a file to write the statistics
    with open("sample_statistics_positive.txt", "a") as file:
        # Write header
        
        for index, (patient_loss, data) in enumerate(top_difficult_samples):
            imgs, patient_labels, img_labels, slice_labels, patient_lengths, patient_ids, patients = data
            imgs = imgs.to(device)
            img_labels = img_labels.to(device)
            check_id, check_date = patient_ids[0].split('_')[0], patient_ids[0].split('_')[1]

            update_counters(check_id,check_date,True,conditions,counters,total_counters)

            patient_label_value = patient_labels.item()
            if 0 <= patient_label_value < 2:
                count_patient_label_tds[patient_label_value] += 1

            # Sample images and labels
            num_samples = torch.randint(3, 7, (1,)).item()
            sampled_indices = torch.randperm(imgs.size(0))[:num_samples]
            sampled_imgs = imgs[sampled_indices]
            sampled_patient_indices = [int(sampled_idx.item()) for sampled_idx in sampled_indices]
            sampled_patients = [patients[0][sampled_idx] for sampled_idx in sampled_patient_indices]
            sampled_img_labels = img_labels[sampled_indices]
            sampled_img_labels_list = sampled_img_labels.tolist()
            sampled_slice_labels = slice_labels[sampled_indices]
            sampled_slice_labels_list = sampled_slice_labels.tolist()

            flag_3, flag_5 = False,False
            def check_conditions(labels):
                if labels[3]==1 :
                    flag_3 = True
                if labels[5] == 1:
                    flag_5 = True
                return (labels[1] == 1 or labels[2] == 1 or (labels[3] == 1 and labels[5] == 1))
            conditions_met = [check_conditions(labels) for labels in sampled_img_labels]

            if not any(conditions_met) and not (flag_3 and flag_5) and patient_labels == 1:
                patient_labels.fill_(0)
                count_change += 1

            patient_label_value_after = patient_labels.item()
            if 0 <= patient_label_value_after < 2:
                count_patient_label_ac[patient_label_value_after] += 1
            
            difficult_samples_data = [(sampled_patients[i], patient_label_value_after, sampled_img_labels_list[i], sampled_slice_labels_list[i]) for i in range(len(sampled_indices))]
            
            new_dataset.append_samples(difficult_samples_data)

        # Write statistics to file
        for condition, count in counters.items():
            label_text = f"{condition} : {count}\n"
            file.write(label_text)
        
        file.write("Before changing:\n")
        for label in range(2):
            file.write(f'patient label {label}= {count_patient_label_tds[label]}\n')
        
        file.write("After changing:\n")
        for label in range(2):
            file.write(f'patient label {label}= {count_patient_label_ac[label]}\n')
        
        file.write(f'change {count_change}\n\n')  
    
    nw = min([os.cpu_count(), new_batch_size if new_batch_size > 1 else 0, 8])
    new_data_loader = DataLoader(new_dataset, 
                                  batch_size=new_batch_size, 
                                  shuffle=True, 
                                  num_workers=nw,
                                  pin_memory=True,
                                  collate_fn=data_loader.dataset.collate_fn,
                                  drop_last=False)
    
    print("Total difficult samples:", len(top_difficult_samples))
    print("Count change: ",count_change)
    print("New dataset size:", len(new_data_loader.dataset))
    return new_data_loader, len(top_difficult_samples)-count_change, count_change
