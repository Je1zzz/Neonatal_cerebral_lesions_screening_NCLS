import os
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
import pickle
import dill
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ['WORLD_SIZE'])
        args['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args['rank'] = int(os.environ['SLURM_PROCID'])
        args['gpu'] = args['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args['distributed'] = False
        return
    args['distributed'] = True
    torch.cuda.set_device(args['gpu'])
    args['dist_backend'] = 'nccl'  # Communication backend, recommended to use NCCL for Nvidia GPU
    print('| distributed init (rank {}): {}'.format(
        args['rank'], args['dist_url']), flush=True)
    dist.init_process_group(backend=args['dist_backend'], init_method=args['dist_url'],
                            world_size=args['world_size'], rank=args['rank'])
    dist.barrier()



def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        if isinstance(value, np.ndarray):
            value = torch.tensor(value,dtype=torch.float32)
        elif isinstance(value, (int, float)):
            value = torch.tensor([value], dtype=torch.float32)
        elif torch.is_tensor(value):
            value = value.float()
        else:
            raise TypeError("Unsupported type for 'value'. Expected a PyTorch tensor, NumPy array, or scalar, but got {}".format(type(value)))
        value = value.to('cuda')
        dist.barrier()
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value
    

def gather_patient_pred(local_data):
    world_size = get_world_size()
    
    if dist.get_rank() == 0:  # is_main_process()
        gathered_data = [None] * world_size
    else:
        gathered_data = None
    pickled_data = dill.dumps(local_data)

    dist.gather_object(pickled_data, object_gather_list=gathered_data if is_main_process() else None)
    dist.barrier()  # Ensure all processes have completed gather_object

    if is_main_process():
        all_patient_predictions = {}
        for data in gathered_data:
            part = dill.loads(data)
            for patient_id, values in part.items():
                if patient_id not in all_patient_predictions:
                    all_patient_predictions[patient_id] = defaultdict(list)
                for key in values:
                    all_patient_predictions[patient_id][key].extend(values[key])
        return all_patient_predictions
    else:
        return None