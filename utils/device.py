import torch

def get_device(device_preference=None):
    if device_preference == 'auto' or device_preference is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    device_preference = device_preference.lower()
    if device_preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_preference == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
        
    return torch.device('cpu')


def get_device_name(device):
    if device.type == 'cuda':
        return "NVIDIA GPU"
    if device.type == 'mps':
        return "Apple GPU"
    return "CPU"


def print_device_info(device):
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print(f"  Device name: Apple Metal Performance Shaders")
    else:
        print(f"  Device name: CPU")
