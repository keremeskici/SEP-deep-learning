import torch

# simple device selection
# checks for GPU support and picks best option

def get_device(device_preference=None):
    # auto select best device
    if device_preference is None or device_preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    device_preference = device_preference.lower()
    
    # cuda requested
    if device_preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("CUDA not available, using auto")
            return get_device('auto')
    
    # mps requested
    elif device_preference == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS not available, using CPU")
            return torch.device('cpu')
    
    # cpu requested
    elif device_preference == 'cpu':
        return torch.device('cpu')
    
    # unknown
    else:
        print(f"Unknown device '{device_preference}', using auto")
        return get_device('auto')

def get_device_name(device):
    # get readable device name
    if device.type == 'cuda':
        return "NVIDIA GPU"
    elif device.type == 'mps':
        return "Apple GPU"
    else:
        return "CPU"

def print_device_info(device):
    # print device info
    print(f"Using: {get_device_name(device)}")
