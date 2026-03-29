import torch
import jax

def get_torch_device():

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_device_info():
   
    if torch.cuda.is_available():
        return {
            "framework": "PyTorch",
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
    else:
        return {
            "framework": "PyTorch",
            "device": "cpu"
        }

def get_jax_device():
    
    devices = jax.devices()
    return devices[0]


def get_jax_device_info():
   
    devices = jax.devices()

    return {
        "framework": "JAX",
        "device_type": devices[0].device_kind,
        "num_devices": len(devices)
    }

def get_full_device_info():
  
    return {
        "pytorch": get_torch_device_info(),
        "jax": get_jax_device_info()
    }
