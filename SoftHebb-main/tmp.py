# Import PyTorch dependencies
import torch
import intel_extension_for_pytorch as ipex


from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
from torcheval.metrics import MulticlassAccuracy

def get_public_properties(obj):
    return {
        prop: getattr(obj, prop)
        for prop in dir(obj)
        if not prop.startswith("__") and not callable(getattr(obj, prop))
    }
print(f'PyTorch Version: {torch.__version__}')
print(f'Intel PyTorch Extension Version: {ipex.__version__}')
xpu_device_count = torch.xpu.device_count()
dict_properties_list = [get_public_properties(torch.xpu.get_device_properties(i)) for i in range(xpu_device_count)]
print(dict_properties_list)