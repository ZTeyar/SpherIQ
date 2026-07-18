import torch
from torch import nn as nn
from huggingface_hub import hf_hub_url
from pyiqa.utils.download_util import load_file_from_url
from collections import OrderedDict

# --------------------------------------------
# Common utils
# --------------------------------------------

def clean_state_dict(state_dict):
    """
    Clean checkpoint by removing .module prefix from state dict if it exists from parallel training.

    Args:
        state_dict (dict): State dictionary from a model checkpoint.

    Returns:
        dict: Cleaned state dictionary.
    """
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def get_url_from_name(
    name: str, store_base: str = 'hugging_face', base_url: str = None
) -> str:
    """
    Get the URL for a given file name from a specified storage base.

    Args:
        name (str): The name of the file.
        store_base (str, optional): The storage base to use. Options are "hugging_face" or "github". Default is "hugging_face".
        base_url (str, optional): Base URL to use if provided.

    Returns:
        str: The URL of the file.
    """
    if base_url is not None:
        url = f'{base_url}/{name}'
    elif store_base == 'hugging_face':
        url = hf_hub_url(repo_id='chaofengc/IQA-PyTorch-Weights', filename=name)
    elif store_base == 'github':
        url = f'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/{name}'
    return url


def load_pretrained_network(
    net: torch.nn.Module,
    model_path: str,
    strict: bool = True,
    weight_keys: str = None,
) -> None:
    """
    Load a pretrained network from a given model path.

    Args:
        net (torch.nn.Module): The network to load the weights into.
        model_path (str): Path to the model weights file. Can be a URL or a local file path.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by net's state_dict(). Default is True.
        weight_keys (str, optional): Specific key to extract from the state_dict. Default is None.

    Returns:
        None
    """
    if model_path.startswith('https://') or model_path.startswith('http://'):
        model_path = load_file_from_url(model_path)

    print(f'Loading pretrained model {net.__class__.__name__} from {model_path}')
    state_dict = torch.load(
        model_path, map_location=torch.device('cpu'), weights_only=False
    )
    if weight_keys is not None:
        state_dict = state_dict[weight_keys]
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)
