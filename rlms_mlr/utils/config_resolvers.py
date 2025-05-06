import logging

import hydra
from omegaconf import OmegaConf
import dotenv
import torch

def register_resolver(name: str,):
    """
    Decorator to register a custom OmegaConf resolver.
    Args:
        name: The name of the resolver to register.

    Returns: None

    """
    def decorator(resolver_func):
        OmegaConf.register_new_resolver(name, resolver_func)
        return resolver_func
    return decorator


@register_resolver("from_dotenv")
def from_env_resolver(env_var: str) -> str:
    """
    Custom OmegaConf resolver to get environment variables.
    Args:
        env_var: Name of the environment variable to retrieve.

    Returns: value of the environment variable

    """
    logging.info(f"Opening .env file to get variable '{env_var}'")
    dotenv.load_dotenv()
    value = dotenv.get_key(dotenv.find_dotenv(), env_var)
    if value is None:
        raise ValueError(f"Environment variable '{env_var}' not found.")
    return value


@register_resolver("get_device")
def get_device_resolver(device: str) -> str:
    """
    Custom OmegaConf resolver to validate the device.

    Args:
        device: The device to use, either 'cuda' or 'cpu'.

    Returns: The validated device string.

    """
    if device == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Defaulting to CPU.")
            return "cpu"
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.")

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)