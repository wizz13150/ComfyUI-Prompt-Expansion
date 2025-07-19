from .prompt_expansion import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .prompt_expansion import fooocus_expansion_path, wizzgpt_weights_path
from .model_loader import load_file_from_url
import os


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


def download_models():
    fooocus_url = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin'
    wizzgpt_url = 'https://huggingface.co/Wizz13150/WizzGPTv7/blob/main/pytorch_model.bin'
    model_dir = fooocus_expansion_path
    file_name = 'pytorch_model.bin'

    load_file_from_url(url=fooocus_url, model_dir=model_dir, file_name='pytorch_model.bin')
    load_file_from_url(url=wizzgpt_url, model_dir=model_dir, file_name=os.path.basename(wizzgpt_weights_path))


download_models()
