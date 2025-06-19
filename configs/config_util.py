import yaml
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    path: str
    max_new_tokens: int
    enable_thinking: bool
    torch_dtype: str
    device_map: str
    attn_implementation: str


def load_config(file_path: str) -> ModelConfig:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    required_keys = {'path', 'max_new_tokens', 'enable_thinking', 'torch_dtype', 'device_map', 'attn_implementation'}
    if not required_keys.issubset(config_data['model']):
        missing = required_keys - set(config_data['model'].keys())
        config['torch_dtype'] = dtype_map.get(config['torch_dtype'], torch.bfloat16)
        raise ValueError(f"Missing required keys in config: {missing}")

    return ModelConfig(
        path=config_data['model']['path'],
        max_new_tokens=config_data['model']['max_new_tokens'],
        enable_thinking=config_data['model']['enable_thinking'],
        torch_dtype = config_data['model']['torch_dtype'],
        device_map = config_data['model']['device_map'],
        attn_implementation = config_data['model']['attn_implementation']
    )


# Пример использования
if __name__ == "__main__":
    config = load_config("Qwen3_test.yaml")
    print(config)