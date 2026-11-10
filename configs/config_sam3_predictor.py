"""
Config du prédicteur sam3
"""
from pathlib import Path

from configs.config_cityscapes import setting_config_cityscapes


class SettingConfigSam3(setting_config_cityscapes):
    split = 'val'
    original_mask_encoding = 'trainid'

    prompts = PROMPTS = [
        "green laser dot",
        "green laser pointer spot",
        "green laser projection spot",
        "green dot",
    ]
    confidence = 0.3
    limit = 250

    metrics = ['recall']

    # Sortie
    output = Path("output")
    save_masks = True

    # Token
    token_file = 'sam3_token'
