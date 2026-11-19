"""
Config du prédicteur sam3
"""
from pathlib import Path

from configs.config_cityscapes import setting_config_cityscapes


class SettingConfigSam3(setting_config_cityscapes):
    split = 'train'
    original_mask_encoding = 'trainid'

    prompts = [
        "green laser dot",
        "green laser pointer spot",
        "green laser projection spot",
        "green dot",
    ]
    confidence = 0.2
    limit = None

    metrics = ['iou', 'dice', 'precision', 'recall', 'map']

    # Sortie
    output = Path("output")
    save_masks = True

    # Token
    token_file = 'sam3_token'
