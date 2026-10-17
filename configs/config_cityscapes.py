"""
Configuration du projet pour le dataset Cityscapes. (Inspiré de : https://github.com/JCruan519/VM-UNet/blob/main/configs/config_setting.py)
"""

class setting_config_cityscapes:
    # Emplacements des datasets
    data_path = 'data/cityscapes'

    # Paramètres du dataset
    num_classes = 19
    ignore_index = 255

    # Résolution d'entrée
    input_size_h = 512
    input_size_w = 1024

    # Paramètres du dataloader
    batch_size = 2
    num_workers = 4 # A adapter en fonction du nombre de cpus par tache (--cpus-per-task)
    pin_memory = True

    # Reproductibilité
    seed = 42