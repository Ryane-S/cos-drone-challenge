"""
Inférence zero-shot SAM3 pour détecter un point laser.
Itère sur l'ensemble du dataloader et optimise le chargement du modèle.
"""

import warnings

from configs.config_sam3_predictor import SettingConfigSam3
from models.sam3.evaluate_laser import evaluate

# Masque les alertes de dépréciation et les avertissements de cache Hugging Face sous Windows
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from transformers import Sam3Model, Sam3Processor
from dataset_loaders.load_cityscapes_dataset import build_dataloader


class Sam3LaserDetector:
    def __init__(
        self,
        config,
        model_name_or_path="facebook/sam3",
        device="cuda",
        token=""
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible.")
        self.confidence_threshold = config.confidence

        if token == "":
            raise RuntimeError("Token HF non renseigné")

        self.processor = Sam3Processor.from_pretrained(model_name_or_path, token=token)
        self.model = Sam3Model.from_pretrained(model_name_or_path, token=token).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_tensor, prompt="laser"):
        """
        image_tensor : [3, H, W] float32 in [0, 1]
        prompt : Texte décrivant l'objet à détecter
        Retourne un dict avec masks, boxes, scores.
        """
        # 1. Tensor -> PIL Image (SAM3 attend du uint8 RGB)
        # Fonctionne même si l'image a été placée sur le GPU.
        rgb = (
            image_tensor.detach()
            .cpu()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        pil_image = Image.fromarray(rgb)

        # 2. Préparation des entrées via l'API standard d'Hugging Face (Texte dynamique + Image)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        # 3. Inférence du modèle
        outputs = self.model(**inputs)

        # 4. Post-processing
        post_processed_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        masks = post_processed_results["masks"]    # Tenseur de masques binaires redimensionnés
        boxes = post_processed_results["boxes"]    # Boîtes au format [x1, y1, x2, y2]
        scores = post_processed_results["scores"]  # Scores de confiance de chaque détection

        return {
            "masks": masks.cpu(),
            "scores": scores.cpu(),
            "boxes": boxes.cpu(),
            "image": pil_image,
        }


def save_visualization(result, output_path):
    """Sauvegarde une figure avec l'image, les masques et les boîtes."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(result["image"])
    ax.set_title(f'Prompt: "{result["prompt"]}" — {len(result["scores"])} détection(s)')
    ax.axis("off")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result["scores"]), 1)))

    for i, (mask, score, box) in enumerate(
        zip(result["masks"], result["scores"], result["boxes"])
    ):
        mask_np = mask.detach().cpu().numpy().squeeze().astype(bool)
        overlay = np.zeros((*mask_np.shape, 4))
        overlay[mask_np] = [*colors[i % len(colors)][:3], 0.5]
        ax.imshow(overlay)

        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=colors[i % len(colors)], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"{score:.2f}",
            color=colors[i % len(colors)], fontsize=9, weight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_mask(result, output_path):
    """Affiche et sauvegarde l'image, le masque et sa superposition."""
    image = result["image"]
    width, height = image.size

    masks = result["masks"].detach().cpu()

    if masks.numel() == 0:
        merged_mask = np.zeros((height, width), dtype=bool)
    else:
        # Accepte les formes (N, H, W) et (N, 1, H, W).
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        if masks.ndim != 3:
            raise ValueError(f"Forme des masques inattendue : {tuple(masks.shape)}")

        merged_mask = masks.bool().any(dim=0).numpy()

        if merged_mask.shape != (height, width):
            raise ValueError("Le masque et l'image n'ont pas les mêmes dimensions.")

    overlay = np.zeros((height, width, 4), dtype=float)
    overlay[merged_mask] = [0.0, 1.0, 0.0, 0.5]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        constrained_layout=True,
    )

    axes[0].imshow(image)
    axes[0].set_title("Image originale")

    axes[1].imshow(
        merged_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("Masque prédit")

    axes[2].imshow(image)
    axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title("Superposition")

    for ax in axes:
        ax.axis("off")

    prompt = result.get("prompt", "")
    fig.suptitle(
        f'Prompt : "{prompt}" — {len(result["scores"])} détection(s)'
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def main(prompts=[], nb_images=None):
    PROMPTS = [
        # "laser",
        # "laser dot",
        # "laser beam",
        # "laser spot",
        # "laser pointer spot",
        # "laser point",
        # "laser projection spot",
        # "light",
        # "dot",
        # "ellipse",
        # "green laser",
        "green laser dot",
        # "green laser beam",
        # "green laser spot",
        "green laser pointer spot",
        # "green laser point",
        "green laser projection spot",
        # "green light",
        "green dot",
        # "green ellipse",
        # "colored laser",
        # "colored laser beam",
        # "colored laser dot",
        # "colored laser spot",
        # "colored laser pointer spot",
        # "colored laser point",
        # "colored laser projection spot",
        # "colored light",
        "colored dot",
        # "colored ellipse",
        # "green",
        # "projection"
    ]
    config = SettingConfigSam3()
    # pred_df = pd.DataFrame(prompts, columns=["Prompt name"])
    # data_path = "data/cityscapes/laser"
    # split = "val"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")

    with open(config.token_file, "r") as file:
        token = file.read().strip()

    # --- Dataloader complet ---
    loader = build_dataloader(
        data_path=config.data_path,
        split=config.split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        img_size=(config.input_size_h, config.input_size_w),
        drop_last=False,
    )

    # output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialisation UNIQUE du détecteur ---
    print("Chargement de SAM3 en mémoire...")
    detector = Sam3LaserDetector(
        config,
        device=DEVICE,
        token=token
    )
    evaluate(config, detector)

    # Compteur global d'images traitées
    # global_image_idx = 0

    # --- Boucle principale sur l'ensemble du dataset ---
    # for batch_idx, (batch_images, batch_targets) in enumerate(loader):
    #     for img_in_batch_idx, image in enumerate(batch_images):
    #         preds = np.zeros(len(prompts))
    #
    #         # Pour chaque image, on teste l'ensemble des prompts configurés
    #         for i, prompt in enumerate(prompts):
    #             result = detector.predict(image, prompt=prompt)
    #             result["prompt"] = prompt
    #             # masks = result["masks"]
    #             # predicted_mask = masks.any(dim=0)
    #
    #             preds[i] = len(result['scores'])
    #             print(f"[Image {global_image_idx}] Prompt '{prompt}': {len(result['scores'])} détection(s)")
    #
    #             if len(result["scores"]) > 0:
    #                 print(f"  Scores: {result['scores'].tolist()}")
    #                 print(f"  Boîtes: {result['boxes'].tolist()}")
    #
    #             safe_prompt = prompt.replace(" ", "_")
    #             filename = f"img_{global_image_idx:03d}_{safe_prompt}.png"
    #             save_mask(result, config.output / filename)
    #
    #         pred_df[str(global_image_idx)] = preds
    #         print(pred_df)
    #
    #     global_image_idx += 1
    #     if nb_images is not None and nb_images < global_image_idx:
    #         break
    #
    # print(f"\nInférence terminée. {global_image_idx} images traitées avec succès.")
    # print(pred_df)
    # pred_df.to_csv("best_prompts.csv")


if __name__ == "__main__":
    main()
