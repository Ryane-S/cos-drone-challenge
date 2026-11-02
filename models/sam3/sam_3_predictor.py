"""
Inférence zero-shot SAM3 pour détecter un point laser.
Itère sur l'ensemble du dataloader et optimise le chargement du modèle.
"""

import warnings
# Masque les alertes de dépréciation et les avertissements de cache Hugging Face sous Windows
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import Sam3Model, Sam3Processor
from dataset_loaders.load_cityscapes_dataset import build_dataloader

class Sam3LaserDetector:
    def __init__(
        self,
        model_name_or_path="facebook/sam3",
        device="cuda",
        confidence_threshold=0.3,
        token=""
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible.")
        self.confidence_threshold = confidence_threshold

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
        rgb = (
            image_tensor.clamp(0, 1)
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
        mask_np = mask.squeeze().numpy() if mask.ndim > 2 else mask.numpy()
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


def main():
    # --- Paramètres ---
    PROMPTS = [
        # "laser",
        # "laser dot",
        # "laser beam",
        "laser spot",
        "laser pointer spot",
        "laser point",
        "laser projection spot",
        # "light",
        # "dot",
        # "ellipse",
        # "green laser",
        # "green laser dot",
        # "green laser beam",
        "green laser spot",
        "green laser pointer spot",
        "green laser point",
        "green laser projection spot",
        # "green light",
        # "green dot",
        # "green ellipse",
        # "colored laser",
        # "colored laser beam",
        # "colored laser dot",
        "colored laser spot",
        "colored laser pointer spot",
        "colored laser point",
        "colored laser projection spot",
        # "colored light",
        # "colored dot",
        # "colored ellipse",
        # "green",
        "projection"
    ]

    PROMPTS_RESULTS = {
        "laser": (0, 0, 0),
        "laser dot": (0, 0, 0),
        "laser beam": (0, 0, 0),
        "laser spot": (0, 0, 0),
        "laser pointer spot": (0, 0, 0),
        "laser point": (0, 0, 0),
        "laser projection spot": (0, 0, 0),
        "light": (25, 25, 12),
        "dot": (1, 0, 0),
        "ellipse": (0, 0, 0),
        "green laser": (1, 0, 0),
        "green laser dot": (1, 0, 0),
        "green laser beam": (1, 0, 0),
        "green laser spot": (1, 0, 0),
        "green laser pointer spot": (1, 0, 0),
        "green laser point": (1, 0, 0),
        "green laser projection spot": (1, 1, 0),
        "green light": (1, 1, 1),
        "green dot": (1, 1, 0),
        "green ellipse": (1, 0, 0),
        "colored laser": (0, 0, 0),
        "colored laser beam": (0, 0, 0),
        "colored laser dot": (1, 0, 0),
        "colored laser spot": (1, 0, 0),
        "colored laser pointer spot": (1, 0, 0),
        "colored laser point": (1, 0, 0),
        "colored laser projection spot": (0, 0, 0),
        "colored light": (4, 0, 2),
        "colored dot": (1, 0, 0),
        "colored ellipse": (0, 0, 0),
        "green": (0, 0, 0),
        "projection": (0, 0, 0),
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")

    with open("sam3_token", "r") as file:
        token = file.read().strip()

    # --- Dataloader complet ---
    loader = build_dataloader(
        "samples/laser",
        split="val",
        batch_size=1,
        num_workers=0,
        img_size=(512, 1024),
        drop_last=False,
    )

    output_dir = Path("predictions_laser")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialisation UNIQUE du détecteur ---
    print("Chargement de SAM3 en mémoire...")
    detector = Sam3LaserDetector(
        device=DEVICE,
        confidence_threshold=0.3,
        token=token
    )

    # Compteur global d'images traitées
    global_image_idx = 0

    # --- Boucle principale sur l'ensemble du dataset ---
    for batch_idx, (batch_images, batch_targets) in enumerate(loader):
        for img_in_batch_idx, image in enumerate(batch_images):

            # Pour chaque image, on teste l'ensemble des prompts configurés
            for prompt in PROMPTS:
                result = detector.predict(image, prompt=prompt)
                result["prompt"] = prompt

                print(f"[Image {global_image_idx}] Prompt '{prompt}': {len(result['scores'])} détection(s)")
                if len(result["scores"]) > 0:
                    print(f"  Scores: {result['scores'].tolist()}")
                    print(f"  Boîtes: {result['boxes'].tolist()}")

                safe_prompt = prompt.replace(" ", "_")
                filename = f"img_{global_image_idx:03d}_{safe_prompt}.png"
                save_visualization(result, output_dir / filename)

            global_image_idx += 1

    print(f"\nInférence terminée. {global_image_idx} images traitées avec succès.")


if __name__ == "__main__":
    main()
