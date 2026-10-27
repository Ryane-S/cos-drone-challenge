"""
Inférence zero-shot SAM3 pour détecter un point laser.
Garde le dataloader existant, mais utilise un prompt unique.
"""

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
        prompt="laser",
        token=""
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible.")
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold

        if token == "":
            raise RuntimeError("Token HF non renseigné")

        self.processor = Sam3Processor.from_pretrained(model_name_or_path, token=token)
        self.model = Sam3Model.from_pretrained(model_name_or_path, token=token).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_tensor):
        """
        image_tensor : [3, H, W] float32 in [0, 1]
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

        # 2. Préparation des entrées via l'API standard d'Hugging Face (Texte + Image)
        inputs = self.processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)

        # 3. Inférence du modèle
        outputs = self.model(**inputs)

        # 4. Post-processing avec les fonctions utilitaires du processeur Hugging Face
        # (Seuil par défaut à 0.5 ajusté selon vos besoins de détection)
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
    PROMPTS = ["laser", "laser dot", "laser beam", "car", "building"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")

    with open("sam3_token", "r") as file:
        token = file.read().strip()  # .strip() pour enlever les espaces ou retours à la ligne du token

    # --- Dataloader ---
    loader = build_dataloader(
        "samples",
        split="val",
        batch_size=1,
        num_workers=0,
        img_size=(512, 1024),
        drop_last=False,
    )

    # --- Récupérer quelques images ---
    images, targets = next(iter(loader))

    output_dir = Path("predictions_laser")
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt in PROMPTS:
        print(f"\n=== Prompt : '{prompt}' ===")
        detector = Sam3LaserDetector(
            device=DEVICE,
            confidence_threshold=0.3,
            prompt=prompt,
            token=token
        )

        for index, image in enumerate(images):
            result = detector.predict(image)
            result["prompt"] = prompt

            print(f"Image {index}: {len(result['scores'])} détection(s)")
            if len(result["scores"]) > 0:
                print(f"  Scores: {result['scores'].tolist()}")
                print(f"  Boîtes: {result['boxes'].tolist()}")

            safe_prompt = prompt.replace(" ", "_")
            save_visualization(
                result,
                output_dir / f"{safe_prompt}_sample_{index:02d}.png",
            )


if __name__ == "__main__":
    main()
