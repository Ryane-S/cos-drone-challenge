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
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible.")
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold

        self.processor = Sam3Processor.from_pretrained(model_name_or_path)
        self.model = Sam3Model.from_pretrained(model_name_or_path).to(self.device).eval()

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

        # 2. Encodage de l'image
        state = self.processor.set_image(pil_image)

        # 3. Prompt texte
        output = self.processor.set_text_prompt(state=state, prompt=self.prompt)

        masks = output.get("masks", torch.empty(0, 1, *rgb.shape[:2]))
        scores = output.get("scores", torch.empty(0))
        boxes = output.get("boxes", torch.empty(0, 4))

        # 4. Filtrer par seuil de confiance
        if scores.numel() > 0:
            keep = scores >= self.confidence_threshold
            masks = masks[keep]
            scores = scores[keep]
            boxes = boxes[keep]

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
    PROMPTS = ["laser", "red laser dot", "red dot", "bright spot"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataloader (même principe, juste moins d'images) ---
    loader = build_dataloader(
        "data/cityscapes",
        split="val",
        batch_size=1,
        num_workers=0,
        img_size=(512, 1024),
        drop_last=False,
    )

    # --- Récupérer quelques images ---
    images, targets = next(iter(loader))
    # Si vous avez un dataset preprocessed avec laser, adaptez le chemin ci-dessus.

    output_dir = Path("predictions_laser")
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt in PROMPTS:
        print(f"\n=== Prompt : '{prompt}' ===")
        detector = Sam3LaserDetector(
            device=DEVICE,
            confidence_threshold=0.3,
            prompt=prompt,
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