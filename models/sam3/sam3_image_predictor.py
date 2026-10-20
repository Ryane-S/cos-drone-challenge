from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import Sam3Model, Sam3Processor

from dataset_loaders.load_cityscapes_dataset import build_dataloader

model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")



CITYSCAPES_PROMPTS = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

IGNORE_PROMPTS = [
    "parking area",
    "rail track",
    "guard rail",
    "bridge",
    "tunnel",
    "caravan",
    "trailer",
]


class Sam3CityscapesPredictor:
    def __init__(
            self,
            model_name_or_path="facebook/sam3",
            device="cuda",
            threshold=0.5,
            mask_threshold=0.5,
            detect_ignored=False,
            local_files_only=False,
    ):
        self.device = torch.device(device)
        
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable."
            )

        self.threshold = threshold
        self.mask_threshold = mask_threshold

        self.processor = Sam3Processor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        self.model = Sam3Model.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image):
        """
        input RGB tensor [3, H, W], float32 in [0, 1].
        output tensor CPU [H, W], uint8
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Image attendue [3,H,W], reçu {image.shape}")

        image = image.detach().cpu()

        rgb = (
            image.clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        pil_image = Image.fromarray(rgb)

        height, width = rgb.shape[:2]

        prediction = torch.full(
            (height, width),
            255,
            dtype=torch.uint8,
            device=self.device,
        )
        best_score = torch.full(
            (height, width),
            -1.0,
            dtype=torch.float32,
            device=self.device,
        )

        prompts = list(enumerate(CITYSCAPES_PROMPTS))

        if self.detect_ignored:
            prompts.extend((255, prompt) for prompt in IGNORE_PROMPTS)

        # Un seul encodage visuel par image.
        state = self.processor.set_image(pil_image)

        for class_id, prompt in prompts:
            self.processor.reset_all_prompts(state)

            output = self.processor.set_text_prompt(
                prompt=prompt,
                state=state,
            )

            masks = output["masks"]
            scores = output["scores"].flatten().float()

            # Le processeur officiel retourne [N, 1, H, W].
            if masks.ndim != 4 or masks.shape[1] != 1:
                raise RuntimeError(
                    f"Format inattendu des masques : {masks.shape}"
                )

            masks = masks[:, 0].bool()

            # Fusion progressive : évite un tenseur temporaire
            # supplémentaire [N, H, W] rempli de scores.
            for mask, score in zip(masks, scores):
                update = mask & (score > best_score)
                prediction[update] = class_id
                best_score[update] = score

        return prediction.cpu()


def save_masks(prediction, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = prediction.numpy().astype(np.uint8)

    Image.fromarray(ids).save(output_dir / f"{name}_trainIds.png")

    lut = np.full(256, 255, dtype=np.uint8)
    lut[:19] = np.rint(np.linspace(0, 230, 19)).astype(np.uint8)

    Image.fromarray(lut[ids]).save(output_dir / f"{name}_gray.png")


def main():
    loader = build_dataloader(
        "data/cityscapes",
        split="val",
        batch_size=1,
        num_workers=0,
        img_size=(512, 1024),
        drop_last=False,
    )

    predictor = Sam3CityscapesPredictor(
        checkpoint_path="checkpoints/sam3.pt",
        confidence_threshold=0.5,
        detect_ignored=False,
    )

    images, targets = next(iter(loader))

    for index, image in enumerate(images):
        prediction = predictor.predict(image)

        assert prediction.shape == targets[index].shape

        save_masks(
            prediction,
            output_dir="predictions",
            name=f"sample_{index:04d}",
        )

        print("Prédictions :", torch.unique(prediction).tolist())
        print("Attendues   :", torch.unique(targets[index]).tolist())


if __name__ == "__main__":
    main()
