"""Aperçu manuel : python demo_manual.py mon_image.jpg"""
import matplotlib.pyplot as plt
from dataset_loaders.synthetic_laser import LaserParams, load_image, render_laser

# Tous les paramètres sont accessibles ici. None centre la marque.
PARAMS = LaserParams(
    position=None,  # Exemple : (250, 180).
    a=20.0,
    b=15.0,
    angle_deg=30.0,
    p=2.0,
    intensity=0.5,
    color=(0.0, 1.0, 0.0),
    white_strength=0.8,
    white_threshold=0.6,
    speckle_strength=0.5,
    speckle_scale=1.2,
    bloom_strength=0.5,
    bloom_sigma=8.0,
    bloom_threshold=0.4,
    flare_strength=0.15,
    flare_radius=25.0,
    flare_threshold=0.4,
    ghost_positions=(0.6, 1.2, 1.7),
    flare_color=(0.15, 1.0, 0.15),
    star_strength=0.25,
    star_rays=6,
    star_length=100.0,
    star_width=1.5,
    star_angle_deg=15.0,
    star_threshold=0.4,
    star_color=(0.75, 1.0, 0.75),
    seed=42,
)


def main():
    image = load_image("sample.png")
    result = render_laser(image, PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, frame, title in zip(axes, (image, result), ('Original', 'Laser synthétique')):
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
