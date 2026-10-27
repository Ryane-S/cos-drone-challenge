"""Dix générations : python demo_random.py mon_image.jpg --seed 42"""
import matplotlib.pyplot as plt
import numpy as np
from dataset_loaders.synthetic_laser import load_image, generate_synthetic_laser


def main():
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('image')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--count', type=int, default=10)
    # parser.add_argument('--position', type=float, nargs=2, metavar=('X', 'Y'))
    # args = parser.parse_args()
    # if args.count < 1:
    #     parser.error('--count doit être >= 1')
    count = 25
    image = load_image("sample.png")
    seeds = np.random.default_rng(42).integers(0, 2**32, size=count)
    cols = min(5, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for ax, seed in zip(axes.flat, seeds):
        result, metadata = generate_synthetic_laser(
            image, seed=int(seed))
        ax.imshow(result)
        ax.set_title(f"seed={seed} | I={metadata['latents']['intensity']:.2f}")
    plt.show()


if __name__ == '__main__':
    main()
