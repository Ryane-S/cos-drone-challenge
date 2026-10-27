"""Laser synthétique 2D. Dépendances : numpy, scipy, pillow.

Composition RGB simplifiée, non radiométrique. Les effets ne constituent
pas une simulation calibrée d'un objectif ou d'un capteur.
"""
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter


@dataclass
class LaserParams:
    # Position : pixels, origine en haut à gauche. None = centre.
    position: tuple | None = None
    a: float = 70.0
    b: float = 40.0
    angle_deg: float = 30.0
    p: float = 2.0
    intensity: float = 0.8
    color: tuple = (0.0, 1.0, 0.0)  # Couleur RGB, pas une longueur d'onde.
    white_strength: float = 0.9
    white_threshold: float = 0.6
    speckle_strength: float = 0.5
    speckle_scale: float = 1.2
    bloom_strength: float = 0.5
    bloom_sigma: float = 8.0
    bloom_threshold: float = 0.4
    flare_strength: float = 0.15
    flare_radius: float = 25.0
    flare_threshold: float = 0.4
    ghost_positions: tuple = (0.6, 1.2, 1.7)
    flare_color: tuple = (0.15, 1.0, 0.15)
    star_strength: float = 0.25
    star_rays: int = 6  # Nombre pair de branches.
    star_length: float = 100.0  # Distance de décroissance à 1/e.
    star_width: float = 1.5  # Écart-type transversal.
    star_angle_deg: float = 15.0
    star_threshold: float = 0.4
    star_color: tuple = (0.75, 1.0, 0.75)
    seed: int | None = None  # Contrôle uniquement le speckle du rendu.


def load_image(path):
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert('RGB'), dtype=float) / 255


def _validate(image, p):
    image = np.asarray(image, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Image RGB attendue : (hauteur, largeur, 3).')
    if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError('Image flottante finie dans [0, 1] attendue.')
    for name, value in asdict(p).items():
        if name not in ('seed', 'position') and not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'{name} doit être fini.')
    for name in ('a', 'b', 'speckle_scale', 'bloom_sigma', 'flare_radius', 'star_length', 'star_width'):
        if getattr(p, name) <= 0:
            raise ValueError(f'{name} doit être > 0.')
    for name in ('intensity', 'bloom_strength', 'flare_strength', 'star_strength',
                 'bloom_threshold', 'flare_threshold', 'star_threshold'):
        if getattr(p, name) < 0:
            raise ValueError(f'{name} doit être >= 0.')
    if p.p < 1 or not 0 <= p.white_threshold < 1:
        raise ValueError('p >= 1 et 0 <= white_threshold < 1 requis.')
    for name in ('white_strength', 'speckle_strength'):
        if not 0 <= getattr(p, name) <= 1:
            raise ValueError(f'{name} doit être dans [0, 1].')
    for name in ('color', 'flare_color', 'star_color'):
        c = np.asarray(getattr(p, name))
        if c.shape != (3,) or np.any((c < 0) | (c > 1)):
            raise ValueError(f'{name} doit être un triplet RGB dans [0, 1].')
    if not isinstance(p.star_rays, (int, np.integer)) or p.star_rays < 2 or p.star_rays % 2:
        raise ValueError('star_rays doit être un entier pair >= 2.')
    if np.asarray(p.ghost_positions).ndim != 1:
        raise ValueError('ghost_positions doit être une séquence 1D.')
    h, w = image.shape[:2]
    position = ((w - 1) / 2, (h - 1) / 2) if p.position is None else p.position
    if len(position) != 2 or not np.isfinite(position).all():
        raise ValueError('position doit contenir deux coordonnées finies.')
    x0, y0 = position
    if not (0 <= x0 <= w - 1 and 0 <= y0 <= h - 1):
        raise ValueError('Position hors image.')
    return image, (float(x0), float(y0))


def _speckle(shape, p):
    if p.speckle_strength == 0:
        return np.ones(shape)
    rng = np.random.default_rng(p.seed)
    real, imag = [gaussian_filter(rng.standard_normal(shape), p.speckle_scale,
                                  mode='wrap') for _ in range(2)]
    field = real**2 + imag**2
    field /= max(field.mean(), np.finfo(float).tiny)
    return 1 - p.speckle_strength + p.speckle_strength * field


def _optical_effects(source, dx, dy, position, p):
    brightness = source.max(axis=-1)
    weight = np.maximum(brightness - p.bloom_threshold, 0)
    weight = np.divide(weight, brightness, out=np.zeros_like(weight), where=brightness > 0)
    bloom = p.bloom_strength * gaussian_filter(
        source * weight[..., None], (p.bloom_sigma, p.bloom_sigma, 0), mode='constant')
    flare = np.zeros_like(source)
    star = np.zeros_like(source)
    region = dx**2 + dy**2 <= 25
    level = float(brightness[region].mean())
    h, w = brightness.shape
    x0, y0 = position
    # Vecteur source -> centre, en coordonnées cartésiennes (y vers le haut).
    vx, vy = (w - 1) / 2 - x0, y0 - (h - 1) / 2
    for i, t in enumerate(p.ghost_positions):
        r = p.flare_radius * (1 + 0.3 * i)
        ghost = np.exp(-2 * (((dx - t * vx) / r)**2 + ((dy - t * vy) / r)**2))
        flare += (p.flare_strength * max(level - p.flare_threshold, 0)
                  / (1 + i) * ghost[..., None] * np.asarray(p.flare_color))
    n = p.star_rays // 2
    for i in range(n):
        angle = np.deg2rad(p.star_angle_deg) + i * np.pi / n
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        ray = np.exp(-abs(u) / p.star_length - 0.5 * (v / p.star_width)**2)
        star += (p.star_strength * max(level - p.star_threshold, 0)
                 / n * ray[..., None] * np.asarray(p.star_color))
    return bloom, flare, star


def render_laser(image, params=None, return_layers=False):
    """Rendu manuel. Aucun tirage de paramètres ; seul le speckle utilise seed."""
    p = LaserParams() if params is None else params
    image, position = _validate(image, p)
    h, w = image.shape[:2]
    Y, X = np.indices((h, w), dtype=float)
    dx, dy = X - position[0], position[1] - Y
    angle = np.deg2rad(p.angle_deg)
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    with np.errstate(over='ignore'):
        relative = np.exp(-2 * ((u / p.a)**2 + (v / p.b)**2)**p.p)
    envelope = p.intensity * relative
    t = np.clip((relative - p.white_threshold) / (1 - p.white_threshold), 0, 1)
    white = p.white_strength * t**2 * (3 - 2 * t) if p.intensity > 0 else np.zeros_like(t)
    smooth = np.clip(image + envelope[..., None] * np.asarray(p.color), 0, 1)
    smooth = smooth * (1 - white[..., None]) + white[..., None]
    speckle = _speckle((h, w), p)
    source = (smooth - image) * speckle[..., None]
    bloom, flare, star = _optical_effects(source, dx, dy, position, p)
    result = np.clip(image + source + bloom + flare + star, 0, 1)
    if not return_layers:
        return result
    return result, dict(envelope=envelope, speckle=speckle, white=white,
                        contribution=source, bloom=bloom, flare=flare, star=star)


def sample_latents(rng):
    """Quelques variables maîtresses. Plages provisoires à personnaliser."""
    return dict(
        intensity=float(rng.uniform(0.5, 1.5)),
        dispersion=float(rng.uniform(0.7, 1.5)),
        size_px=float(rng.uniform(7, 20)),
        roughness=float(rng.uniform(0.2, 0.8)),
        color=tuple(float(v) for v in rng.dirichlet([0.5, 3.0, 0.5])),
    )


def params_from_latents(latents, shape, rng, position=None):
    """STUB : centraliser ici TES relations et leur variabilité résiduelle.

    Les règles ci-dessous sont seulement une base exécutable, pas un modèle
    physique. Aucun lien vers une longueur d'onde n'est supposé.
    """
    h, w = shape[:2]
    level = latents['intensity']
    spread = latents['dispersion']
    size = latents['size_px']
    jitter = lambda: float(rng.uniform(0.9, 1.1))
    if position is None:
        position = (float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1)))
    color = np.asarray(latents['color'], dtype=float)
    color /= color.max()
    # Laser rules
    return LaserParams(
        position=position,
        a=size, b=size * float(rng.uniform(0.5, 1.0)),
        angle_deg=float(rng.uniform(0, 180)), p=float(rng.uniform(1, 3)),
        intensity=level, color=tuple(color),
        white_strength=float(np.clip((0.2 + 0.4 * level) * jitter(), 0, 1)),
        white_threshold=float(np.clip((1.15 - 0.4 * level) * jitter(), 0.5, 0.99)),
        speckle_strength=latents['roughness'], speckle_scale=1.2 * spread * jitter(),
        bloom_strength=0.5 * level * jitter(), bloom_sigma=8 * spread * jitter(),
        flare_strength=0.15 * level * jitter(), flare_radius=25 * spread * jitter(),
        flare_color=tuple(0.15 + 0.85 * color),
        star_strength=0.25 * level * jitter(), star_length=100 * spread * jitter(),
        star_angle_deg=float(rng.uniform(0, 180)), star_color=tuple(0.75 + 0.25 * color),
        seed=int(rng.integers(0, 2**32)),
    )


def generate_synthetic_laser(image, seed=None, position=None, return_layers=False):
    """Contrôleur : tirage -> relations -> rendu. Même seed = même résultat.

    Retourne (image, metadata), ou (image, metadata, layers).
    """
    rng = np.random.default_rng(seed)
    latents = sample_latents(rng)
    params = params_from_latents(latents, np.shape(image), rng, position)
    rendered = render_laser(image, params, return_layers=return_layers)
    metadata = dict(seed=seed, latents=latents, params=asdict(params))
    if return_layers:
        result, layers = rendered
        return result, metadata, layers
    return rendered, metadata
