"""
Fonctions utilitaires pour augmentation de données.
"""

import random
import cv2


def random_rot_flip(image, label):
    """
    Rotation 90°*k + flip aléatoire.
    """
    k = random.randint(0, 3)
    axis = random.randint(0, 1)

    # Rotation 90° * k
    if k == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif k == 2:
        image = cv2.rotate(image, cv2.ROTATE_180)
        label = cv2.rotate(label, cv2.ROTATE_180)
    elif k == 3:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)

    # Flip --> 1 = horizontal, 0 = vertical
    if axis == 1:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    else:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)

    return image, label


def random_rotate(image, label, max_angle=20):
    """
    Rotation aléatoire ±max_angle.
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    # Matrice de rotation autour du centre
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale=1.0)

    # Image : interpolation bicubique, bords reflétés
    image = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Masque : interpolation nearest, bords dupliqués
    label = cv2.warpAffine(
        label, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return image, label