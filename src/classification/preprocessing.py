from typing import Optional

from PIL import Image


def square_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    margin: float,
    max_side: Optional[int] = None,
) -> Image.Image:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + 2 * margin) / 2
    size = max(1, round(2 * half))
    left, top = round(cx - half), round(cy - half)
    # fora da imagem, o PIL preenche com preto
    crop = image.crop((left, top, left + size, top + size))
    if max_side and size > max_side:
        crop = crop.resize((max_side, max_side), Image.Resampling.LANCZOS)
    return crop