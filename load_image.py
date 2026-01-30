import numpy as np
from PIL import Image


def ft_load(path: str) -> list:
    """
    Loads an image from the specified file path and returns
    its pixel data as a NumPy array.

    Args:
        path (str): The file path to the image.

    Returns:
        list: A NumPy array representing the pixel data of the image.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        OSError: If the file is not a valid image format.

    Example:
        pixels = ft_load("example.jpg")
        print(pixels.shape)
    """
    try:
        img = Image.open(path)
        pixels = np.array(img)
        return pixels
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None
    except OSError:
        print(f"Error: The file '{path}' is not a valid image format.")
        return None
