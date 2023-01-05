
import io

import PIL
from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel, validator


class ImageValidator(BaseModel):
    """Pydantic validator for images"""

    image: UploadFile

    @validator("image")
    def check_image(cls, image):
        """Checks that the input file is actually an image"""
        img = image.file.read()
        try:
            Image.open(io.BytesIO(img))
            image.file.close()
        except PIL.UnidentifiedImageError as exc:
            raise ValueError(
                "Image upload error, the file provided is not an image."
            ) from exc
        return img
