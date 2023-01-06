import io

import PIL
from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel, validator


class PredictImageDTO(BaseModel):
    
    image_file: UploadFile

    @validator("image_file")
    def check_image(cls, image_file) -> Image:
        """Checks that the input file is actually an image"""

        try:
            image_bytes = image_file.file.read()
            image_file.close()
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except PIL.UnidentifiedImageError as exc:
            raise ValueError(
                "Image upload error, the file provided is not an image."
            ) from exc
