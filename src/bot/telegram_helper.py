from io import BytesIO


async def get_photo(photo_file) -> BytesIO:
    photo = BytesIO()
    await photo_file.download_to_memory(photo)
    photo.seek(0)
    return photo
