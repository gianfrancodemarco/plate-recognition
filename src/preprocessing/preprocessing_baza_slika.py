import os
import shutil

# from src.constants import BAZA_SLIKA_PATH

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
BAZA_SLIKA_PATH = f'{dir_path}/../../datasets/baza_slika'
OUTPUT_PATH = f'{dir_path}/../../resources/baza_slika'

if os.path.exists(OUTPUT_PATH):
    try:
        shutil.rmtree(OUTPUT_PATH)
    except OSError as e:
        print("Error: %s : %s" % (OUTPUT_PATH, e.strerror))

os.mkdir(OUTPUT_PATH)

idx = 0
for f in os.scandir(BAZA_SLIKA_PATH):
    if f.is_dir():
        for image in os.listdir(f.path):
            shutil.copy(f"{f.path}/{image}", f"{OUTPUT_PATH}/{idx}.jpg")
            idx += 1