import logging
import logging.config
import os
import sqlite3

from joblib import Parallel, delayed

from src import utils
from src.data.misc import download_image, fetch_or_resume

logging.config.fileConfig(os.path.join(utils.SRC_PATH, "logging.conf"))

CARDS_DATABASE_PATH = os.path.join(
    utils.DATA_PATH, "raw", "card_database.sqlite")
IMAGES_DESTIN = os.path.join(utils.DATA_PATH, "raw", "card_images")

IMAGES_DESTINATION = os.path.join(utils.DATA_PATH, "raw", "card_images")


def get_cursor():
    def dict_factory(cursor, row):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    con = sqlite3.connect(CARDS_DATABASE_PATH)
    con.row_factory = dict_factory
    cur = con.cursor()
    return cur


def download_cards_db():
    logging.info("Downloading cards database...")
    DATA_URL = os.getenv("CARD_DATABASE_URL",
                         "https://mtgjson.com/api/v5/AllPrintings.sqlite")
    fetch_or_resume(DATA_URL, CARDS_DATABASE_PATH)
    logging.info("Download complete.")


def download_cards_images():

    logging.info("Downloading images for all of the card in the database...")

    if not os.path.exists(IMAGES_DESTINATION):
        os.makedirs(IMAGES_DESTINATION)

    cursor = get_cursor()
    cards = cursor.execute("SELECT id, scryfallId FROM cards").fetchall()

    # This was too slow
    # for idx, card in enumerate(cards):
    #     logging.info(f"Downloading image {idx}/{len(cards)}")
    #     download_card_image(card)

    Parallel(
        n_jobs=-1,
        prefer="threads"
    )(delayed(download_card_image)(idx, card) for (idx, card) in enumerate(cards))

    logging.info("Download complete.")


def download_card_image(idx, card):

    logging.info("Downloading image %s", idx)

    scryfallId = card['scryfallId']
    version = "normal"
    image_url = f"https://api.scryfall.com/cards/{scryfallId}?format=image&version={version}"

    card_filename = f"{card['id']}_{version}.jpg"
    card_path = os.path.join(IMAGES_DESTINATION, card_filename)

    if os.path.isfile(card_path):
        logging.info("The card image is already present. Skipping.")
    else:
        try:
            download_image(image_url, card_path)
        except Exception:
            pass


def download_templates():
    pass

if __name__ == "__main__":
    download_cards_db()
    download_cards_images()
    download_templates()