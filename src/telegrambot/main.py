import logging
import cv2
import sys

from processor.Processor import Processor

from flask import Flask
from flask import request
from flask import Response
import shutil
import requests

sys.path.append('.')

app = Flask(__name__)

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def parse_message(message):
    print("message-->", message)
    chat_id = message['message']['chat']['id']
    txt = message['message']['text']
    print("chat_id-->", chat_id)
    print("txt-->", txt)
    return chat_id, txt


def tel_send_message(chat_id, text):
    url = f'https://api.telegram.org/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text
    }

    r = requests.post(url, json=payload)
    return r


def tel_send_image(chat_id, photo, caption=None):

    url = f'https://api.telegram.org/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/sendPhoto'
    files = {'photo': photo}
    data = {'chat_id': chat_id, 'caption': caption}

    r = requests.post(url, files=files, data=data)
    return r


def tel_parse_message(message):
    print("message-->", message)
    try:
        chat_id = message['message']['chat']['id']
        txt = message['message'].get('text')
        photo = message['message'].get('photo')

        print("\nchat_id-->", chat_id)
        print("\ntxt-->", txt)
        print("\nphoto-->", photo)

        return chat_id, txt, photo
    except Exception as e:
        logging.exception(e)


def download_photo(photo):
    file_id = photo[-1]['file_id']

    url = f'https://api.telegram.org/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/getFile?file_id={file_id}'
    file_path = requests.get(url).json()['result']['file_path']
    print(file_path)

    url = f'https://api.telegram.org/file/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/{file_path}'
    image = requests.get(url, stream=True)

    path = 'tmp/img.png'
    with open(path, 'wb') as out_file:
        shutil.copyfileobj(image.raw, out_file)
    return path


@app.route('/', methods=['POST'])
def index():
    msg = request.get_json()
    try:
        logger.info('Received a message')
        chat_id, txt, photo = tel_parse_message(msg)
        # tel_send_message(chat_id, "Thanks for the message")

        if photo:
            logger.info('It is a photo')
            # tel_send_message(chat_id, "I'm retrieving the photo...")
            logger.info("I'm retrieving the photo...")
            photo_path = download_photo(photo)
            # tel_send_message(chat_id, "I've got the photo!")
            logger.info("I've got the photo!")

            images, bboxes = processor.get_plate_detection_prediction(cv2.imread(photo_path))

            plates = []
            for (image, bbox) in zip(images, bboxes):
                cv2.imwrite(photo_path, image)
                tel_send_image(chat_id, photo=open(photo_path, 'rb'))

                plate = processor.get_ocr_prediction(cv2.imread(photo_path), bbox)
                plates.append(plate)

            logger.info(f'Plates before post processing: {", ".join(plates)}')
            plates = processor.post_process_plates(plates)
            logger.info(f'Plates after post processing: {", ".join(plates)}')

            if len(plates) == 0:
                tel_send_message(chat_id, f"No plates found, please try again with another image")
            else:
                message = f"Possible plates: {', '.join(plates)}"
                tel_send_message(chat_id, f"Possible plates: {', '.join(plates)}")


        else:
            tel_send_message(chat_id, "Send a picture of a plate")

    except Exception as e:
        logging.exception(e)

    return Response('ok', status=200)


if __name__ == '__main__':
    processor = Processor()
    app.run(debug=True)
    logger.info('Server running.')
