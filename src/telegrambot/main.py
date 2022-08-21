import logging

from flask import Flask

app = Flask(__name__)

from flask import Flask
from flask import request
from flask import Response
import shutil
import requests


app = Flask(__name__)


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


def tel_send_image(chat_id):
    url = f'https://api.telegram.org/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/sendPhoto'
    payload = {
        'chat_id': chat_id,
        'photo': "https://raw.githubusercontent.com/fbsamples/original-coast-clothing/main/public/styles/male-work.jpg",
        'caption': "This is a sample image"
    }

    r = requests.post(url, json=payload)
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

    file_id = photo[-2]['file_id']

    url = f'https://api.telegram.org/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/getFile?file_id={file_id}'
    file_path = requests.get(url).json()['result']['file_path']
    print(file_path)

    url = f'https://api.telegram.org/file/bot5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk/{file_path}'
    image = requests.get(url, stream=True)
    with open('img.png', 'wb') as out_file:
        shutil.copyfileobj(image.raw, out_file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        try:
            chat_id, txt, photo = tel_parse_message(msg)

            if photo:
                tel_send_message(chat_id, "I'm retrieving the photo...")
                download_photo(photo)
                tel_send_message(chat_id, "I got the photo!")


        except Exception as e:
            logging.exception(e)

        return Response('ok', status=200)
    else:
        return "<h1>Welcome!</h1>"


if __name__ == '__main__':
    app.run(debug=True)
