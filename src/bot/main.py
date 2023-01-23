import logging
import os

from telegram import ForceReply, Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from src.bot import telegram_helper
from src.bot.api import plate_recognition_api
from src.bot.monitoring import (increase_messages_counter,
                                increase_photos_counter, start_monitoring)

DEFAULT_MESSAGE = "Send me the picture of a car plate and i'll try to transcribe it for you."
TOKEN = os.getenv("TELEGRAM_API_TOKEN", "5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk")

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"""
        Hi {user.mention_html()}!
        {DEFAULT_MESSAGE}
        """,
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def text_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    increase_messages_counter()
    await update.message.reply_text(DEFAULT_MESSAGE)


async def photo_handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logging.info("Received a photo message")
    increase_photos_counter()
    try:
        await update.message.reply_text(
            "Processing your request...",
            reply_to_message_id=update.message.id
        )
        photo_file = await update.message.photo[-1].get_file()
        photo = await telegram_helper.get_photo(photo_file)
        logging.info("Requesting the plate to the plate recognition api...")
        response = plate_recognition_api.get_plate_text(photo)
        await update.message.reply_text(
            response.json()["data"]["plate"],
            reply_to_message_id=update.message.id
        )
        logging.info("Done.")
    except Exception as exc:
        logging.exception(exc)
        await update.message.reply_text(
            "There was an error with your request. Please try again later.",
            reply_to_message_id=update.message.id
        )

def start_bot() -> None:

    """Start the bot."""
    logging.info("Starting the telegram bot")
    # Create the Application and pass it your bot's token.
    application = Application.builder()\
        .token(TOKEN)\
        .build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # on image make a request for the plate prediction
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(close_loop=False)
    logging.info("Telegram bot started")

if __name__ == "__main__":
    start_monitoring()
    start_bot()
