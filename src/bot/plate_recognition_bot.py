"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import os

from telegram import ForceReply, Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)


DEFAULT_MESSAGE = "Send me the picture of a car plate and i'll try to transribe it for you."
TOKEN = os.getenv("BOT_TOKEN", "5608820637:AAG7cHLFOafgcVqTGS5QDVdebhCEGm-CJjk")
PLATE_RECOGNITION_APP_SCHEMA = os.get_env("PLATE_RECOGNITION_APP_SCHEMA", "http")
PLATE_RECOGNITION_APP_HOST = os.get_env("PLATE_RECOGNITION_APP_HOST", "localhost")
PLATE_RECOGNITION_APP_PORT = os.get_env("PLATE_RECOGNITION_APP_PORT", "8080")

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


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(DEFAULT_MESSAGE)


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Thanks for the pic!",
        reply_to_message_id=update.message.id
    )

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    
    # on image make a request for the plate prediction
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()
