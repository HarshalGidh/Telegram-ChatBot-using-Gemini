from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
import textwrap
from IPython.display import Markdown

load_dotenv()
TOKEN = os.getenv("TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure generativeai with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize bot
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot)


class Reference:
    def __init__(self):
        self.response = ""


reference = Reference()


def clear_past():
    reference.response = ""


@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    A handler to clear the previous conversation and context.
    """
    clear_past()
    await message.reply("I've cleared the past conversation and context.")


@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """This handler receives messages with /start or  `/help `command

    Args:
        message (types.Message): description
    """
    await message.reply("Hi\nI am a Chat Bot! Created by Harshal Gidh!. How can i assist you?")


@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    A handler to display the help menu.
    """
    help_command = """
    Hi There, I'm a bot created by Harshal Gidh! Please follow these commands - 
    /start - to start the conversation
    /clear - to clear the past conversation and context.
    /help - to get this help menu.
    I hope this helps. :)
    """
    await message.reply(help_command)


def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text


@dispatcher.message_handler()
async def main_bot(message: types.Message):
    """
    A handler to process the user's input and generate a response using the Gemini model.
    """
    print(f">>> USER: \n\t{message.text}")

    response = get_gemini_response(message.text)
    
    print(f">>> gemini: \n\t{response}")
    
    await bot.send_message(chat_id=message.chat.id, text=response)


if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates=True)