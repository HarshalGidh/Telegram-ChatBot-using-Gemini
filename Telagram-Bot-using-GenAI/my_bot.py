from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai

load_dotenv()
TOKEN = os.getenv("TOKEN")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect with GEMINI
model = genai.GenerativeModel("gemini-pro")
print("Gemini Pro initialized successfully.")

# Initialize bot
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot)

class Reference:
  def __init__(self) -> None:
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


@dispatcher.message_handler(commands=['start', 'help'])
async def welcome(message: types.Message):
  """
  This handler receives messages with `/start` or `/help` command

  Args:
      message (types.Message): _description_
  """
  help_command = """
  Hi There, I'm a bot created by aHarshal Gidh! Please follow these commands - 
  /start - to start the conversation
  /clear - to clear the past conversation and context.
  /help - to get this help menu.
  I hope this helps. :)
  """
  await message.reply(help_command)


@dispatcher.message_handler()
async def main_bot(message: types.Message):
  """
  A handler to process the user's input and generate a response using the Generative AI model.
  """

  print(f">>> USER: \n\t{message.text}")

  # Prepare messages with user input
  messages = [{"text": reference.response}]  # Previous response from assistant
  messages.append({"text": message.text})  # User's current message

  # Generate response using Gemini
  response = model.generate_content(messages)

  # Update reference with latest response
  reference.response = response.candidates[0].content

  print(f">>> Gemini: \n\t{reference.response}")

  # Check if message has a chat attribute and access chat ID accordingly
  if hasattr(message, 'chat'):
    chat_id = message.chat.id
    await bot.send_message(chat_id=chat_id, text=reference.response)
  else:
    print("Error: message object missing chat attribute")


if __name__ == "__main__":
  executor.start_polling(dispatcher, skip_updates=True)
