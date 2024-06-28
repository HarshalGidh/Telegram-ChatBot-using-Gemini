from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
import textwrap
from IPython.display import Markdown

# from aiogram.methods import GetFile

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


## Get File 

## Create File :
def create_file(folder_name="data", file_name="info.txt"):
    try:
        # Create the data folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully.")

        # Define the file path
        file_path = os.path.join(folder_name, file_name)

        # Create the info.txt file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("This is a sample text file created using Python!\n")
                file.write("You can add more content to this file.\n")
            print(f"File '{file_name}' created successfully in '{folder_name}' folder.")
        else:
            print(f"File '{file_name}' already exists in '{folder_name}' folder.")

        # Return the file path for reference
        return file_path

    except Exception as e:
        print(f"An error occurred while creating the file: {e}")
        return None

@dispatcher.message_handler(content_types=['document'])
async def get_file(message: types.Message):
    """
    A handler to process the user's input and generate a response using the Gemini model.
    """
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "data/info.txt")

    # Define the path to the file
    file_path = create_file()

    # Verify the file creation by reading its content
    if file_path:
        try:
            # Attempt to read the file content using different encodings
            encodings_to_try = ["utf-8", "latin-1", "iso-8859-1"]
            file_content = None

            for encoding in encodings_to_try:
                try:
                    with open(file_path, "r", encoding=encoding) as file:
                        file_content = file.read()
                        print(f"Content of '{os.path.basename(file_path)}' (decoded with {encoding}):\n{file_content}")
                        break  # Stop trying encodings if one succeeds

                except UnicodeDecodeError:
                    print(f"Failed to decode '{os.path.basename(file_path)}' with {encoding} encoding")

            if file_content:
                # Process file content using Gemini model
                response = get_gemini_response(file_content)
                print(f">>> gemini: \n\t{response}")

                # Send response to user
                await bot.send_message(chat_id=message.chat.id, text=response)

        except FileNotFoundError:
            print(f"Error: File '{os.path.basename(file_path)}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


# @dispatcher.message_handler(content_types=['document'])
# async def get_file(message: types.Message):
#     """
#     A handler to process the user's input and generate a response using the Gemini model.
#     """
#     file_id = message.document.file_id
#     file = await bot.get_file(file_id)
#     file_path = file.file_path
#     await bot.download_file(file_path, "data/info.txt") 

#     print(f">>> gemini: \n\t file uploaded successfully")
#     print(f">>> USER: \n\t{message.text}")

#         # Define the path to the file
#     filePath = create_file()
#     # Verify the file creation by reading its content
#     if filePath:
#         try:
#             with open(filePath, "r",encoding="utf-8") as file:
#                 file_content = file.read()
#                 print(f"Content of '{os.path.basename(filePath)}':\n{file_content}")
#         except FileNotFoundError:
#             print(f"Error: File '{os.path.basename(filePath)}' not found.")
#         except Exception as e:
#             print(f"An error occurred: {e}")

#     # Initialize an empty variable to store the file content
#     # file_content = ""

#     # Display the content of the file if it was read successfully
#     if file_content:
#         print("Content of 'info.txt':")
#         print(file_content)

#         response = get_gemini_response(file_content)

#         print(f">>> gemini: \n\t{response}")
        
#         await bot.send_message(chat_id=message.chat.id, text=response) 

@dispatcher.message_handler(content_types=['document'])
async def chat_bot(message: types.Message):
    """
    A handler to process the user's file input and generate a response using the Gemini model.
    """
    print(f">>> USER: \n\t{message.text}")
    get_file(message.filename)
    response = get_gemini_response("data/info.txt")

    print(f">>> gemini: \n\t{response}")
    
    await bot.send_message(chat_id=message.chat.id, text=response)

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