# Telegram-ChatBot-using-Gemini

## Overview
Chatbots serve various purposes, including customer service, information retrieval, and entertainment. They can streamline interactions, provide quick responses, and enhance user experiences across different platforms and industries. As technology advances, chatbots continue to evolve, offering increasingly personalized and contextually relevant interactions tailored to individual user needs.

Empowering the Power of ChatBots , this Telegram ChatBot called telebot is integrated with GEMINI-Pro that handles users queries efficiently and gives appropriate response.

## Prerequisites

To follow this tutorial, you will need:

- Python 3.10 or higher
- A Telegram account and a smartphone
- A Gemini-Pro API key

# How to use ?

-Run the app on any IDE then search the telegram bot's username on Telegram and use the chatbot.

-For example, my chatbot is named telebot so when i run my application, the chatbot gets activated and can be used by anyone.

### Start the Chat

![Screenshot 2024-04-21 184234](https://github.com/HarshalGidh/Telegram-ChatBot-using-Gemini/assets/126465410/4b3352c7-2bc8-42f7-9a38-371153fe39ea)

### Remembers the chat history and context 

![Screenshot 2024-04-21 184321](https://github.com/HarshalGidh/Telegram-ChatBot-using-Gemini/assets/126465410/128ce81c-d845-42a1-acc0-58dc7baccb03)

### Use Custom Commands to give custom Messages

![Screenshot 2024-04-21 184411](https://github.com/HarshalGidh/Telegram-ChatBot-using-Gemini/assets/126465410/b2e4ec3b-bbe3-4809-8d74-726731729a88)

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/HarshalGidh/Telegram-ChatBot-using-Gemini
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n telebot python=3.10 -y
```

```bash
conda activate telebot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### AIogram docs
https://docs.aiogram.dev/en/latest/


## Telegram Setup:

1. Search for botfather
2. /newbot
   - mybot88
   - mybot88_bot

   * Now click on url to access the bot
   * Make sure you collect the access token


### Add in .env
Store your Telgram Bot Token and GEMINI API key in .env file
```ini
GEMINI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_BOT_TOKEN=xxxxxxxxxx:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Start the chat with your own custom chatbot :D
