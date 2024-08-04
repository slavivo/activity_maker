import configparser
import argparse
import logging
import openai
import asyncio
import base64
from utils import (
    chat_completion_request,
    RequestParams,
)

logging.basicConfig(level=logging.INFO)
config = configparser.ConfigParser()

# You need to create a config.ini file with your OpenAI API key and GPT model
config.read("src/config.ini")
OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the file containing the image",
        required=True,
    )
    args = parser.parse_args()

    client = openai.AsyncClient(api_key=OPENAI_KEY)

    with open("docs/prompt.txt", "r") as file:
        prompt = file.read()

    text_input = input("Enter the message for the chatbot for an activity: ")

    with open(args.file, "rb") as file:
        image_input = base64.b64encode(file.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_input}"},
                },
            ],
        },
    ]

    params = RequestParams(
        client=client,
        messages=messages,
        max_tokens=3000,
        temperature=0.5,
        top_p=0.5,
    )

    response = await chat_completion_request(params)

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
