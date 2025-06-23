
import os
from openai import *
import pandas as pd
import numpy as np
import pdb
import time
import sys
sys.path.append('../../../../infrastructure/tools')
from utilities import *
from plotting import *
import asyncio
from openai import AsyncOpenAI
from loguru import logger


sample_ratio = 0.1


async def generate_response(request, OPENAI_API_KEY):
    """Sends a prompt to OpenAI's ChatGPT API and returns the response."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Initialize AsyncOpenAI client
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or other suitable model
            messages=[{"role": "user", "content": request}]
        )
        return response.choices[0].message.content  # Extract the response content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


async def main():

    raw_data = pd.read_csv("../../../data/complaints.csv")

    raw_data = raw_data.fillna('unknown')

    raw_data = raw_data.rename(columns={'Unnamed: 0': 'complaint_id'})

    sample = raw_data.sample(frac=sample_ratio, random_state=123).reset_index(drop=True)

    with open("../../../../api_keys/openai_api.txt", "r") as password_file:  # Open in binary read mode

        password = password_file.read()


    sample['chatgpt_keyword_extraction'] = 'unknown'

    # labels = [p for p in sample['product'].unique()]

    messages = []

    for i in range(sample.shape[0]):

        request = 'Summarize the following context? context:' + f"{sample.at[i, 'narrative']}"

        messages.append(request)

    logger.info(f'Begin OpenAi requests: # of requests {sample.shape[0]}')

    tasks = [generate_response(p, password) for p in messages]

    responses = await asyncio.gather(*tasks)


    for i, response in enumerate(responses):
        if response:
            sample.at[i, 'chatgpt_keyword_extraction'] = response

    logger.info(f'Received returned OpenAi requests: # of requests {sample.shape[0]}')

    sample.to_csv('../../../data/intermediate/sample_gpt_labeled.csv', index=False)


if __name__ == "__main__":
    asyncio.run(main())