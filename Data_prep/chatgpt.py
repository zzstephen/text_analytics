
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
from loguru import logger
from prompt_request import prompt_request
import asyncio

# sample_ratio = 0.1


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


def main():

    raw_data = pd.read_csv("../../../data/complaints.csv")

    raw_data = raw_data.fillna('unknown')

    raw_data = raw_data.rename(columns={'Unnamed: 0': 'complaint_id'})

    # sample = raw_data.sample(frac=sample_ratio, random_state=123).reset_index(drop=True)

    with open("../../../../api_keys/openai_api.txt", "r") as password_file:  # Open in binary read mode

        password = password_file.read()


    raw_data['chatgpt_keyword_extraction'] = 'unknown'

    # labels = [p for p in sample['product'].unique()]


    with open("../../../../api_keys/openai_api.txt", "r") as password_file:  # Open in binary read mode

        password = password_file.read()

    openai = prompt_request(password)

    batch_size = 20000

    current_row = 0

    summary = []

    while current_row < raw_data.shape[0]:

        sample = raw_data.loc[current_row:min(raw_data.shape[0], current_row + batch_size), 'narrative']

        current_row += batch_size

        requests = sample.tolist()

        requests = ['Summarize the following context: '+r for r in requests]

        summary += openai.submit_requests(requests)

    raw_data['narrative_summary'] = summary

    sample.to_csv('../../../data/intermediate/raw_gpt_labeled.csv', index=False)




if __name__ == "__main__":
    main()