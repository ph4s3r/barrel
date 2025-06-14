"""LLM module of handling user requests."""

import os
import json
from openai import AzureOpenAI

from credentials.secrets import secrets


class SuperPrompt:
    """Custom class to handle user questions."""

    endpoint = "https://glacius.openai.azure.com/"
    deployment = "gpt-4.1"

    subscription_key = secrets.llm_api_key
    api_version = "2024-12-01-preview"

    def __init__(self) -> None:
        """Instantiate Azure OpenAI LLM client."""
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )

    def process_prompt(self, prompt: str, context_text: str) -> list[str]:
        """Process user question."""

        super_prompt = f""" 
Here is a question: {prompt}

Please answer the question exclusively based on the documentation articles retrieved from our vector database, below encapsulated in the optional json block in the content key with a lot of metadata. If there is no relevant information in the documentation articles, please state to the user that "we don't have relevant enough information in our vector store to answer this specific question"

The relevance of the articles are expressed in a numerical value in matches[].score, the higher the value the more relevant the article to our question. Please consider all the metadata when trying to answer the user's question.

Make sure you don't use your own knowledge or anything to answer the question, ONLY the information in the documentation articles.

Make double check that you include all the json data in the response.

If there are articles to cite, then cite the relevant article(s) just saying "most relevant article(s):" verbatim in the following json format in a list (omit the keys where there was no value in the original retrieved documentation article json):
Please omit the empty json keys where there is no value
Please format your answer itself in markdown format (not the citation part)
[
    {{
    "citation": {{
        "www": "",
        "source_url": "",
        "web_url": "",
        "content": "", 		
        "source_format": "",
        "main_category": "",
        "sub_category": "",
        "markdown.data": {{ 	
        "main_header": "",
        "header_0": "",
        "header_1": "",
        "header_2": "",
        "header_3": "",
        }}
        "ms.headers": {{
        "title": "",
        "titleSuffix": "",
        "description": "",
        "ms.custom": "",
        "ms.date": "",
        "ms.service": "",
        "ms.topic": "",
        "intent": "",
        }},
    }}
    }}
]

Here are the documentation articles as a context to answer the question from:

{context_text}
"""

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": super_prompt,
                }
            ],
            temperature=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=self.deployment
        )

        print("User query:", prompt)
        print("RAG Answer:", response.choices[0].message.content)

        return response.choices[0].message.content