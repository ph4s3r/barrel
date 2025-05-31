"""LLM module of handling user requests."""
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from credentials.secrets import secrets


class SuperPrompt:
    """Custom class to handle user questions."""
    def __init__(self) -> None:
        """Instantiate OpenAI LLM client."""
        self.model = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            openai_api_key=secrets.llm_api_key
        )

    def process_prompt(self, prompt: str, context_text: str) -> list[str]:
        """Process user question."""
        if context_text is not None:
            super_prompt = """ 
Here is a question: {question}

Please answer the question exclusively based on the documentation articles retrieved from our vector database, below encapsulated in the optional json block in the content key with a lot of metadata. If there is no relevant information in the documentation articles, please state to the user that "we don't have relevant enough information in our vector store to answer this specific question"

The relevance of the articles are expressed in a numerical value in matches[].score, the higher the value the more relevant the article to our question. Please consider all the metadata when trying to answer the user's question.

Make sure you don't use your own knowledge or anything to answer the question, ONLY the information in the documentation articles.

Make double check that you include all the json data in the response.

If there are articles to cite, then cite the relevant article(s) just saying "most relevant article(s):" verbatim in the following json format in a list (omit the keys where there was no value in the original retrieved documentation article json):
Please omit the empty json keys where there is no value
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

            super_prompt = ChatPromptTemplate.from_template(super_prompt)
            prompt = super_prompt.format(context_text=context_text, question=prompt)
            llm_output = self.model.invoke(prompt).content

            return [llm_output]
