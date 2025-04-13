"""LLM module of handling user requests."""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from credentials.secrets import secrets


class SuperPrompt:
    """Custom class to handle user questions."""
    def __init__(self) -> None:
        """Instantiate OpenAI LLM client."""
        self.model = ChatOpenAI(
            model="gpt-4o",
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
answer the question based only on the following context from a vector store: the bigger the similarity score the more relevant the content is.
{context_text}
answer the question based on the above context: {question}.
provide a detailed answer.
don't give information not mentioned in the context information.
Please cite the relevant document files in your answer from the context with full reference: metadata source
(the full path of the source is really important: it indicates the document location in the documentation structure / category),
metadata header (which was the title/header of the text block in the original markdown documentation) and similarity score with
8 decimal point precision (no flooring) e.g. 0.84780987 and also quote an excerpt of the original context as-is. Please format your answer
as a markdown where citations are clearly distinguished. If the context with the highest similarity score is not relevant, please double
check and explain why it is not relevant. So please try to make sure that the most similar contexts are verified. However, if the contexts
are not containing anything relevant, then do not explain what are the most relevant sources, just provide your answer, obviously
don't need to mention anything about citations. If you are sure there are no relevant context, please use your own knowledge.
            """
            super_prompt = ChatPromptTemplate.from_template(super_prompt)
            prompt = super_prompt.format(context_text=context_text, question=prompt)
            llm_output = self.model.invoke(prompt).content

            return [llm_output]

        fallback_prompt = """
Please state that
'We don't have information about this in our vector store'
verbatim. Do not add any other information to the response.
"""
        super_prompt = ChatPromptTemplate.from_template(fallback_prompt)
        prompt = super_prompt.format(question=prompt)
        llm_output = self.model.invoke(prompt).content

        return [llm_output]
