DEFAULT_GENERATION_PARAMS = {"temperature": 0.7, "top_p": 0.9}

DEFAULT_FINAL_PROMPT = """# Role
As an expert with access to various documents and databases, you are asked to provide an answer to the following question based on the provided context and conversation.
# Chat History
{history}
# Context
{context}
# User Question
{question}
# Answer"""

DEFAULT_REWRITE_PROMPT = """As an expert on question rewriting, you are asked to rewrite the following question to make it easier to answer.
# Original Question
{question}
# Rewritten Question"""


DEFAULT_SUMMARIZE_PROMPT = """As an expert on summarization, you are asked to summarize the following text.
# Text
{text}"""

DEFAULT_MAKE_TITLE_PROMPT = """As an expert on title generation, you are asked to generate a title for the following text.
# Text
{text}"""
