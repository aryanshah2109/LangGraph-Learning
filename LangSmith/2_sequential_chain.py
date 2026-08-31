from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set project name for langsmith
os.environ['LANGCHAIN_PROJECT'] = "new_project_learning"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(
    model="openai/gpt-oss-20b",
    max_tokens=1024
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# Log run name, metadata and tags
config = {
    "run_name": "sequential_chain",
    "tags": ["llm-app", "report-generation"],
    "metadata": {
        "model_name": "openai/gpt-oss-20b",
        "model_max_tokens": 1024,
        "parser": "stroutputparser"
    }
}

result = chain.invoke({'topic': 'Rise of AI in India'}, config=config)

print(result)
