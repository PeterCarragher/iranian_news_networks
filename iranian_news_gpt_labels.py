from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import pandas as pd
import tqdm
import os

# OPENAI Key
OPENAI_KEY = os.getenv("OPENAI_KEY")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI agent who labels iranian news domains as supportive of the government, reformist, anti-government, or neutral."),
    ("human", "iranfocus.com"),
    ("ai", "Anti-government"),    
    ("human", "isna.ir"),
    ("ai", "Supportive"),
    ("human", "sharghdaily.ir"),
    ("ai", "Reformist"),    
    ("human", "{news_domain}"),
])

chain = LLMChain(
    llm=ChatOpenAI(openai_api_key=OPENAI_KEY, temperature=0.01, model="gpt-4"),
    prompt=chat_prompt,
)

labels = []
domains = pd.read_csv('iranian_news_urls.csv')

for domain in tqdm.tqdm(domains['url'].to_list()):
    labels.append(chain.run(domain))
    print(f"{domain}: {labels[-1]}")
domains['gpt_label'] = labels
domains.to_csv('iranian_news_urls_labeled.csv', index=False)