import ssl
import os

# Bypass SSL verification for corporate proxy networks
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

import httpx
import requests
import requests.adapters
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper

requests.packages.urllib3.disable_warnings()

# Patch requests globally
original_send = requests.adapters.HTTPAdapter.send
def patched_send(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_send(self, *args, **kwargs)
requests.adapters.HTTPAdapter.send = patched_send

load_dotenv(find_dotenv())

def main():
    celeb = input("enter a celeb name :")
    
    wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    info = wiki.run(celeb)

    prompt_template = PromptTemplate(input_variables=["info", "celeb"], 
                                     template="""Summarise the following info about {celeb}:

{info}

Provide:
1. One good thing about {celeb}
2. One bad thing about {celeb}""")
    
    # ✅ Custom httpx client disables SSL check for Groq's API calls
    with httpx.Client(verify=False) as http_client:
        llm = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)  # or "llama3-70b-8192" for smarter model
        chain = prompt_template | llm | StrOutputParser()
        result = chain.invoke({"celeb": celeb, "info": info})
    print(result)

if __name__ == "__main__":
    main()
