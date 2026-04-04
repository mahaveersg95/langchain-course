import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # go up to parent

from utils.ssl_fix import apply_ssl_fix, get_http_client   
apply_ssl_fix()                                                

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper

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
    with get_http_client() as http_client:
        llm = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)  # or "llama3-70b-8192" for smarter model
        chain = prompt_template | llm | StrOutputParser()
        result = chain.invoke({"celeb": celeb, "info": info})
    print(result)

if __name__ == "__main__":
    main()
