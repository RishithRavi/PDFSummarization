import os
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import langchain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob

from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFDirectoryLoader


_ = load_dotenv(find_dotenv())  # read local .env file
OpenAI.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0.2)
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in glob.glob(pdfs_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        print("Summary for: ", pdf_file)
        print(summary)
        print("\n")
        summaries.append(summary)
    
    return summaries

def custom_summary(pdf_folder, custom_prompt):
    summaries = []
    for pdf_file in glob.glob(pdf_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
        
    return summaries

# summaries = summarize_pdfs_from_folder("./pdfs")

loader = PyPDFDirectoryLoader("./pdfs/")

docs = loader.load()

# Create the vector store index
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What is the research and development expenses for year 2022 versus 2021?"

print(index.query(query))