# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import Document Schema
from langchain.schema import Document
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Create and load PDF Loader
file_name = "/Users/xhagrg/Downloads/IGARSS_Blockchain.pdf"
loader = PyPDFLoader(file_name)

# Split pages from pdf
pages = loader.load_and_split()
updated_pages = list()

split_number = 0
page_splits = 5

for index, page in enumerate(pages):
    single_page_length = int(len(page.page_content) / page_splits)
    for split in range(1, page_splits):
        updated_pages.append(Document(page_content=page.page_content[
                ((split - 1) * single_page_length):(split * single_page_length)
            ],
            metadata={
                'source': file_name,
                'page': split_number
            })
        )
        split_number += 1
# Load documents into vector database aka ChromaDB
collection_name = 'document'
store = Chroma.from_documents(updated_pages, collection_name=collection_name)

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name=collection_name,
    description="a document as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('Question and Answer based on a document')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander
    with st.expander('Context'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Write out the first
        st.write(search[0][0].page_content)
