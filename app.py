from multi_modal_rag import extract_pdf_elements, categorize_elements
from langchain.text_splitter import CharacterTextSplitter
from multi_modal_rag import generate_text_summaries , generate_img_summaries
from openai import OpenAI as OpenAI_vLLM
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_community.vectorstores import Chroma
from multi_modal_rag import create_multi_vector_retriever, multi_modal_rag_context_chain
from langchain.embeddings import HuggingFaceEmbeddings


#------------------------------------------------------------------------------------
# File path
folder_path = "./data/"
file_name = "gtm_benchmarks_2024.pdf"

#------------------------------------------------------------------------------------
# UNSTRUCTURED LIBRARY TEXT EXTRACTION

# Get elements
raw_pdf_elements = extract_pdf_elements(folder_path, file_name)

# Get text, tables
texts, tables = categorize_elements(raw_pdf_elements)

# Enforce a specific token size for texts
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000, chunk_overlap = 50
)
joined_texts = " ".join(texts)
texts_token = text_splitter.split_text(joined_texts)

print("No of Textual Chunks:", len(texts))
print("No of Table Elements:", len(tables))
print("No of Text Chunks after Tokenization:", len(texts_token))

#------------------------------------------------------------------------------------
# LLM initialization
llm_client = VLLMOpenAI(
    base_url = "http://localhost:8000/v1",
    api_key = "dummy",
    model_name = "llava-hf/llava-1.5-7b-hf",
    temperature = 1.0,
    max_tokens = 300
)

#------------------------------------------------------------------------------------
# Generate text summaries
# Get text, table summaries
text_summaries, table_summaries = generate_text_summaries(
    texts_token, tables, summarize_texts=False
)
print("No of Text Summaries:", len(text_summaries))
print("No of Table Summaries:", len(table_summaries))

#-------------------------------------------------------------------------------------

# Image summaries
img_base64_list, image_summaries = generate_img_summaries(folder_path)
assert len(img_base64_list) == len(image_summaries)

#-------------------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
# The vectorstore to use to index the summaries
vectorstore = Chroma(
    collection_name="mm_rag_vectorstore", embedding_function=embeddings, persist_directory="./chroma_db" 
)

#-------------------------------------------------------------------------------------

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)

#-------------------------------------------------------------------------------------
# Retrieve the relevant context including text and images
chain_multimodal_context = multi_modal_rag_context_chain(retriever_multi_vector_img)

#-------------------------------------------------------------------------------------
context = chain_multimodal_context.invoke(query)[0].content

chat_response = vlm_client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": context,
    }],
    stream=True
)

for chunk in chat_response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)