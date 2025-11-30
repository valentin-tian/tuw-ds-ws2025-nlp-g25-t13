import os
import json  
import logging
import sys
import gzip
import os
import tarfile
import xml.etree.ElementTree as ET

# --- 1. CONFIGURATION & IMPORTS ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from typing import List, Any
import gradio as gr

# Insert your local path to the input and output folder
TAR_PATH = (
    r""
)
# e.g. r"C:\Users\user\Corpus\Conllu_output"
OUTPUT_DIR = (
    r""
)

# Google API key: https://aistudio.google.com/api-keys
api_key = ""

# Define where you want to save the indexed data
PERSIST_DIR = "./storage"

DOC_LIMIT = 10
SOFA_NAMESPACE = "{http:///uima/cas.ecore}Sofa"


# --- 2. STORAGE FOR Q&A ---
CHAT_LOG = []  # list for Chat log
LOG_FILE = "chat_history.json" # 

def save_log_to_disk():
    """Saves the current CHAT_LOG to a JSON file."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(CHAT_LOG, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(CHAT_LOG)} interactions to {LOG_FILE}")


# --- 3. SETUP THE MODELS ---
print("Loading Embedding Model")
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)


# --- 4. LOAD DATA---
def extract_raw_text(tar_path, doc_limit):
    """
    Iterates through the .xmi files in the tar archive and extracts the
    raw text content using ElementTree.
    """
    docs = []

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for m in tar.getmembers():
                name = m.name.lower()

                # Filter for XMI files
                if name.endswith((".xmi", ".xmi.gz", ".xmi.xmi.gz", ".xmi.xmi.gz")):
                    f = tar.extractfile(m)
                    if f is None:
                        continue

                    data = f.read()

                    if name.endswith(".gz"):
                        data = gzip.decompress(data)

                    # API for parsing and creating XML data
                    root = ET.fromstring(data)

                    # find sofa element with the text
                    sofa = root.find(f".//{SOFA_NAMESPACE}")

                    if sofa is not None:
                        text = sofa.get("sofaString")
                        if text:
                            # clean up unwanted characters before preprocessing
                            text = text.replace('\r\n', ' ').replace('\n', ' ').strip()
                            docs.append({
                                "id": os.path.basename(m.name).replace(".xmi", ""),
                                "text": text
                            })

                    # Stop after reaching the defined limit
                    if len(docs) >= doc_limit:
                        print(f"Reached document limit of {doc_limit}")
                        break

    except tarfile.TarError as e:
        print(f"Error reading tar file: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XMI content: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return []

    print(f"Successfully extracted raw text from {len(docs)} documents.")
    return docs

raw_data = extract_raw_text(TAR_PATH, DOC_LIMIT)
if not raw_data:
    print("ERROR: No documents were found! Check your TAR_PATH and XML Namespace.")
    sys.exit(1)


print("Converting dictionaries to Documents")
documents = []
for entry in raw_data:
    doc = Document(
        text=entry["text"],
        id_=entry["id"],
        metadata={
            "id_": entry["id"]}
    )
    documents.append(doc)


# --- 5. INDEXING ---
if os.path.exists(PERSIST_DIR):
    print("Loading index from storage")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Creating new index")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)


# --- 6. ADAPTER CLASS ---
class LlamaIndexToLangChainRetriever(BaseRetriever):
    llama_retriever: Any 
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LangChainDocument]:
        nodes = self.llama_retriever.retrieve(query)
        langchain_docs = []
        for node in nodes:
            langchain_docs.append(
                LangChainDocument(page_content=node.get_content(), metadata=node.metadata)
            )
        return langchain_docs


# --- 7. CONNECT TO LANGCHAIN ---
raw_retriever = index.as_retriever(similarity_top_k=5)
llama_retriever = LlamaIndexToLangChainRetriever(llama_retriever=raw_retriever)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

# --- 8. CREATE PROMPT ---
custom_template = """
    Du bist ein spezialisierter Assistent für Parlamentsprotokolle.
    Nutze nur die Informationen der Dokumente, um die Frage zu beantworten.
    Erfinde keine Fakten.

    Regeln:
    - Antworte präzise und kurz.
    - Wenn die Information fehlt, sage es direkt.
    - Strukturiere die Antwort wenn möglich mit Aufzählungspunkten.

    Kontext aus den Dokumenten:
    {context}

    Frage des Nutzers: {question}

    Antwort:
    """

PROMPT = PromptTemplate(
    template=custom_template, 
    input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=llama_retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


# --- 9. CHAT LOGIC  ---
def chat_logic(message, history):
    try:
        # 1. Run the Chain
        response = qa_chain.invoke({"query": message})
        answer_text = response['result']
        
        # 2. Extract Sources
        sources_list = []
        sources_text = "\n\n**Sources:**"
        for doc in response.get('source_documents', []):
            s_id = doc.metadata.get('id_', 'Unknown')
           
            sources_text += f"\n- ID: {s_id} "
            
            sources_list.append({"id": s_id})

        # 3. STORE THE DATA
        interaction_dict = {
            "question": message,
            "answer": answer_text,
            "sources": sources_list
        }
        
        CHAT_LOG.append(interaction_dict)
        save_log_to_disk()
            
        return answer_text + sources_text
        
    except Exception as e:
        return f"Error: {str(e)}"
   
# --- 10. LAUNCH UI ---
demo = gr.ChatInterface(
    fn=chat_logic, 
    title="German Parliament RAG",
    description="Questions are automatically saved to 'chat_history.json'.",
    examples=[""]
)

demo.launch(share=False)

