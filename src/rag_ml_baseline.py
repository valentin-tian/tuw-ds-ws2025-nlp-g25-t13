import os
import json
import logging
import sys
import gzip
import tarfile
import xml.etree.ElementTree as ET
import random

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
from langchain_classic.chains import create_retrieval_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from typing import List, Any
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Insert your local path to the input and output folder
TAR_PATH = (
    r""
)

# Google API key: https://aistudio.google.com/api-keys
api_key = ""

# Define where you want to save the indexed data
PERSIST_DIR = "./storage"

DOC_LIMIT = 100
SOFA_NAMESPACE = "{http:///uima/cas.ecore}Sofa"

# --- 2. STORAGE FOR Q&A ---
CHAT_LOG = []  # list for Chat log
LOG_FILE = "chat_history.json" #

def save_log_to_disk():
    """Saves the current CHAT_LOG to a JSON file"""
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
    google_api_key=api_key,
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

    Frage des Nutzers: {input}

    Antwort:
    """

PROMPT = ChatPromptTemplate.from_template(custom_template)

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=PROMPT
)

qa_chain = create_retrieval_chain(
    retriever=llama_retriever,
    combine_docs_chain=document_chain
)


# --- 9. CHAT LOGIC  ---
def chat_logic(message, history):
    try:
        response = qa_chain.invoke({"input": message})
        answer_text = response["answer"]

        sources_list = []
        sources_text = "\n\n**Sources:**"

        for doc in response.get("context", []):
            s_id = doc.metadata.get("id_", "Unknown")
            sources_text += f"\n- ID: {s_id} "
            sources_list.append({"id": s_id})

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

NUM_TEST_QUESTIONS = 5
EVAL_OUTPUT_FILE = "eval_results.json"


def clean_json_string(json_str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    return json_str


def generate_qa(document_text):
    snippet = document_text[:2000]

    teacher_prompt = f"""
    Du erstellst Fragen für einen Test.
    Hier ist ein Auszug aus einem Parlamentsdokument:
    
    "{snippet}"
    
    Aufgabe:
    Erstelle EINE Frage, die nur mit diesem Text beantwortet werden kann.
    Gib auch die korrekte Antwort basierend auf dem Text.
    
    WICHTIG: Antworte NUR im validen JSON-Format:
    {{
        "question": "Deine Frage hier",
        "truth": "Deine Musterlösung hier"
    }}
    """

    try:
        response = llm.invoke(teacher_prompt).content
        data = json.loads(clean_json_string(response))
        return data
    except Exception as e:
        print(f"Error generating question: {e}")
        return None


def evaluate_rag_performance(question, truth, rag_answer):

    judge_prompt = f"""
    Du überprüfst die Antworten eines Tests. Vergleiche die Antwort eines Studenten mit der Musterlösung.
    
    Frage: {question}
    
    Musterlösung:
    {truth}
    
    Antwort des Studenten:
    {rag_answer}
    
    Bewerte:
    1. Ist die Antwort inhaltlich korrekt, ignoriere Unterschiede in der Formulierung? 
    2. Gib Punkte von 0 (Falsch) bis 5 (Perfekt).
    
    Antworte NUR im validen JSON-Format:
    {{
        "is_correct": true/false,
        "score": 0-5,
        "reasoning": "Kurze Erklärung warum"
    }}

    try:
        response = llm.invoke(judge_prompt).content
        data = json.loads(clean_json_string(response))
        return data
    except Exception as e:
        print(f"Error evaluation: {e}")
        return {"is_correct": False, "score": 0, "reasoning": "Judge crashed"}


def run_eval_pipeline():
    print(f"\n--- STARTING AUTOMATED EVALUATION ({NUM_TEST_QUESTIONS} Docs) ---\n")

    results = []

    if len(documents) < NUM_TEST_QUESTIONS:
        test_docs = documents
    else:
        test_docs = random.sample(documents, NUM_TEST_QUESTIONS)

    for i, doc in enumerate(test_docs):
        print(f"Processing Test {i+1}/{NUM_TEST_QUESTIONS}...")

        qa_pair = generate_qa(doc.text)
        if not qa_pair:
            continue

        question = qa_pair["question"]
        ground_truth = qa_pair["truth"]
        doc_id = doc.metadata.get("id_", "Unknown")

        print(f"  Generated Q: {question}")

        try:
            rag_response = qa_chain.invoke({"input": question})
            rag_answer = rag_response["answer"]

            retrieved_ids = [
                d.metadata.get("id_", "")
                for d in rag_response.get("context", [])
            ]
            found_source = doc_id in retrieved_ids

        except Exception as e:
            rag_answer = f"ERROR: {str(e)}"
            found_source = False

        eval_result = evaluate_rag_performance(question, ground_truth, rag_answer)

        report_card = {
            "id": doc_id,
            "question": question,
            "ground_truth": ground_truth,
            "rag_answer": rag_answer,
            "score": eval_result.get("score", 0),
            "is_correct": eval_result.get("is_correct", False),
            "reasoning": eval_result.get("reasoning", ""),
            "retrieval_success": found_source,
        }

        results.append(report_card)
        print(f"  Score: {report_card['score']}/5 | Correct: {report_card['is_correct']}")
        print("-" * 30)

    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"\nAverage Score: {avg_score:.1f}/5")
        print(f"Results saved to {EVAL_OUTPUT_FILE}")

run_eval_pipeline()
