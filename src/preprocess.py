import os
import tarfile
import stanza
from stanza.utils.conll import CoNLL
import xml.etree.ElementTree as ET
import gzip

# --- Configuration ---
TAR_PATH = r"C:/Users/Diara/Desktop/Data_Science/NLP/Schweiz.tar"
OUTPUT_DIR = r"C:/Users/Diara/Desktop/Data_Science/NLP/conllu_output"
DOC_LIMIT = 5 # sample subset to keep manageable

# Step 1: Environment setup
def setup_environment():
    """Initializes Stanza and downloads the German model."""
    print("--- 1. Environment Setup ---")
    
    try:
        stanza.download('de', verbose=False)
        nlp_de = stanza.Pipeline('de', processors='tokenize,lemma,pos', verbose=False)
        return nlp_de
    except Exception as e:
        print(f"Error setting up Stanza: {e}")
        return None

# Step 2: extract xmi files from tarfile and extract raw text
def extract_raw_text(tar_path, doc_limit):
    """
    Iterates through the .xmi files in the tar archive and extracts the 
    raw text content using ElementTree.
    """   
    docs = []
    
    # Define the namespace for the Sofa element in UIMA XMI
    SOFA_NAMESPACE = "{http:///uima/cas.ecore}Sofa"

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

# Step 3: preprocess text and save as CoNLL-U file 
def process_and_save_conllu(documents, nlp_pipeline):
    """Preprocess raw documents and save the output in CoNLL-U format"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    processed_count = 0
    
    for i, doc_data in enumerate(documents):
        document_id = doc_data['id']
        text = doc_data['text']
        output_filepath = os.path.join(OUTPUT_DIR, f"{document_id}.conllu")
        
        try:
            # Stanza pipeline includes tokenization, sentence segmentation, lemma, and part-of-speech (POS) tagging
            doc = nlp_pipeline(text)
            
            # write the Stanza Doc object
            CoNLL.write_doc2conll(doc, output_filepath)
            
            processed_count += 1
            print(f"Processed {processed_count}/{len(documents)} documents. Last saved: {output_filepath}")
        
        except Exception as e:
            print(f"Error processing document {document_id}: {type(e).__name__} - {str(e)[:50]}...")
            
# Step 5: inspect the output document
def verify_output(documents):
    """Read and print a sample of the generated CoNLL-U file for inspection"""
    if not docs:
        print(f"No documents were extracted.")
        return
    

    # id of the first extracted document
    first_doc_id = documents[0]['id']
    sample_file = os.path.join(OUTPUT_DIR, f"{first_doc_id}.conllu")
    
    if os.path.exists(sample_file):
        print(f"Inspecting content of sample file ({first_doc_id}.conllu):\n")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(''.join(lines[:20])) 
    else:
        print("Verification failed: CoNLL-U file not found")



def main():
    """Main execution function."""
    # Step 1: environment setup
    nlp_de = setup_environment()
    

    # Step 2: extract raw docuemtns
    raw_documents = extract_raw_text(TAR_PATH, DOC_LIMIT)
    

    # Step 3: preprocess text with Stanza pipeline and save as CoNLL-U file 
    process_and_save_conllu(raw_documents, nlp_de)

    # Verify the output format
    verify_output(raw_documents)

if __name__ == "__main__":
    main()


