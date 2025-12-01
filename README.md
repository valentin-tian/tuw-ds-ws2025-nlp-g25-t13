# 194.093 Natural Language Processing and Information Extraction (WS2025) - Project
## Topic 13 - Retrieval-augmented Generation (RAG) for German official documents

### Contributors (Group 25)
- [Valentin Tian](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=205694)
- [Diara Rinnerbauer](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=211007)
- [Stefan Sick](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=129430)
<!--- [Shakoor Muhammad Nouman](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=211748)-->

### Instructor
- [Gabor Recski](https://tiss.tuwien.ac.at/person/336863.html)

### Timeline
| Deadline | Activity | Description |
|:----------|:--------:|------------:|
| 02/11/2025 | Milestone 1 | CoNLL-U Preprocessing |
| 30/11/2025 | Milestone 2 | Baseline Evaluation |
| 19/12/2025 | Review Meeting | Status Update |
| 16/01/2026 | Final Presentation | Presentation and Feedback |
| 25/01/2026 | Final Submission | Final Solution |

## Project Structure
### Milestone 1
#### Deliverables
- [Preprocessing Code](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/preprocess.py)
- [Raw Data](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/tree/main/data/raw)
- [CoNLL-U Files](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/tree/main/data/processed)

#### Process Documentation
1. Data Extraction
   - All ```.xmi``` files were read from the ```Schweiz.tar``` archive - [25 - Nationlarat (CH)](http://lrec2022.gerparcor.texttechnologylab.org/).
   - The [Python Script](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/preprocess.py) extracted the raw text from the ```<Sofa>``` element of each ```.xmi``` document using ```xml.etree.ElementTree```.
2. NLP Preprocessing
   - Text was processed using the Stanza German language pipeline (processors: ```tokenize```, ```mwt```, ```pos```, ```lemma```).
   - The tokenized, lemmatized text with part of speach tags was created. The ```mwt``` processor is responsible for expanding multiword tokens and recovering their full structure from short forms.
3. Output
   - The documents were saved in the CoNLL-U format using ```stanza.utils.conll.CoNLL.write_doc2conll()```.
   - Each file was saved with the same name ID as an original one (e.g. ```20000324.xmi.xmi``` -> ```20000324.gz.conllu```).
   - All result files can be find at the [Google Drive CoNLL-U](https://drive.google.com/drive/folders/1lk53aSkx_aZ6wANKXdhMattQz2hUac7B?usp=sharing) link.
4. Verification
   - A random sample of the generated files was manually inspected.
5. Result
   - A total of 100 ```.xmi``` documents were succesfully converted into CoNLL-U format.
 
### Milestone 2
#### Deliverables
   - [Rule-based Embeddings (TF-IDF) RAG Baseline Code](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/rag_rb_baseline.py)
   - [Multilingual E5 Text Embeddings Model RAG Baseline Code](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/rag_ml_baseline.py)

#### Process Documentation

For the Milestone 2, two baseline RAG systems were implemented, one was implemented using TF-IDF aproach for the embeddings processing and another one was implemented using pre-trained Multilingual E5 Text Embeddings Model from Hugging Face.

- First, we loaded a corpus of parliamentary documents from a tar archive. The raw text is extracted from the XMI files from a special Sofa element. For each file a structure with an id and text was created. Then those dictionaries were converted into Document objects from LlamaIndex.
- Next, a search layer was buit on the top of the documents. In the rule-based baseline we have used TF-IDF aproach to build vectors and then the indices were created. In the machine learning aproach we have used Multilingual E5 Text Embeddings Model to create embeddings.
- Then we created a RAG chain: based on the user`s request, the system is trying to find relevant documents, retrive a correct context and put the context and the question in the predifined promt, and send everything to the LLM (what is Gemini 2.5 flash in our case).
- With a Gradio UI we created a web-interface for the chatting. While using this chat interface, the system logs all interactions in a separate JSON file.
- In the end we evaluated our system using automatic evaluation approach: first, LLM as a teacher generates questions and reference answer pairs from documents. These questiones are then posed to the RAG system and its answers are cmpared with the benchmark using the same LLM but now as an "examiner". The model evauates the answer`s accuracy with a score from 0 to 5 and explains the reasoning behind the answer. Question, truth value, answer and score are saved in a separate JSON file and a metric of the mean score is calculated.

#### Quantitative Evaluation

Two approaches were compared using a mean score from 0 to 5. The score shows the accuracy and usefulness of answers on the corpus of parliamentary documents.

| Approach | Mean-Score| Max Possible |
|:----------|:--------:|:--------:|
| Rule-based | 3.6 | 5 |
| Machine learning | 4.5 | 5 |
| Delta | +0.9 | - |
| Relative improvement | +25% | - |
| Error reduction | 64% | - |

Quantitative analysis showed that the ML approach significantly outperforms the rule-based approach, the average score increases from 3.6 to 4.5, corresponding to an increase in accuracy of approximately 25% and a 64% reduction in errors relative to the ideal result. This indicates a significantly higher ability of the ML system to correctly extract relevant context and generate accurate answers even with variable wording, whereas the rule-based approach remains limited and degrades performance on more complex questions.

#### Qualitative Evaluation

We have analysed the answers generated from the RAG model for both rule-based and machine learning based embeddings. The evaluation utilized a pre-defined test set consisting of 12 domain-specific questions. 
1) Correctness: Majority of questions have been answered correctly. However, in several instances, the generated answers contained multiple sentences, only one of which was factually correct and supported by the source material.
2) Sources:  A recurring issue was the generation of an accurate single sentence that was attributed to multiple, often redundant, source documents. Nevertheless, for the questions that could not have any source document, the model correctly indicated that no such source exists.
3) Hallucinations: we performed a dedicated hallucination test by posing a question that could not be answered using the provided document corpus. We are pleased to report that the model successfully passed this test, either by responding that the information was not present or by returning a null answer

#### Possible Improvements

- Try another model to create embeddings.
- Add a tool for detecting hallucinations.
- Make a promt "stronger" - hard rejection if there is no information, request to cite, limit the scope of the context.
- Add a simple verifier - a second prompt that checks the answer whithin the context.
