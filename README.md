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

## Repository Structure

```text
├── data/
│   Contains all datasets and a list of questions for evaluation
|   used in the project including raw input data and any intermediate or preprocessed
|   files required for the Milestone 1 and further experiments.
│
├── src/
│   Core source code of the project. This folder includes the implementation of
│   data processing, model logic and evaluation components.
│
├── output/
│   Stores all generated outputs such as RAG outputs.
│
├── NLP_IE_2025WS_Exercise.pdf
│   Official exercise description and task specification.
│
├── WS2025_NLP_G25_T13_Slides.pdf
│   Slides used for the final project presentation.
│
├── README.md
│   High-level project description and instructions on how to reproduce the key results and details
|   of the Milestone 1 and Milestone 2 results.
│
├── requirements.txt
│   List of Python dependencies required to run the project.
│
├── environment_ollama.yml
│   Conda environment specification for running experiments with Ollama-based models.
│
├── LICENSE
│   License information for this repository.
```
## Implemented approaches

1. Rule-based (Vectoring: BM-25, Answering: ChatGPT 4.1-mini). Source: [NLP_RAG_TFDF.ipynb](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/NLP_RAG_TFDF.ipynb), output: [Rule-based answers](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/output/100_docs/tfdf_20q_100d.json)
2. ML-based (Vectoring: E5 Text Embeddings, Answering: ChatGPT 4.1-mini). Source: [NLP_RAG_LLMRetriev.ipynb](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/NLP_RAG_LLMRetriev.ipynb), output: [ML-based answers](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/output/100_docs/llm_20q_100d.json)
3. Local SLM (Vectoring: Nomic-embed text, Answering: LLama 3.2). Source: [NLP_RAG_LLMLocal.ipynb](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/NLP_RAG_LLMLocal.ipynb), output: [Local SLM answers](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/output/100_docs/output_llama_20q_100d.json)
4. Verbatim (Vectoring: E5 Text Embeddings, Answering: Verbatim). Source: [NLP_VerbatimRAG_LLMRetriev.ipynb](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/NLP_VerbatimRAG_LLMRetriev.ipynb), output: [Verbatim answers](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/output/100_docs/verbatim_20q_100d.json)

## Interim results
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

## Final results
### Retrieval evaluation
#### Description of the metrics used

To assess the quality of the retrieval systems, standard information retrieval metrics were used: Recall@k and MRR.

Recall@1 shows the proportion of queries for which the relevant document was found in the first search result position. 
This metric reflects the system's ability to immediately return the correct result and is especially important for scenarios where the user expects an accurate answer.

Recall@5 measures the proportion of queries for which the relevant document appears in the top 5 results. 
It characterizes the overall coverage of the retrieval system and its ability to find relevant information even if it is not in the first position.

MRR (Mean Reciprocal Rank) takes into account the position of the first relevant document in the search results and averages the inverse rank across all queries. 
The higher the MRR, the closer to the top of the list the system typically places the correct document, reflecting the quality of the ranking, not just the fact of finding it.

| System                 | Recall@1 | Recall@5 | MRR  |
|------------------------|----------|----------|------|
| TF-IDF (BM25)          | 0.737    | 0.895    | 0.80 |
| E5 Text Embeddings     | 0.74     | 0.79     | 0.75 |
| Verbatim               | 0.74     | 0.79     | 0.75 |
| Nomic-embed-text       | 0.53     | 0.73     | 0.62 |

The results show that TF-IDF (BM25) demonstrates the most consistent and strong performance among all the methods. It achieves a high Recall@5 of 0.895 and the best MRR of 0.80, 
indicating effective ranking and the ability to consistently return relevant documents in the top search results.

E5 Text Embeddings and Verbatim retrieval show comparable results: both methods achieve a Recall@1 of approximately 0.74 and an MRR of 0.75 due to that they used the same vectoring method and model.
This suggests that semantic embeddings are competitive with classical lexical methods in the retrieval task but they are inferior to BM25 in terms of depth of coverage.

The Nomic-embed-text model significantly lags behind the other approaches across all metrics. The low Recall@1 of 0.53 and MRR of 0.62 indicate problems both with finding relevant documents
and with their correct ranking, making this method the least suitable for the retrieval scenario under consideration.

Overall, the results confirm that classical lexical retrieval remains a strong baseline solution, especially in tasks where exact term matching and correct ranking are key. 
Semantic embeddings show promising but less stable results and require additional tuning or combination with lexical methods.

### Answering evaluation

#### Manual Answer Quality Assessment Method

To assess the quality of the RAG system responses, a manual assessment was conducted to verify the factual correctness and consistency of the responses with the source data.

Each response was scored on a three-point scale:

- 1 (True) - the response is completely correct and supported by the provided data;
- 0.5 (Almost True) - the response is generally correct but contains noise, redundant information, subjective elements, or incorrect/inaccurate references;
- 0 (False) - the response is incorrect or not supported by the data.

Additionally questions were grouped by type:

- Facts with numbers - questions requiring the precise extraction of numerical facts;
- Definitional facts - questions defining or explaining concepts;
- No answer in data - control questions for which there is no answer in the source data.

| Question Group          | TF-IDF (BM25) | E5 Text Embeddings | Verbatim | Nomic-embed-text |
|-------------------------|---------------|-------------------|----------|------------------|
| Facts with numbers      | 6/8           | 6.5/8             | 3.5/8    | 1.5/8            |
| Definitional facts     | 7/10          | 6/10              | 6.5/10   | 4/10             |
| No answer in data       | 2/2           | 2/2               | 2/2      | 2/2              |
| **Total**               | **15/20**     | **14.5/20**       | **12/20**| **7.5/20**       |
| **Accuracy**            | **0.75**      | **0.75**          | **0.6**  | **0.375**        |

The results of manual evaluation show that the TF-IDF (BM25) and E5 Text Embeddings-based systems achieve the highest overall accuracy of 0.75. Importantly, the same response model (ChatGPT 4.1-mini)
was used to generate responses in both cases, allowing for a direct comparison of the impact of the retrieval approach on the overall performance of the RAG system. 
These results indicate that differences in response quality are primarily due to the quality of the retrieved context rather than the generative model.

The Verbatim method demonstrates lower accuracy of 0.6, which is due to its fundamental feature: the system attempts to extract a direct answer from the text without using additional knowledge or reformulation. 
This approach reduces the risk of hallucinations but makes the system sensitive to question wording and limits its ability to generalize information, ultimately leading to lower response accuracy, what is actually more important
than just high accuracy in the domain of parliamentary documents.

The Nomic-embed-text-based system significantly underperforms the other approaches with Accuracy = 0.375, especially on questions requiring precise factual matching. This indicates insufficient relevance of the extracted context, 
which negatively impacts the quality of the final answers even when using the same generation scheme.

Finally, all the systems examined correctly handle "No answer in data" questions, confirming the absence of systematic hallucinations in scenarios where the answer is missing from the source data.
