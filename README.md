# 194.093 Natural Language Processing and Information Extraction (WS2025) - Project
## Topic 13 - Retrieval-augmented Generation for German official documents

### Contributors (Group 25): 
- [Valentin Tian](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=205694)
- [Diara Rinnerbauer](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=211007)
- [Stefan Sick](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=129430)
- [Shakoor Muhammad Nouman](https://tuwel.tuwien.ac.at/user/view.php?course=76899&id=211748)

### Instructor:
- [Gabor Recski](https://tiss.tuwien.ac.at/person/336863.html)

### Timeline:
| Date | Activity | Description |
|:----------|:--------:|------------:|
| 02/11/2025 | [Milestone 1](### Milestone 1:) | CoNLL-U Preprocessing |
| 30/11/2025 | Milestone 2 | Baseline Evaluation |
| 19/12/2025 | Review Meeting | Status Update |
| 16/01/2026 | Final Presentation | Presentation and Feedback |
| 25/01/2026 | Final Submission | Final Solution |

## Project Structure
### Milestone 1:
### Deliverables:
- [Preprocessing Code](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/preprocess.py)
- [Raw Data](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/tree/main/data/raw)
- [CoNLL-U Files](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/tree/main/data/processed)

### Process Documentation:
1. Data Extraction
   - All ```.xmi``` files were read from the ```Schweiz.tar``` archive.
   - The [python script](https://github.com/valentin-tian/tuw-ds-ws2025-nlp-g25-t13/blob/main/src/preprocess.py) extracted the raw text from the ```<Sofa>``` element of each ```.xmi``` document using ```xml.etree.ElementTree```.
2. NLP Preprocessing
   - Text was processed usinf the Stanza German language pipeline (processors: ```tokenize```, ```mwt```, ```pos```, ```lemma```).
   - The tokenized, lemmatized text with part of speach tags was created. The ```mwt``` processor is responsible for expanding multiword tokens and recovering their full structure from short forms.
3. Output
   - The documents were saved in the CoNLL-U format using ```stanza.utils.conll.CoNLL.write_doc2conll()```.
   - Each file was saved with the same name ID as an original one (e.g. ```20000324.xmi.xmi``` -> ```20000324.gz.conllu```).
   - All result files can be find at the [Google Drive CoNLL-U folder link](https://drive.google.com/drive/folders/1lk53aSkx_aZ6wANKXdhMattQz2hUac7B?usp=sharing).
4. Verification
   - A random sample of the generated files was manually inspected.
5. Result
   - A total of 100 ```.xmi``` documents were succesfully converted into CoNLL-U format.
