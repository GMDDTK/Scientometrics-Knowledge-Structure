# Scientometrics Knowledge Structure

This repository contains the data, code, and clustering outputs used in the study *Mapping Domain Knowledge Structure through Semantic Clustering and Hierarchical Organization: A Case Study of Scientometrics*.

## Data

Records were retrieved from the OpenAlex core works corpus. The query was limited to English-language journal articles published through 2024 and combined the OpenAlex topics **“scientometrics and bibliometrics research”** and **“web visibility and informetrics”** with OR.

The archived export contained 40,446 records. Retaining records with at least one location marked as CWTS Core produced 29,365 records. Relevance screening resulted in a final analytical corpus of 24,767 publications.

- `Data_Initial.csv`: reduced-column copy of the initial OpenAlex export, including the fields needed to verify the retrieval and CWTS Core screening.
- `Data_Final.xlsx`: final analytical dataset of 24,767 publications.
- `all.txt`: exact title-and-abstract text supplied to the sentence-embedding model, with one publication per line in the same order as `Data_Final.xlsx`.

Large text fields were omitted from `Data_Initial.csv` to reduce its repository size. The complete initial export is available from the corresponding author upon reasonable request.

## Code and outputs

- `Embedding.py`: creates 1,024-dimensional document embeddings with `sentence-transformers/all-roberta-large-v1`.
- `Kmeans.py`, `GMM.py`, `HDBSCAN.py`, and `Spectral Clustering.py`: run the four clustering configurations.
- `Hierarchical Aggregation.py`: constructs the topic-level hierarchical aggregation.
- `0_Kmeans.xlsx`, `0_GMM.xlsx`, `0_HDBSCAN.xlsx`, and `0_Spectral Clustering.xlsx`: document assignments and cluster summaries produced by the four configurations.

## Reproduction

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn umap-learn hdbscan sentence-transformers torch scipy tqdm openpyxl
```

Run the scripts from the repository root. First run `Embedding.py` to generate `embeddings.npy`, then run the clustering scripts. `Hierarchical Aggregation.py` uses the retained spectral-clustering output.

The generated embedding matrix is not stored in the repository because of its size. It can be regenerated from `all.txt` using `Embedding.py`.
