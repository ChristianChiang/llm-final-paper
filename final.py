import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm.autonotebook import tqdm
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU.")
    device = torch.device("cpu")

print("\nLoading the PrimeIntellect/stackexchange-question-answering dataset...")
full_dataset = load_dataset("PrimeIntellect/stackexchange-question-answering", split="train")

subset_size = 50000
dataset_subset = full_dataset.select(range(subset_size))
df = dataset_subset.to_pandas()

corpus = {}
queries = {}
qrels = {}
skipped_rows = 0

print("Processing dataset and building corpus, queries, and qrels...")
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    if 'prompt' in row and 'gold_standard_solution' in row and pd.notna(row['prompt']) and pd.notna(row['gold_standard_solution']):
        try:
            full_prompt = str(row['prompt'])
            question_text = full_prompt.split("Question:\n")[1].split("\n\nNow provide the response")[0].strip()

        except IndexError:
            skipped_rows += 1
            continue

        answer_text = str(row['gold_standard_solution']).strip()
        
        if question_text and answer_text:
            doc_id = str(idx)
            query_id = str(idx)
            
            corpus[doc_id] = answer_text
            queries[query_id] = question_text
            
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = 1
        else:
            skipped_rows += 1
    else:
        skipped_rows += 1

print(f"\nSuccessfully processed the dataset.")
print(f"Total rows processed: {df.shape[0]}")
print(f"Rows skipped due to missing/empty data or parsing errors: {skipped_rows}")
print(f"Corpus size (number of answers): {len(corpus)}")
print(f"Queries size (number of questions): {len(queries)}")
print(f"Relevance judgments (qrels) size: {len(qrels)}")

corpus_doc_ids = list(corpus.keys())

print("\n--- Setting up Baseline: BM25 ---")
print("Indexing documents for BM25...")

if corpus:
    tokenized_corpus = [doc.split(" ") for doc in tqdm(corpus.values())]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 indexing complete.")
else:
    print("Corpus is empty, skipping BM25 indexing.")
    bm25 = None


def search_bm25(query_text, k=10):
    if not bm25:
        return []
    tokenized_query = query_text.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    
    top_n_indices = np.argsort(doc_scores)[::-1][:k]
    top_n_doc_ids = [corpus_doc_ids[i] for i in top_n_indices]
    
    return top_n_doc_ids

print("\n--- Setting up Improved Method 1: Bi-Encoder ---")

bi_encoder_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)
print("Bi-Encoder model loaded.")

if corpus:
    print("Encoding the corpus with the Bi-Encoder (this may take a while)...")
    corpus_embeddings = bi_encoder_model.encode(
        list(corpus.values()), 
        convert_to_tensor=True, 
        show_progress_bar=True,
        batch_size=128
    )
    corpus_embeddings = corpus_embeddings.cpu().numpy()
    
    print("Building FAISS-CPU index...")
    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(corpus_embeddings)

    print(f"FAISS-CPU index built successfully with {index.ntotal} vectors.")
else:
    print("Corpus is empty, skipping Bi-Encoder setup.")
    index = None

def search_bi_encoder(query_text, k=10):
    if not index:
        return []
    query_embedding = bi_encoder_model.encode(query_text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    
    distances, indices = index.search(query_embedding, k)
    top_n_doc_ids = [corpus_doc_ids[i] for i in indices[0]]
    return top_n_doc_ids

print("\n--- Setting up Improved Method 2: Re-Ranking with Cross-Encoder ---")

cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device=device)
print("Cross-Encoder model loaded.")

def rerank_with_cross_encoder(query_text, doc_ids):
    if not doc_ids:
        return []
    
    query_doc_pairs = [[query_text, corpus[doc_id]] for doc_id in doc_ids]
    
    scores = cross_encoder_model.predict(query_doc_pairs, show_progress_bar=False)
    
    scored_docs = sorted(zip(scores, doc_ids), key=lambda x: x[0], reverse=True)
    
    reranked_doc_ids = [doc_id for score, doc_id in scored_docs]
    return reranked_doc_ids

def search_and_rerank(query_text, k=10):
    
    candidate_doc_ids = search_bi_encoder(query_text, k=50)
    
    reranked_ids = rerank_with_cross_encoder(query_text, candidate_doc_ids)
    
    return reranked_ids[:k]

print("\n--- Starting Evaluation ---")

def evaluate(search_function, name, k_values=[1, 5, 10]):
    
    print(f"Evaluating: {name}...")
    
    reciprocal_ranks = []
    
    recalls = {k: 0 for k in k_values}
    
    evaluation_queries = {qid: queries[qid] for qid in list(queries.keys())[:1000]}
    
    if not evaluation_queries:
        print(f"No queries to evaluate for {name}. Skipping.")
        return {f'Recall@{k}': 0 for k in k_values} | {'MRR@10': 0}

    for qid, query_text in tqdm(evaluation_queries.items()):
        relevant_docs = set(qrels.get(qid, {}).keys())
        if not relevant_docs:
            continue
        
        retrieved_docs = search_function(query_text, k=max(k_values))
        
        rank = 0
        for i, doc_id in enumerate(retrieved_docs[:10]):
            if doc_id in relevant_docs:
                rank = i + 1
                break
        
        if rank > 0:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
            
        retrieved_set = set(retrieved_docs)
        for k in k_values:
            if len(set(retrieved_docs[:k]).intersection(relevant_docs)) > 0:
                 recalls[k] += 1
                 
    mrr_at_10 = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    final_recalls = {k: count / len(evaluation_queries) for k, count in recalls.items()}
    
    metrics = {'MRR@10': mrr_at_10}
    for k in k_values:
        metrics[f'Recall@{k}'] = final_recalls[k]
        
    return metrics

evaluation_results = {}

evaluation_results['BM25 (Baseline)'] = evaluate(
    lambda q, k: search_bm25(q, k), 
    name="BM25"
)

evaluation_results['Bi-Encoder (Semantic)'] = evaluate(
    lambda q, k: search_bi_encoder(q, k), 
    name="Bi-Encoder"
)

evaluation_results['Retrieve & Re-rank'] = evaluate(
    lambda q, k: search_and_rerank(q, k), 
    name="Retrieve & Re-rank"
)

results_df = pd.DataFrame(evaluation_results).T
print("\n--- FINAL EVALUATION RESULTS ---")
print(results_df.to_string())

results_df.to_csv("stackexchange_qa_retrieval_results.csv")
print("\nEvaluation results saved to stackexchange_qa_retrieval_results.csv")

print("\n--- Starting Qualitative Analysis ---")
print("Finding examples where semantic search outperforms the baseline...")

qualitative_examples = []
for qid, query_text in tqdm(list(queries.items())[:1000]):
    if qid not in qrels:
        continue
    
    relevant_docs = set(qrels[qid].keys())
    
    bm25_top1_list = search_bm25(query_text, k=1)
    rerank_top1_list = search_and_rerank(query_text, k=1)
    
    if not bm25_top1_list or not rerank_top1_list:
        continue

    bm25_top1 = bm25_top1_list[0]
    rerank_top1 = rerank_top1_list[0]
    
    if bm25_top1 not in relevant_docs and rerank_top1 in relevant_docs:
        example = {
            "query_id": qid,
            "query_text": query_text,
            "relevant_doc_id": list(relevant_docs)[0],
            "relevant_doc_text": corpus[list(relevant_docs)[0]][:500] + "...",
            "bm25_retrieved_doc_id": bm25_top1,
            "bm25_retrieved_doc_text": corpus[bm25_top1][:500] + "...",
            "semantic_retrieved_doc_id": rerank_top1,
            "semantic_retrieved_doc_text": corpus[rerank_top1][:500] + "..."
        }
        qualitative_examples.append(example)
    
    if len(qualitative_examples) >= 10:
        break

if qualitative_examples:
    examples_df = pd.DataFrame(qualitative_examples)
    examples_df.to_csv("qualitative_analysis_examples.csv", index=False)
    print(f"\nFound {len(qualitative_examples)} valuable examples for qualitative analysis.")
    print("Saved to qualitative_analysis_examples.csv")
else:
    print("\nCould not automatically find strong qualitative examples in the first 1000 queries. Consider increasing the search range.")
