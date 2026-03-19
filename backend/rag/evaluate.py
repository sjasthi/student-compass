import json
from sentence_transformers import SentenceTransformer, util
from query import run_query as query

# Load your embedding model for scoring
scoring_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global list to store experiment results
experiment_results = []

def load_gold_questions():
    with open("gold_questions.json", encoding="utf-8") as f:
        return json.load(f)["questions"]

def set_retriever_params(chunk_size, top_k):
    import query
    query.CURRENT_TOP_K = top_k

def run_all_50_questions(gold_questions):
    results = []
    for item in gold_questions:
        question = item["question"]
        gold_answer = item["gold_answer"]

        # Call your existing RAG query function
        model_answer = query(question)

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer
        })
    return results

def score_results(results):
    scores = []
    for r in results:
        gold = r["gold_answer"]
        pred = r["model_answer"]

        # Compute semantic similarity
        emb_gold = scoring_model.encode(gold, convert_to_tensor=True)
        emb_pred = scoring_model.encode(pred, convert_to_tensor=True)
        sim = util.cos_sim(emb_gold, emb_pred).item()

        # Convert similarity to score
        if sim > 0.85:
            score = 3
        elif sim > 0.70:
            score = 2
        elif sim > 0.50:
            score = 1
        else:
            score = 0

        scores.append(score)

    return sum(scores) / len(scores)  # return accuracy

def save_scores(chunk_size, top_k, accuracy):
    experiment_results.append({
        "chunk_size": chunk_size,
        "top_k": top_k,
        "accuracy": accuracy
    })

def print_comparison_table():
    print("\n=== RAG Accuracy Comparison ===\n")
    print("| Chunk | TopK | Accuracy |")
    print("|-------|------|----------|")
    for r in experiment_results:
        print(f"| {r['chunk_size']} | {r['top_k']} | {r['accuracy']:.2f} |")

# -----------------------------
# Main Evaluation Loop
# -----------------------------

gold_questions = load_gold_questions()

for chunk_size in [200, 300, 500, 800]:
    for top_k in [1, 2, 3]:
        set_retriever_params(chunk_size, top_k)
        results = run_all_50_questions(gold_questions)
        accuracy = score_results(results)
        save_scores(chunk_size, top_k, accuracy)

print_comparison_table()