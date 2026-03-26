import json
import shutil
import subprocess
import os
import time
import requests
import sys
import csv
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

def query(question, top_k):
    response = requests.post(
        "http://127.0.0.1:8000/query",
        json={"question": question, "top_k": top_k}
    )
    return response.json()["answer"]

scoring_model = SentenceTransformer("all-MiniLM-L6-v2")
experiment_results = []


def safe_rmtree(path, retries=20, delay=0.2):
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(delay)
    raise PermissionError(f"Could not delete {path} after multiple retries")


def load_gold_questions():
    with open("gold_questions.json", encoding="utf-8") as f:
        return json.load(f)["questions"]


def run_all_50_questions(gold_questions, top_k):
    results = []

    def process_question(item):
        model_answer = query(item["question"], top_k)
        return {
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "model_answer": model_answer
        }

    # use 10 threads to evaluate multiple questions concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_question, item): item for item in gold_questions}

        for i, future in enumerate(as_completed(futures), start=1):
            print(f"Completed {i}/{len(gold_questions)}")
            results.append(future.result())

    return results


def score_results(results):
    scores = []
    for r in results:
        emb_gold = scoring_model.encode(r["gold_answer"], convert_to_tensor=True)
        emb_pred = scoring_model.encode(r["model_answer"], convert_to_tensor=True)
        sim = util.cos_sim(emb_gold, emb_pred).item()

        if sim > 0.85:
            scores.append(3)
        elif sim > 0.70:
            scores.append(2)
        elif sim > 0.50:
            scores.append(1)
        else:
            scores.append(0)

    return sum(scores) / len(scores)

def save_scores(chunk_size, top_k, accuracy):
    experiment_results.append({
        "chunk_size": chunk_size,
        "top_k": top_k,
        "accuracy": accuracy
    })

def export_to_csv(filename="rag_results.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_size", "top_k", "accuracy"])
        for r in experiment_results:
            writer.writerow([r["chunk_size"], r["top_k"], f"{r['accuracy']:.2f}"])


def print_comparison_table():
    print("\n=== RAG Accuracy Comparison ===\n")
    print("Score Rubric:")
    print("3 = Perfect match")
    print("2 = Good match")
    print("1 = Partial match")
    print("0 = Incorrect\n")

    print(f"{'Chunk':<8} {'TopK':<6} {'Accuracy':<10}")
    print(f"{'-' * 8} {'-' * 6} {'-' * 10}")

    for r in experiment_results:
        print(f"{r['chunk_size']:<8} {r['top_k']:<6} {r['accuracy']:<10.2f}")

# -----------------------------
# SERVER MANAGEMENT
# -----------------------------

def wait_for_server_ready(timeout=30):
    url = "http://127.0.0.1:8000/docs"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=0.5)
            if r.status_code == 200:
                return True
        except:
            time.sleep(0.5)
    return False

def start_server():
    print("Starting server...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    print("Waiting for server to become ready...")
    if wait_for_server_ready():
        print("Server is ready.")
    else:
        print("ERROR: Server did NOT become ready in time.")
        sys.exit(1)

    return proc

def stop_server(proc):
    print("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

# -----------------------------
# MAIN LOOP
# -----------------------------

gold_questions = load_gold_questions()[:50]
CHROMA_PATH = "rag/chroma"

for chunk_size in [200, 300, 500, 800, 1000, 1200]:

    print(f"\n===== Testing chunk size: {chunk_size} =====")

    if os.path.exists(CHROMA_PATH):
        safe_rmtree(CHROMA_PATH)

    print(f"Rebuilding vector store with chunk size {chunk_size}...")
    subprocess.run([sys.executable, "ingest.py", str(chunk_size)], check=True)

    time.sleep(2)  # ensure Chroma files are fully written

    server_proc = start_server()

    def evaluate_top_k(k):
        print(f"\n--- Evaluating top_k={k} ---")
        results = run_all_50_questions(gold_questions, k)
        accuracy = score_results(results)
        save_scores(chunk_size, k, accuracy)
        return k, accuracy

    # run top_k = 1,2,3 in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_top_k, k) for k in [1, 2, 3]]

        for future in as_completed(futures):
            k, acc = future.result()
            print(f"Finished top_k={k} with accuracy={acc:.2f}")

    stop_server(server_proc)
    time.sleep(0.5)

print_comparison_table()
export_to_csv()
