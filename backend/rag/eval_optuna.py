# eval_optuna.py
# Intelligent RAG parameter search using Optuna (Bayesian optimisation)
# scored with RAGAS metrics (faithfulness + answer_relevancy).
#
# This script is run from the command line — it is NOT connected to
# the Flask web server. Results are written to a JSON file and printed
# as a ranked table when the study finishes.
#
# Usage
# -----
#   python eval_optuna.py                          # 40 trials, all defaults
#   python eval_optuna.py --n-trials 60            # more trials
#   python eval_optuna.py --questions 20           # use first 20 gold questions
#   python eval_optuna.py --out results_optuna.json
#
# Requirements (add to requirements.txt)
# ───────────────────────────────────────
#   optuna
#   ragas
#   datasets          (ragas dependency)
#
# How it works
# ─────────────
# 1. Optuna proposes a parameter set {chunk_size, top_k, temperature, top_p}
#    using a Tree-structured Parzen Estimator (TPE) — smarter than a grid search.
# 2. For each trial, all gold questions are run through
#    run_query_for_eval_with_context() which returns both the answer and
#    the retrieved chunk texts that RAGAS needs.
# 3. RAGAS scores each (question, answer, contexts, ground_truth) tuple
#    on two metrics:
#      • faithfulness      — is the answer grounded in the retrieved chunks?
#      • answer_relevancy  — does the answer actually address the question?
#    The trial objective is their mean.
# 4. After all trials, the best parameter set is printed and all results
#    are saved to --out (default: optuna_results.json).

import argparse
import json
import logging
import os
import time
from typing import Any, List, Optional

import optuna
import google.generativeai as genai
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

from query import run_query_for_eval_with_context

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence Optuna's per-trial chatter — we print our own progress.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
GOLD_PATH   = os.path.join(os.path.dirname(__file__), "gold_questions.json")
CHROMA_PATH = os.environ.get("CHROMA_PATH_TEST", "rag/chroma_test")

# Throttle: minimum seconds between Gemini calls.
# Keeps us well under quota and dramatically reduces 503s.
INTER_CALL_DELAY = 0.6   # seconds

# ─────────────────────────────────────────────
# Search space
# Optuna samples from these ranges each trial.
# ─────────────────────────────────────────────
CHUNK_SIZE_CHOICES  = [200, 300, 500, 800, 1000, 1200]
TOP_K_RANGE         = (1, 5)          # int, inclusive
TEMPERATURE_RANGE   = (0.0, 1.0)      # float
TOP_P_RANGE         = (0.7, 1.0)      # float


def load_gold_questions(n: int) -> list[dict]:
    """Load up to n questions from gold_questions.json."""
    with open(GOLD_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    questions = data.get("questions", [])
    if n and n < len(questions):
        questions = questions[:n]
    logger.info("Loaded %d gold questions.", len(questions))
    return questions


def build_ragas_evaluator() -> tuple:
    """
    Build a RAGAS 0.1.x evaluator.

    LLM:        Custom LangChain LLM subclass wrapping google.generativeai
                directly — no langchain_google_genai dependency, avoiding
                all version conflicts.
    Embeddings: Local HuggingFace sentence-transformers model — same model
                used by the RAG pipeline, no API calls needed.

    Returns (metrics_list, llm_wrapper, embeddings).
    """

    # Custom LLM that wraps genai directly — avoids langchain_google_genai entirely
    class _GeminiLLM(LLM):
        model_name: str = "gemini-2.5-flash"

        @property
        def _llm_type(self) -> str:
            return "gemini"

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            model    = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else str(response)

    llm_wrapper = LangchainLLMWrapper(_GeminiLLM())

    # Local embeddings — no API calls, no v1beta routing issues
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    faithfulness.llm            = llm_wrapper
    answer_relevancy.llm        = llm_wrapper
    answer_relevancy.embeddings = embeddings

    return [faithfulness, answer_relevancy], llm_wrapper, embeddings


def run_trial(
    trial: optuna.Trial,
    gold_questions: list[dict],
    ragas_metrics: list,
    ragas_embeddings,
) -> float:
    """
    Optuna objective function.
    Suggests parameters, runs all gold questions, scores with RAGAS,
    returns mean(faithfulness, answer_relevancy) as the objective value.
    """
    chunk_size  = trial.suggest_categorical("chunk_size",  CHUNK_SIZE_CHOICES)
    top_k       = trial.suggest_int(        "top_k",       *TOP_K_RANGE)
    temperature = trial.suggest_float(      "temperature", *TEMPERATURE_RANGE)
    top_p       = trial.suggest_float(      "top_p",       *TOP_P_RANGE)

    logger.info(
        "Trial %d — chunk=%d  top_k=%d  temp=%.2f  top_p=%.2f",
        trial.number, chunk_size, top_k, temperature, top_p,
    )

    questions_col  = []
    answers_col    = []
    contexts_col   = []
    ground_truths  = []

    for i, item in enumerate(gold_questions):
        question     = item["question"]
        gold_answer  = item["gold_answer"]

        result = run_query_for_eval_with_context(
            question=question,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
            chroma_path=CHROMA_PATH,
        )

        # Skip questions where retrieval or generation failed entirely
        if not result["answer"] or not result["contexts"]:
            logger.warning("  Q%d: empty result — skipping.", i + 1)
            continue

        questions_col.append(question)
        answers_col.append(result["answer"])
        contexts_col.append(result["contexts"])
        ground_truths.append(gold_answer)

        time.sleep(INTER_CALL_DELAY)   # throttle Gemini calls

    if not questions_col:
        logger.warning("Trial %d: no valid results — returning 0.", trial.number)
        return 0.0

    # Build a HuggingFace Dataset — RAGAS 0.1.x column names
    dataset = Dataset.from_dict({
        "question":    questions_col,
        "answer":      answers_col,
        "contexts":    contexts_col,
        "ground_truth": ground_truths,
    })

    try:
        scores = evaluate(
            dataset,
            metrics=ragas_metrics,
            raise_exceptions=False,
        )
        # RAGAS 0.1.x returns a plain dict — access keys directly
        faith  = float(scores.get("faithfulness",     0.0) or 0.0)
        relev  = float(scores.get("answer_relevancy", 0.0) or 0.0)
        mean   = (faith + relev) / 2.0
    except Exception as exc:
        logger.error("RAGAS evaluation failed for trial %d: %s", trial.number, exc)
        return 0.0

    logger.info(
        "  → faithfulness=%.3f  answer_relevancy=%.3f  mean=%.3f",
        faith, relev, mean,
    )

    # Store per-trial breakdown as user attributes for later inspection
    trial.set_user_attr("faithfulness",     faith)
    trial.set_user_attr("answer_relevancy", relev)
    trial.set_user_attr("n_questions",      len(questions_col))

    return mean


def print_results_table(study: optuna.Study) -> None:
    """Print a ranked table of all completed trials."""
    trials = [t for t in study.trials if t.value is not None]
    trials.sort(key=lambda t: t.value or 0.0, reverse=True)

    header = (
        f"{'Rank':>4}  {'Chunk':>5}  {'TopK':>4}  {'Temp':>5}  "
        f"{'TopP':>5}  {'Faith':>6}  {'Relev':>6}  {'Mean':>6}"
    )
    print("\n" + "=" * len(header))
    print("OPTUNA + RAGAS RESULTS — ranked by mean score")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, t in enumerate(trials, start=1):
        p = t.params
        print(
            f"{rank:>4}  {p['chunk_size']:>5}  {p['top_k']:>4}  "
            f"{p['temperature']:>5.2f}  {p['top_p']:>5.2f}  "
            f"{t.user_attrs.get('faithfulness', 0.0):>6.3f}  "
            f"{t.user_attrs.get('answer_relevancy', 0.0):>6.3f}  "
            f"{t.value:>6.3f}"
        )

    print("=" * len(header))
    best = study.best_trial
    print(
        f"\n✅ Best configuration:\n"
        f"   chunk_size={best.params['chunk_size']}  "
        f"top_k={best.params['top_k']}  "
        f"temperature={best.params['temperature']:.2f}  "
        f"top_p={best.params['top_p']:.2f}\n"
        f"   faithfulness={best.user_attrs.get('faithfulness', 0.0):.3f}  "
        f"answer_relevancy={best.user_attrs.get('answer_relevancy', 0.0):.3f}  "
        f"mean={best.value:.3f}\n"
    )


def save_results(study: optuna.Study, out_path: str) -> None:
    """Serialise all trial results to JSON."""
    output = {
        "best_params": study.best_params,
        "best_value":  study.best_value,
        "trials": [
            {
                "number":      t.number,
                "params":      t.params,
                "value":       t.value,
                "faithfulness":     t.user_attrs.get("faithfulness"),
                "answer_relevancy": t.user_attrs.get("answer_relevancy"),
                "n_questions":      t.user_attrs.get("n_questions"),
            }
            for t in study.trials
            if t.value is not None
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("Results saved to %s", out_path)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna + RAGAS hyperparameter search for the RAG pipeline."
    )
    parser.add_argument(
        "--n-trials", type=int, default=40,
        help="Number of Optuna trials (default: 40).",
    )
    parser.add_argument(
        "--questions", type=int, default=0,
        help="Gold questions to use per trial (0 = all, default: all).",
    )
    parser.add_argument(
        "--out", default="optuna_results.json",
        help="Path to write JSON results (default: optuna_results.json).",
    )
    parser.add_argument(
        "--study-name", default="rag_param_search",
        help="Optuna study name (default: rag_param_search).",
    )
    args = parser.parse_args()

    gold_questions = load_gold_questions(args.questions)
    ragas_metrics, _, ragas_embeddings = build_ragas_evaluator()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info(
        "Starting Optuna study '%s' — %d trials × %d questions.",
        args.study_name, args.n_trials, len(gold_questions),
    )

    study.optimize(
        lambda trial: run_trial(trial, gold_questions, ragas_metrics, ragas_embeddings),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    print_results_table(study)
    save_results(study, args.out)


if __name__ == "__main__":
    main()
