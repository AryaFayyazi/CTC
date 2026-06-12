"""
Backfill: add SC, Debate, MoA, LLM-Judge to existing records by replaying
coordination math on cached agent probs. No LLM inference needed.

Usage:
    python3 scripts_backfill_baselines.py results/raw_results.json
"""

from __future__ import annotations
import json, os, sys, random, time
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List
import numpy as np

from ctc_llm.tasks.mmlu import (
    load_mmlu, load_truthfulqa, load_arc, MMLU_SUBSET, Question,
)
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.agents.corrupt_agent  import make_corrupt_agent
from ctc_llm.coordination.self_consistency import (
    self_consistency_answer, self_consistency_confidence,
)
from ctc_llm.coordination.debate import (
    debate_answer, debate_confidence,
)
from ctc_llm.coordination.mixture_of_agents import (
    mixture_of_agents_answer, mixture_of_agents_confidence,
)
from ctc_llm.coordination.llm_judge import (
    llm_judge_answer, llm_judge_confidence,
)


def load_task(name: str) -> List[Question]:
    if name == "mmlu":       return load_mmlu(subjects=MMLU_SUBSET)
    if name == "truthfulqa": return load_truthfulqa(max_questions=600)
    if name == "arc":        return load_arc(max_questions=600)
    raise ValueError(name)


def make_test_set(qs: List[Question], seed: int) -> List[Question]:
    rng = random.Random(seed)
    shuffled = qs[:]
    rng.shuffle(shuffled)
    cal_n = max(50, len(shuffled) // 3)
    return shuffled[cal_n:]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/raw_results.json"
    print(f"Loading {path} …")
    with open(path) as f:
        records = json.load(f)
    print(f"  {len(records)} records")

    need_keys = ["self_consistency_accuracy", "debate_accuracy",
                 "mixture_of_agents_accuracy", "llm_judge_accuracy"]
    need = [r for r in records if any(k not in r for k in need_keys)]
    print(f"  {len(need)} records need new baselines backfill")
    if not need:
        return

    task_cache: Dict[str, List[Question]] = {}
    agents_cache: Dict = {}

    t0 = time.time()
    for idx, r in enumerate(need):
        task     = r["task"]
        model_id = r["model_id"]
        n_agents = r["n_agents"]
        k_corr   = r["n_corrupt"]
        attack   = r["attack"]
        seed     = r["seed"]

        if task not in task_cache:
            task_cache[task] = load_task(task)
        qs = task_cache[task]

        agents_key = (model_id, n_agents)
        if agents_key not in agents_cache:
            agents_cache[agents_key] = [
                LocalLLMAgent(i, model_id=model_id,
                              hf_cache_dir="/data/models",
                              result_cache_dir="data/cache")
                for i in range(n_agents)
            ]
        agents = agents_cache[agents_key]

        test_q = make_test_set(qs, seed)
        if not test_q: continue

        sc_correct = dbg_correct = moa_correct = jdg_correct = 0
        sc_confs = dbg_confs = moa_confs = jdg_confs = None
        sc_confs, dbg_confs, moa_confs, jdg_confs = [], [], [], []
        for q_idx, q in enumerate(test_q):
            corrupt_seed = seed * 100_000 + q_idx
            rng = random.Random(corrupt_seed)
            corrupt_ids = set(rng.sample(range(n_agents), k=min(k_corr, n_agents)))
            agent_probs: Dict[int, np.ndarray] = {}
            for i, agent in enumerate(agents):
                if i in corrupt_ids:
                    corrupt = make_corrupt_agent(attack, agent,
                                                  correct_idx=q.correct,
                                                  agent_id=i,
                                                  model_id=model_id,
                                                  hf_cache_dir="/data/models",
                                                  result_cache_dir="data/cache")
                    agent_probs[i] = corrupt.get_probs(q.question, q.choices)
                else:
                    agent_probs[i] = agent.get_probs(q.question, q.choices)
            sc_seed = (corrupt_seed * 31 + 17) & 0x7FFFFFFF
            sc  = self_consistency_answer(agent_probs, seed=sc_seed)
            dbg = debate_answer(agent_probs)
            moa = mixture_of_agents_answer(agent_probs)
            jdg = llm_judge_answer(agent_probs)
            _, sc_c  = self_consistency_confidence(agent_probs, seed=sc_seed)
            _, dbg_c = debate_confidence(agent_probs)
            _, moa_c = mixture_of_agents_confidence(agent_probs)
            _, jdg_c = llm_judge_confidence(agent_probs)
            if sc  == q.correct: sc_correct  += 1
            if dbg == q.correct: dbg_correct += 1
            if moa == q.correct: moa_correct += 1
            if jdg == q.correct: jdg_correct += 1
            sc_confs.append(sc_c); dbg_confs.append(dbg_c)
            moa_confs.append(moa_c); jdg_confs.append(jdg_c)
        n = len(test_q)
        r["self_consistency_accuracy"]  = sc_correct  / n
        r["debate_accuracy"]            = dbg_correct / n
        r["mixture_of_agents_accuracy"] = moa_correct / n
        r["llm_judge_accuracy"]         = jdg_correct / n
        r["self_consistency_conf_per_q"]  = sc_confs
        r["debate_conf_per_q"]            = dbg_confs
        r["mixture_of_agents_conf_per_q"] = moa_confs
        r["llm_judge_conf_per_q"]         = jdg_confs
        # leave per_q empty (tables don't need it for these)
        r["self_consistency_per_q"]  = []
        r["debate_per_q"]            = []
        r["mixture_of_agents_per_q"] = []
        r["llm_judge_per_q"]         = []

        if (idx + 1) % 50 == 0 or (idx + 1) == len(need):
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta = rate * (len(need) - idx - 1)
            print(f"  {idx+1}/{len(need)}  ({elapsed/60:.1f}m, ETA {eta/60:.1f}m)", flush=True)
            # incremental save
            tmp = path + ".tmp"
            with open(tmp, "w") as f: json.dump(records, f)
            os.replace(tmp, path)

    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f: json.dump(records, f)
        os.replace(tmp, path)
    except FileNotFoundError:
        pass  # incremental save already wrote it
    print(f"Backfilled {len(need)} records in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
