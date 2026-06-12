#!/usr/bin/env python3
"""
Smoke test: load Qwen2.5-7B-Instruct and run one MMLU question through
the full pipeline (inference → cache → CTC coordination).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 verify_local.py
"""

import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

def main():
    # 1) Load a few MMLU questions (philosophy, easy to verify)
    print("Loading 5 MMLU questions …")
    from ctc_llm.tasks.mmlu import load_mmlu
    qs = load_mmlu(subjects=["philosophy"], max_per_subject=5)
    if not qs:
        print("ERROR: no questions loaded — check datasets install")
        sys.exit(1)

    # 2) Create 5 agents (all share the same model weights)
    print("\nCreating 5 LocalLLMAgents (Qwen2.5-7B-Instruct) …")
    t0 = time.time()
    from ctc_llm.agents.local_llm_agent import LocalLLMAgent
    agents = [LocalLLMAgent(i, result_cache_dir="data/cache_verify") for i in range(5)]
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # 3) Run inference on first question
    q = qs[0]
    print(f"\nQuestion: {q.question[:90]}…")
    print(f"Choices: {q.choices}")
    print(f"Correct: {q.correct_letter}) {q.choices[q.correct]}\n")

    probs_all = {}
    for agent in agents:
        t1 = time.time()
        p = agent.get_probs(q.question, q.choices)
        dt = time.time() - t1
        probs_all[agent.agent_id] = p
        pred = "ABCD"[np.argmax(p)]
        mark = "✓" if np.argmax(p) == q.correct else "✗"
        print(f"  Agent {agent.agent_id}: A={p[0]:.3f} B={p[1]:.3f} "
              f"C={p[2]:.3f} D={p[3]:.3f}  → {pred} {mark}  ({dt*1000:.0f}ms)")

    print()

    # 4) Calibrate + CTC
    from ctc_llm.conformal.calibrate import calibrate, conformal_set
    from ctc_llm.coordination.ctc import ctc_answer
    from ctc_llm.coordination.majority import majority_answer
    from ctc_llm.coordination.entropy_trust import entropy_trust_answer

    cal_probs = list(probs_all.values())
    cal_correct = [q.correct] * len(cal_probs)
    _, q_hat = calibrate(cal_probs * 10, cal_correct * 10, alpha=0.10)
    print(f"q_hat = {q_hat:.4f}")

    for i, p in probs_all.items():
        cs = conformal_set(p, q_hat)
        print(f"  Agent {i}: conformal set = {['ABCD'[a] for a in cs]}")

    print()
    print("Coordination results (correct =", q.correct_letter + "):")
    print("  majority:     ", "ABCD"[majority_answer(probs_all)])
    print("  entropy_trust:", "ABCD"[entropy_trust_answer(probs_all)])
    print("  ctc:          ", "ABCD"[ctc_answer(probs_all, q_hat)])

    # 5) Simulate corrupt agent
    print("\nWith 2 overconfident corrupt agents (k=2/5):")
    from ctc_llm.agents.corrupt_agent import make_corrupt_agent
    corrupt_probs = dict(probs_all)
    for ci in [3, 4]:
        ca = make_corrupt_agent("overconfident", agents[ci], correct_idx=q.correct)
        corrupt_probs[ci] = ca.get_probs(q.question, q.choices)
        cp = corrupt_probs[ci]
        print(f"  Corrupt agent {ci}: A={cp[0]:.3f} B={cp[1]:.3f} "
              f"C={cp[2]:.3f} D={cp[3]:.3f}  → {['ABCD'][0][np.argmax(cp)]}")
    print("  majority:     ", "ABCD"[majority_answer(corrupt_probs)])
    print("  entropy_trust:", "ABCD"[entropy_trust_answer(corrupt_probs)])
    print("  ctc:          ", "ABCD"[ctc_answer(corrupt_probs, q_hat)])

    print("\nSMOKE TEST PASSED")

    # Cleanup verify cache
    import shutil
    shutil.rmtree("data/cache_verify", ignore_errors=True)


if __name__ == "__main__":
    main()
