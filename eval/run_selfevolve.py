"""Self-evolution experiment runner.

Sequential loop over tau-bench tasks:
    for each task:
        1. agent solves it (with strategies retrieved from the bank)
        2. distill new strategies from the trajectory (success AND failure)
        3. add them to the bank (dedup/merge), record outcome feedback
    -> cumulative pass rate over task order is the self-evolution curve

Ablations (flags):
    --bank-mode topk|random|off   retrieval strategy
    --sources both|success|failure  which trajectories get distilled

Run from repo root with the venv's python. Env vars: DEEPSEEK_API_KEY.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm

litellm.request_timeout = 120  # cap hangs inside the user simulator too

from tau_bench.envs import get_env

from evolvebank.agent import EvolvingToolCallingAgent
from evolvebank.bank import StrategyBank
from evolvebank.distill import distill_strategies, merge_two


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="retail", choices=["retail", "airline"])
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--provider", default="deepseek")
    p.add_argument("--user-model", default="deepseek-chat")
    p.add_argument("--user-provider", default="deepseek")
    p.add_argument("--task-split", default="test", choices=["train", "test", "dev"])
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=-1)
    p.add_argument("--bank-mode", default="topk", choices=["topk", "random", "off"])
    p.add_argument("--bank-k", type=int, default=3)
    p.add_argument("--sources", default="both", choices=["both", "success", "failure"])
    p.add_argument("--distill", type=int, default=1, help="1=distill after every task")
    p.add_argument("--bank-path", default=None, help="reuse/freeze a prebuilt bank")
    p.add_argument("--tag", default="run", help="experiment tag for output files")
    p.add_argument("--resume", action="store_true", help="skip tasks already in the log")
    p.add_argument("--seed", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    import random

    random.seed(args.seed)

    bank = StrategyBank(path=args.bank_path or f"experiments/bank_{args.tag}.json")
    if args.bank_path:
        # frozen prebuilt bank: never write back to it
        bank.path = f"experiments/bank_{args.tag}_copy.json"

    env = get_env(
        args.env,
        user_strategy="llm",
        user_model=args.user_model,
        user_provider=args.user_provider,
        task_split=args.task_split,
    )
    agent = EvolvingToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=args.model,
        provider=args.provider,
        temperature=0.0,
        bank=bank if args.bank_mode != "off" else None,
        bank_k=args.bank_k,
        bank_mode=args.bank_mode,
    )

    end = len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    idxs = list(range(args.start_index, end))

    log_path = f"experiments/results/selfevolve_{args.tag}.jsonl"
    os.makedirs("experiments/results", exist_ok=True)

    n_pass, n_prev = 0, 0
    if args.resume and os.path.exists(log_path):
        done = {}
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["task_id"]] = r
        idxs = [i for i in idxs if i not in done]
        n_pass = sum(int(r["success"]) for r in done.values())
        n_prev = len(done)
        print(f"resuming: {n_prev} tasks already done, {len(idxs)} to go")

    for n, idx in enumerate(idxs, n_prev + 1):
        t0 = time.time()
        reward, messages, error = 0.0, [], None
        for attempt in range(3):  # retry on transient network/API errors
            try:
                task_env = get_env(  # NB: env creation itself calls the user LLM
                    args.env,
                    user_strategy="llm",
                    user_model=args.user_model,
                    user_provider=args.user_provider,
                    task_split=args.task_split,
                    task_index=idx,
                )
                res = agent.solve(env=task_env, task_index=idx)
                reward, messages, error = res.reward, res.messages, None
                break
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                transient = any(k in str(e) for k in ("Timeout", "timed out", "Connection", "rate"))
                if not transient or attempt == 2:
                    break
                print(f"  retry {attempt + 1}/3 after error: {error[:120]}")
                time.sleep(30)
        success = reward >= 1 - 1e-6
        n_pass += int(success)

        task = env.tasks[idx]
        instruction = task.instruction if hasattr(task, "instruction") else task["instruction"]

        # ---- learn from this trajectory ----
        distilled = []
        if args.distill and messages and not error:
            want = (
                (args.sources == "both")
                or (args.sources == "success" and success)
                or (args.sources == "failure" and not success)
            )
            if want:
                try:
                    distilled = distill_strategies(messages, instruction, success, model=args.model)
                except Exception as e:
                    print(f"  distill error: {e}")
                for text in distilled:
                    bank.add(text, source="success" if success else "failure", merge_fn=merge_two)
                bank.record_outcomes(agent.last_used_ids, success)
                if not args.bank_path:
                    bank.save()

        entry = {
            "task_id": idx,
            "reward": reward,
            "success": success,
            "cum_pass_rate": n_pass / n,
            "bank_size": len(bank),
            "n_distilled": len(distilled),
            "distilled": distilled,
            "used_ids": agent.last_used_ids,
            "seconds": round(time.time() - t0, 1),
            "error": error,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        mark = "OK " if success else "FAIL"
        total_n = n_prev + len(idxs)
        print(
            f"[{n}/{total_n}] task={idx} {mark} cum={n_pass / n:.3f} "
            f"bank={len(bank)} +{len(distilled)} ({entry['seconds']}s)"
        )

    total_n = n_prev + len(idxs)
    print(f"\nFinal pass rate: {n_pass}/{total_n} = {n_pass / total_n:.3f}")
    print(f"Bank size: {len(bank)} | log: {log_path}")


if __name__ == "__main__":
    main()
