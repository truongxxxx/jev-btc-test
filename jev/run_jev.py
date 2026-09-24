# -*- coding: utf-8 -*-
"""
Ask Jev about every sample in dataset.jsonl (candidate A_top12) and store the probabilities.

Each request: the state (num or text format) + 4 Noul questions
    next_up     the question under test
    next_down   the inverse; a coherent model gives p_up + p_down ≈ 1
    ctrl_rsi    control, the answer is in the state
    ctrl_ema9   control, the answer is in the state

Results are appended to results_<variant>.jsonl as they arrive; a rerun only sends
the samples that are still missing.

    python run_jev.py --variant num --dry-run
    python run_jev.py --variant num --limit 20
    python run_jev.py --variant num
    python run_jev.py --variant text
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE.parent / ".env"   # one line TYPESAFE_API_KEY=... (or set the environment variable)
PRICE_PER_MTOK = 0.042

QUESTIONS = {
    "next_up": ("The next 5-minute candle, which starts right after the snapshot described in the "
                "state, will close higher than it opens."),
    "next_down": ("The next 5-minute candle, which starts right after the snapshot described in the "
                  "state, will close lower than it opens."),
    "ctrl_rsi": "In the snapshot, RSI(14) on the 5-minute chart is above 50.",
    "ctrl_ema9": "In the snapshot, the close is above the 9-period EMA on the 5-minute chart.",
}


def load_api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TYPESAFE_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["TYPESAFE_API_KEY"] = val
                    return val
    return None


def done_ids(path):
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                ids.add(r["id"])
    return ids


async def run(samples, variant, out_path, concurrency, model):
    from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy, TypeSafeAPIError
    questions = {k: Noul(instructions=v) for k, v in QUESTIONS.items()}
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    st = {"ok": 0, "err": 0, "tok": 0}
    t0 = time.time()
    async with AsyncTypeSafeClient(model=model, retry=RetryPolicy(max_retries=5, backoff_max=20)) as client:
        async def one(s):
            async with sem:
                rec = {"id": s["id"], "variant": variant}
                try:
                    t = time.time()
                    resp = await client.system_one(s[f"state_{variant}"], questions)
                    rec.update({"model": resp.model, "latency_ms": round((time.time() - t) * 1000),
                                "input_tokens": resp.usage.input_tokens,
                                **{f"p_{k}": resp.nouls[k].noul for k in QUESTIONS}})
                    st["ok"] += 1
                    st["tok"] += resp.usage.input_tokens
                except TypeSafeAPIError as e:
                    rec["error"] = f"{type(e).__name__} status={getattr(e, 'status', None)}: {e}"
                    st["err"] += 1
                    if getattr(e, "status", None) in (401, 403):
                        raise
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    st["err"] += 1
                async with lock:
                    with open(out_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    n = st["ok"] + st["err"]
                    if n % 50 == 0 or n == len(samples):
                        print(f"  {n}/{len(samples)} ok={st['ok']} errors={st['err']} "
                              f"{st['tok']:,} tok ≈ ${st['tok']/1e6*PRICE_PER_MTOK:.4f} "
                              f"{time.time()-t0:.0f}s", flush=True)
        await asyncio.gather(*(one(s) for s in samples))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["num", "text"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    samples = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out = HERE / f"results_{a.variant}.jsonl"
    skip = done_ids(out)
    todo = [s for s in samples if s["id"] not in skip][: a.limit]
    q_chars = sum(len(v) for v in QUESTIONS.values())
    # measured on jev-1.13.0: ~790 input tokens per request, about chars / 1.5
    est = sum(len(json.dumps(s[f"state_{a.variant}"])) + q_chars for s in todo) / 1.5
    print(f"Samples: {len(samples)} (done {len(skip)}, to run {len(todo)})  variant={a.variant}")
    print(f"Estimate ~{est:,.0f} tokens ≈ ${est/1e6*PRICE_PER_MTOK:.4f}  -> {out.name}")

    if a.dry_run:
        s = next(x for x in samples if x["split"] == "test_rand")
        print("\n--- Sample request ---")
        print(json.dumps({"model": a.model, "state": s[f"state_{a.variant}"],
                          "questions": {k: {"type": "noul", "instructions": v} for k, v in QUESTIONS.items()}},
                         indent=1, ensure_ascii=False))
        print(f"\nSample {s['id']}: true next_up={s['y']}  boosting p_up={s['p_boost']}  "
              f"ctrl_rsi={s['ctrl_rsi_above_50']} ctrl_ema9={s['ctrl_close_above_ema9']}")
        return
    if not todo:
        print("Nothing left to run.")
        return
    if not load_api_key():
        raise SystemExit(f"Missing TYPESAFE_API_KEY (environment variable or {ENV}).")
    st = asyncio.run(run(todo, a.variant, out, a.concurrency, a.model))
    print(f"Done: ok={st['ok']} errors={st['err']}. Failed samples are retried on the next run.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
