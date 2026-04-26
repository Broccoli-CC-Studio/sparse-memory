# Hard cross-doc QA benchmark — finding (2026-04-26 09:58)

Ran `bench_hard_qa.py` against running MSA server (37 docs, OLD code path). 10 questions designed to require composing facts from 2+ docs. Keyword-based scoring on the answer text.

## Raw scores

| | Count | Notes |
|---|---|---|
| Full pass | 3 | Q6, Q7, Q9 — all single-doc lookup, not real cross-doc |
| Partial | 6 | mix of real partial + scoring artifacts |
| Fail | 1 | Q4, but actually scored wrong |
| Avg keyword recall | 63% | misleading, see corrections |
| Avg latency | 3.1s | warm queries |

## Scoring artifacts (test bugs, not model bugs)

- **Q3** "MSA 和 vector RAG 的根本差別" — model said "稀疏Transformer", I checked for "Sparse Transformer". Chinese-vs-English keyword mismatch. Real result: full pass.
- **Q4** "RTX 3090 能存多少 token" — model said "175,000 token", I checked "175K". Numerical formatting mismatch. Real result: full pass.
- **Q8** "Anthropic 2026 年營收" — model said "300 亿美元 / 25 亿美元 ARR", I checked "30B". Currency formatting mismatch. Real result: full pass.

After correction: 6 full pass, 3 partial, 1 fail. Avg ~80%.

## Real model failures

**Q5 retrieval miss**: "agent 學的網課有哪些 名字是" — model said "未命名為網課" (couldn't find specific course names) even though `project_agent_courses` doc is in the bank with gstack/caveman/hermes explicitly named. The keyword scoring caught caveman+hermes only because they appeared in the model's hedge ("project_caveman: 一個學習項目"), not because the model actually answered. **Real failure: model gave a confident-sounding "I don't know" when the data was retrievable.**

**Q1 context bleed**: "菜花用什麼硬體和什麼工具鏈" — answer quoted distillation pipeline (multi-teacher / QLoRA / MIT/Apache) which is from `project_distillation` doc, not the tooling docs. Top-k retrieval pulled the wrong neighborhood. The hardware piece was right (Docker / RTX 3090), the tooling piece bled in irrelevant content.

**Q10 incomplete**: "uv 跟 source venv activate 哪個好" — answer paraphrased the gist but did not quote the explicit "lockfile / shell state / 進程隔離" reasoning that doc 6 lists. Model summarized rather than retrieved verbatim points.

## Implications

1. The previous "5/5 cross-doc" benchmark was easier than I framed it. Those questions were single-doc lookup wrapped in cross-doc framing. Real multi-fact composition is harder.

2. MSA's retrieval has a real failure mode: top-k can pick the wrong doc neighborhood when topic boundaries are fuzzy. Q1 distillation-bleed and Q5 missed-retrieval are different shapes of this.

3. The "hallucination when bank lacks answer" finding from earlier (`What is MSA?`) generalizes: the model also hallucinates "I don't know" when the answer IS in the bank but retrieval missed it. Confident negatives matter.

4. Keyword-based eval is fragile across languages and number formats. A real benchmark needs semantic match (LLM-as-judge or embedding similarity).

## Honest summary for shipping

- Single-doc lookup: ~100% (3/3 in this set, plus 5/5 from earlier 78-doc and 37-doc runs)
- Two-doc composition: estimated 50-70% on real attempts (corrected scoring)
- Confident negative (says "I don't know" when answer is retrievable): observed once in 10
- Confident wrong (says "X" when X isn't in bank): observed once in earlier "What is MSA?" probe
- Latency: 3.1s avg warm, range 1-7s, dominated by generation length

These are the numbers to put in the next version of the README, not the breezy "5/5 cross-doc" line.

## Next steps (not in this cron)

1. Semantic eval: pipe answers + ground truth into a judge LLM, get pass/fail per question. ~100 lines.
2. Larger benchmark: 50+ questions, balanced across single/two/three-doc complexity. Real numbers.
3. Retrieval debugging: expose `recall_topks` from query so we can see which docs the model attended to. Already designed in earlier xmemory_finding suggestion.
4. Hallucination guard: post-process answer to flag "answer mentions X but no retrieved doc contains X" cases.
