# Judge Validation Review

Local multi-reviewer UI for checking the consensus judge against people. A
reviewer reads one finished conversation and decides one thing: does the
model's final answer match the reference answer?

The eleven judges' verdicts are never sent to the page. A reviewer who could
see them would be checking the judges rather than grading the conversation.

## 1. Draw the sample

From `vendor/conv-smi`:

```powershell
python sample_judge_validation.py --n 400
```

It writes `vendor/conv-smi/human_validation/sample.json`: 400 conversations,
25 per condition, models spread inside each condition, with transcripts,
questions, reference answers, and repo-relative image paths. The first run
reads the whole post-hoc tree and caches the judge votes, so a re-draw with a
different `--n` or `--seed` is quick.

## 2. Review

```powershell
powershell -ExecutionPolicy Bypass -File vendor\judge-review\run.ps1 -Port 8794
```

Reviewers are not fixed in advance. Whoever opens the page types a name; a name
not used before starts a new file. Several people can work at the same time,
and each sees the same conversations in the same order.

Instructions sit in a panel at the top, open on the first conversation and
collapsible after that. Images in a turn appear inline and enlarge on click.
Keyboard: `1` correct, `2` wrong, `3` unclear, `←` back.

## Output

Every decision is written immediately, atomically, to both:

```text
vendor/judge-review/runtime/verdicts_<name>.json     one per reviewer
data/derived/judge-validation/verdicts.json          all reviewers, combined
```

The combined file is rebuilt from the per-reviewer files on every write, so a
file copied in from another machine is picked up without a merge step.

Closing the tab loses nothing. Reopening returns to the first conversation that
reviewer has not decided, and **Previous** changes an answer already given.

## Environment

`JUDGE_REVIEW_SAMPLE`, `JUDGE_REVIEW_RUNTIME` and `JUDGE_REVIEW_EXPORT`
override the three paths above.
