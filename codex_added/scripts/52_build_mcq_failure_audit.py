#!/usr/bin/env python3
# Added by Codex: build a concise audit pack for remaining V6 MCQ failures.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from math_comp.data import index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.prompts import format_options
from math_comp.scoring import extract_answer_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build a compact audit report for V6 MCQ failures.')
    p.add_argument('--data', required=True)
    p.add_argument('--raw', required=True)
    p.add_argument('--v6', required=True)
    p.add_argument('--extractor-details', default=None)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-md', required=True)
    p.add_argument('--max-snippet-chars', type=int, default=1600)
    return p.parse_args()


def tail_snippet(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return '[tail]\n' + text[-max_chars:]


def head_snippet(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n[head truncated]'


def option_mentions(text: str, option_count: int) -> dict[str, int]:
    allowed = [chr(65+i) for i in range(option_count)]
    upper = text.upper()
    return {letter: len(re.findall(rf'\b{letter}\b', upper)) for letter in allowed}


def rough_family(question: str) -> str:
    q = question.lower()
    checks = [
        ('integral', ['integral', '\\int', 'int_']),
        ('geometry', ['triangle', 'parabola', 'hexagon', 'altitude', 'vertex', 'focus', 'directrix']),
        ('sequence', ['sequence', 'recurrence', 'v_n', 'a_1', 'a_{i+1}']),
        ('combinatorics/counting', ['number of', 'positive integers', 'choose', 'integers', 'multiples']),
        ('probability', ['random variables', 'probability', 'independent']),
        ('calculus derivative', ['derivative', "y ="]),
        ('complex/log/product', ['log_2', 'complex', 'determinant', 'omega']),
        ('inequality/optimization', ['positive real', 'maximum', 'minimum']),
    ]
    for name, needles in checks:
        if any(n in q for n in needles):
            return name
    return 'other'


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    raw = index_by_id(read_jsonl(args.raw))
    v6 = index_by_id(read_jsonl(args.v6))
    details = {}
    if args.extractor_details and Path(args.extractor_details).exists():
        details = index_by_id(read_jsonl(args.extractor_details))

    rows = []
    for item in data:
        item_id = int(item['id'])
        if not is_mcq(item):
            continue
        pred = v6.get(item_id)
        if not pred or bool(pred.get('correct')):
            continue
        raw_row = raw[item_id]
        raw_response = str(raw_row.get('response', ''))
        detail = details.get(item_id, {})
        row = {
            'id': item_id,
            'family': rough_family(str(item.get('question', ''))),
            'gold': item.get('answer'),
            'v6_answer_key': pred.get('answer_key') or extract_answer_key(item, str(pred.get('response', '')), strict=True),
            'raw_answer_key': extract_answer_key(item, raw_response, strict=True),
            'raw_hit_token_limit': bool(raw_row.get('hit_token_limit')),
            'raw_generated_tokens': int(raw_row.get('generated_tokens') or 0),
            'extractor_answer_key': (detail.get('extractor_letter') or {}).get('answer_key'),
            'extractor_correct': (detail.get('extractor_letter') or {}).get('correct'),
            'logprob_answer_key': (detail.get('option_logprob') or {}).get('answer_key'),
            'logprob_correct': (detail.get('option_logprob') or {}).get('correct'),
            'option_mentions_tail': option_mentions(raw_response[-6000:], len(item.get('options') or [])),
            'question': item.get('question'),
            'options': item.get('options'),
            'raw_head': head_snippet(raw_response, args.max_snippet_chars),
            'raw_tail': tail_snippet(raw_response, args.max_snippet_chars),
            'v6_response': pred.get('response', ''),
        }
        rows.append(row)

    write_jsonl(args.output_jsonl, rows)

    lines = []
    lines.append('# V6 MCQ Failure Audit Pack')
    lines.append('')
    lines.append(f'Total V6-wrong MCQs: `{len(rows)}`.')
    lines.append('')
    fam_counts = {}
    for row in rows:
        fam_counts[row['family']] = fam_counts.get(row['family'], 0) + 1
    lines.append('## Family Counts')
    lines.append('')
    for family, count in sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'- `{family}`: `{count}`')
    lines.append('')
    lines.append('## Compact Index')
    lines.append('')
    lines.append('| ID | Family | Gold | V6 | Raw | Cap | Extractor | Logprob |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['family']} | {row['gold']} | {row['v6_answer_key']} | "
            f"{row['raw_answer_key']} | {row['raw_hit_token_limit']} | {row.get('extractor_answer_key') or ''} | "
            f"{row.get('logprob_answer_key') or ''} |"
        )
    lines.append('')
    lines.append('## Per-Item Snippets')
    for row in rows:
        lines.append('')
        lines.append(f"### ID {row['id']} ({row['family']})")
        lines.append('')
        lines.append(f"Gold `{row['gold']}`, V6 `{row['v6_answer_key']}`, raw `{row['raw_answer_key']}`, cap `{row['raw_hit_token_limit']}`.")
        if row.get('extractor_answer_key') or row.get('logprob_answer_key'):
            lines.append(f"Extractor `{row.get('extractor_answer_key')}`, logprob `{row.get('logprob_answer_key')}`.")
        lines.append('')
        lines.append('Question:')
        lines.append('')
        lines.append(str(row['question']).replace('\n', ' '))
        lines.append('')
        lines.append('Options:')
        lines.append('')
        lines.append(format_options(row['options'] or []))
        lines.append('')
        lines.append('Raw trace tail:')
        lines.append('')
        lines.append('```text')
        lines.append(row['raw_tail'])
        lines.append('```')
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} MCQ audit rows to {args.output_jsonl} and {args.output_md}')


if __name__ == '__main__':
    main()
