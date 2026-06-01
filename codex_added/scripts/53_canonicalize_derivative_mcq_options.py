#!/usr/bin/env python3
# Added by Codex: guarded derivative-MCQ equivalent-option canonicalizer.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import sympy as sp

CODEX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judger import Judger
from math_comp.data import has_gold, index_by_id, is_mcq, read_jsonl, write_jsonl
from math_comp.final_answer import final_answer_diagnostics, normalize_final_response
from math_comp.scoring import extract_answer_key, score_item, summarize_results

X = sp.symbols('x')
SYMBOLS = {ch: sp.symbols(ch) for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'}
SYMBOLS.update(
    {
        'pi': sp.pi,
        'e': sp.E,
        'i': sp.I,
        'I': sp.I,
        'x': X,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sqrt': sp.sqrt,
        'log': sp.log,
        'ln': sp.log,
        'Abs': sp.Abs,
        'csc': lambda z: 1 / sp.sin(z),
        'sec': lambda z: 1 / sp.cos(z),
        'cot': lambda z: 1 / sp.tan(z),
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Canonicalize equivalent derivative MCQ option forms.')
    p.add_argument('--data', required=True)
    p.add_argument('--responses', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--audit-output', required=True)
    p.add_argument('--score', action='store_true')
    return p.parse_args()


def normalize_option_text(text: str) -> str:
    text = str(text).strip().strip('$')
    text = text.replace('−', '-').replace('∞', 'oo')
    text = text.replace('\\cdot', '*').replace('cdot', '*')
    text = text.replace('\\left', '').replace('\\right', '')
    text = text.replace('\\,', '')
    text = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', text)
    for name in ('sin', 'cos', 'tan', 'sqrt', 'ln', 'log', 'csc', 'sec', 'cot'):
        text = text.replace('\\' + name, name)
    text = text.replace('ln', 'log')
    for _ in range(20):
        new_text = re.sub(r'\\?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
        if new_text == text:
            break
        text = new_text
    text = text.replace('^', '**')
    text = text.replace('{', '(').replace('}', ')')
    text = text.replace('|', '')
    # Insert simple implicit multiplication in cases like 2x or 2*x already absent.
    text = re.sub(r'(?<=[0-9])(?=[A-Za-z])', '*', text)
    text = re.sub(r'(?<=[A-Za-z])(?=[0-9])', '*', text)
    return text


def parse_option(text: str) -> Any | None:
    normalized = normalize_option_text(text)
    if not any(token in normalized for token in ('/', '*', '**', 'sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'sqrt')):
        return None
    try:
        return sp.sympify(normalized, locals=SYMBOLS)
    except Exception:
        return None


def equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        if sp.simplify(sp.trigsimp(left - right)) == 0:
            return True
    except Exception:
        pass
    try:
        expr = left - right
        variables = sorted(expr.free_symbols, key=lambda sym: sym.name)
        for value in (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(5, 4)):
            subs = {sym: value for sym in variables}
            if abs(complex(expr.subs(subs).evalf())) > 1e-8:
                return False
        return True
    except Exception:
        return False


def canonical_option_index(options: list[str], selected_index: int) -> tuple[int, list[int]]:
    parsed = [parse_option(option) for option in options]
    if selected_index < 0 or selected_index >= len(options) or parsed[selected_index] is None:
        return selected_index, [selected_index]
    equivalent_indices = [
        idx
        for idx, expression in enumerate(parsed)
        if expression is not None and equivalent(parsed[selected_index], expression)
    ]
    if len(equivalent_indices) <= 1:
        return selected_index, equivalent_indices
    # Prefer the simplest written equivalent. This fixes derivative options like
    # cos(x)^5 * cos(x) -> cos(x)^6 without touching non-derivative MCQs.
    best = min(equivalent_indices, key=lambda idx: (len(str(options[idx])), str(options[idx]), idx))
    return best, equivalent_indices


def main() -> None:
    args = parse_args()
    data = read_jsonl(args.data)
    data_by_id = index_by_id(data)
    responses = read_jsonl(args.responses)
    judger = Judger(strict_extract=False)
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for row in responses:
        item = data_by_id[int(row['id'])]
        out = dict(row)
        if is_mcq(item) and 'derivative' in str(item.get('question', '')).lower():
            key = extract_answer_key(item, str(row.get('response', '')), strict=True)
            if len(key) == 1:
                selected_index = ord(key) - ord('A')
                canonical_index, group = canonical_option_index(list(item.get('options') or []), selected_index)
                if canonical_index != selected_index:
                    new_key = chr(ord('A') + canonical_index)
                    out['pre_derivative_canonicalizer_response'] = out.get('response', '')
                    out['pre_derivative_canonicalizer_answer_key'] = key
                    out['response'] = f'\\boxed{{{new_key}}}'
                    out['answer_key'] = new_key
                    out['derivative_canonicalizer_changed'] = True
                    out['derivative_canonicalizer_equivalent_options'] = [chr(ord('A') + idx) for idx in group]
                    out.update(final_answer_diagnostics(item, out['response']))
                    audit_rows.append(
                        {
                            'id': int(item['id']),
                            'old_answer_key': key,
                            'new_answer_key': new_key,
                            'equivalent_options': [chr(ord('A') + idx) for idx in group],
                            'old_option': item['options'][selected_index],
                            'new_option': item['options'][canonical_index],
                        }
                    )
        if args.score and has_gold(item):
            out['gold'] = item['answer']
            out['correct'] = score_item(judger, item, str(out.get('response', '')))
        output_rows.append(out)

    write_jsonl(args.output, output_rows)
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        'num_changes': len(audit_rows),
        'changes': audit_rows,
    }
    if args.score:
        audit['summary'] = summarize_results(output_rows)
        print(json.dumps(audit['summary'], indent=2, sort_keys=True))
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Wrote {len(output_rows)} rows to {args.output}; changes={len(audit_rows)}')


if __name__ == '__main__':
    main()
