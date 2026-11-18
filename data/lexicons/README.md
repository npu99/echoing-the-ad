# data/lexicons — the Big Two dictionary

## Source

Pietraszkiewicz, A., Formanowicz, M., Gustafsson Sendén, M., Boyd, R. L.,
Sikström, S., & Sczesny, S. (2019). The big two dictionaries: Capturing agency
and communion in natural language. *European Journal of Social Psychology*,
49(5), 871-887. https://doi.org/10.1002/ejsp.2561

The dictionary files themselves are the supplementary material of that article,
downloaded on 2026-06-19 from https://osf.io/p7fzb/.

## Files

| Path | Name | Agency | Communion | Role |
|------|------|-------:|----------:|------|
| `official/a_AgencyCommunion.dic` | dictionary A | 194 | 186 | **main analysis from 2026-10-10** |
| `official/b_AgencyCommunion.dic` | dictionary B | 64 | 65 | the alternative dictionary of the online material |
| `agency_communion.dic` | A ∪ B | 227 | 218 | robustness variant; the main analysis until 2026-10-10 |

## What they hold

**Dictionary A** (`official/a_AgencyCommunion.dic`) is Table A2 of the paper. The
paper states 192 and 184 entries; the file spells out two of them instead of
holding them as a wildcard — `think*` as think, thinking, thinks (3 instead of 1)
and `team*` as team, teams, teamwork (3 instead of 1) — which makes 194 and 186.

**Dictionary B** (`official/b_AgencyCommunion.dic`) is the alternative dictionary
from the online material of the paper (the OSF file `b_AgencyCommunion.dic`).

**The earlier file** (`agency_communion.dic`) is the union A ∪ B with 227 and 218
entries. It was the main analysis until 2026-10-10 and is the robustness variant
from then on. The file is left unchanged, because the earlier results refer to it.

## How the entries are assigned

`config/lexicon_entry_blocks.csv` lists all 445 (term, cat) pairs with the column
`block`: `table_A2` (380 pairs, in A) and `appended` (65 pairs, in B only).

B without A holds 65 pairs: 33 agency and 32 communion.

## SHA-256

| File | SHA-256 |
|------|---------|
| `official/a_AgencyCommunion.dic` | `90e59007344c2b794db8c4a5e62098782c7fa42519a50e5c132d83877765bbcc` |
| `official/b_AgencyCommunion.dic` | `11e2d3a90a902fcfb7c0f8440392fd86ea47c46f9357e104723b525c4de53c6b` |
| `agency_communion.dic`            | `5131e98a0b688ac51a05ffb60d61e107fe89be1a43e7f2314284c5bdb6ec61ca` |

## How the united file came about (found on 2026-10-10)

`agency_communion.dic` is the union of A and B: `set(A) | set(B) == set(AB)` is
True, verified against `config/lexicon_entry_blocks.csv`. B without A holds
exactly 65 (term, cat) pairs, 33 agency and 32 communion.

## The achievement list

Two versions:

| Version | Entries | Wildcards it adds against the other |
|---------|---------|-------------------------------------|
| A       | 23      | without `excellent*` and `outstanding*`, which are in B only |
| AB      | 25      | with `excellent*` and `outstanding*` |

The exact entries, in both versions (12): earn, earned, earning, earns, mastered,
mastering, mastery, overcome, overcomes, win, wins, won.

The wildcards of A (11): accomplish, achiev, attain, masterful, prevail, succeed,
success, triumph, victor, winner, winning.

The wildcards of AB (13): those of A plus excellent and outstanding.

## Use

```python
from src.lexicon import load_lexicons
lex = load_lexicons(version="A")   # the main analysis
lex = load_lexicons(version="AB")  # the robustness variant
```

A call without `version` raises a `ValueError`.
