# generator.py
# Без хардкоднати ключове — използва get_openai_keys() от config.py
import re
from typing import List, Tuple, Optional

from config import get_openai_keys

# По подразбиране използваме същите base_url-ове/модел като в оригиналния ти скрипт.
DEFAULT_PRIMARY_BASE = "https://api.together.xyz/v1"
DEFAULT_BACKUP_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

narrative_system_prompt = """You are a micro‑narrative generator that MUST enforce topic–comment (theme–rheme) linking via strict local UD-style constraints and explicit coreference. Violations are not allowed.

Global constraints:
Output EXACTLY N lines, where N = length of RULES sequence.
Language: concise, natural English; coherent with the Initial text.
Each line is ONE short clause only (no coordination, no extra sentences).
Append the rule marker in parentheses at the end of the line.
Each line MUST include explicit subject/object annotations and the rule marker, using this exact schema:
Subject Verb Object (RULE)
If a role is not semantically present, use a minimal dummy "—" but only if permitted by the rule; otherwise REWRITE to include an explicit noun phrase or an unambiguous pronoun.
No bullets, numbers, or extra text. One line per rule.

Initial anchoring:
Initial text may have 1–3 sentences.
Define InitialSubject = last explicitly mentioned grammatical subject (nsubj) in Initial text.
Define InitialObject = last explicitly mentioned direct/indirect object (obj/iobj) in Initial text.
Line 1 applies its rule relative to the Initial text using InitialSubject/InitialObject.

Rule semantics (local evaluation: each line relative to the immediately previous line):
A (nsubj_1 → obj/iobj_2): Set CURRENT OBJECT to corefer to PREVIOUS SUBJECT. CURRENT SUBJECT is otherwise free but coherent.
B (obj/iobj_1 → nsubj_2): Set CURRENT SUBJECT to corefer to PREVIOUS OBJECT/IOBJECT. CURRENT OBJECT is otherwise free but coherent.
C (nsubj_1 == nsubj_2): Keep the SAME SUBJECT entity as in the previous line (exact coreference).
D (nsubj_1 ≠ nsubj_2): Switch to a DIFFERENT SUBJECT entity; name it explicitly at least once before pronouns.
E ((nsubj_1 → obj/iobj_2) AND (obj/iobj_1 → nsubj_2)): CURRENT SUBJECT corefers to PREVIOUS OBJECT; CURRENT OBJECT corefers to PREVIOUS SUBJECT.

Pronoun policy:
Prefer pronouns when coreference is unambiguous; otherwise use explicit noun phrases.
Coreference MUST be exact and checkable from context (no ambiguous "they/it" if multiple candidates exist).

Backoff policy (only when needed):
For B/E, if the previous line lacks an explicit object/indirect object, use the most recently mentioned explicit object earlier in the sequence; if none exists yet, use InitialObject.
If no valid backoff exists, REWRITE the previous line minimally to introduce a valid explicit object, then continue (but still produce exactly N lines overall).

Well‑formedness constraints per line:
Subject and object must be surface-realized in the output, unless "—" is allowed by the rule and remains coherent.

The rule marker MUST match the rule for that line: one of (A), (B), (C), (D), (E).
Maintain lexical coherence with the Initial text.

Self‑verification (MANDATORY before finalizing):
For i from 1..N:
1) Verify marker equals RULES[i].
2) Verify subject/object constraints vs. line i-1 (or Initial text for i=1).
3) Verify unambiguous coreference for required links (A/B/E/C); for D verify subject switch.
4) If any check fails, REWRITE the minimal necessary line(s) to satisfy constraints and re‑run checks.
5) Only output when ALL checks pass.

I/O format:
Input variables provided in the user message:
Initial sentence(s)
RULES sequence (array of "A"|"B"|"C"|"D"|"E")
N = len(RULES)

Output: EXACTLY N lines, each in the format:
Subject Verb Object {rules_seg[i]}

Hard prohibitions:
No extra commentary, no explanations, no metadata.
No missing rule markers.
No ambiguous pronouns when multiple antecedents are possible.
"""

def _make_clients(primary_key: Optional[str], fallback_key: Optional[str],
                  primary_base: str = DEFAULT_PRIMARY_BASE,
                  fallback_base: str = DEFAULT_BACKUP_BASE):
    """
    Връща (primary_client, backup_client). Клиентите могат да бъдат None ако няма ключ.
    Не логва ключовете.
    """
    primary_client = None
    backup_client = None
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI library not installed or not importable: " + str(e))

    if primary_key:
        primary_client = OpenAI(api_key=primary_key, base_url=primary_base)
    if fallback_key:
        backup_client = OpenAI(api_key=fallback_key, base_url=fallback_base)
    return primary_client, backup_client

def _call_client_chat(client, model: str, system: str, user: str,
                      temperature: float, top_p: float, max_tokens: int):
    """
    Връща raw response object или хвърля Exception.
    """
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

def generate_narrative(initial_sentence: str, rules_seq: List[str],
                       model: str = DEFAULT_MODEL,
                       temperature: float = 0.2, top_p: float = 0.0, max_tokens: int = 220) -> List[str]:
    """
    Основна функция: връща списък с точно N реда (стрингове).
    Използва ключовете от config.get_openai_keys().
    Опитва първо primary, при грешка или липса — fallback.
    Ако и двата липсват, хвърля ValueError.
    """
    primary_key, fallback_key = get_openai_keys()
    if not primary_key and not fallback_key:
        raise ValueError("No API keys found. Add OPENAI_KEY_PRIMARY and/or OPENAI_KEY_FALLBACK to Streamlit secrets or set environment variables.")

    primary_client, backup_client = _make_clients(primary_key, fallback_key)

    user_prompt = f'''
Initial sentence: "{initial_sentence}"
RULES sequence: {rules_seq}
Generate exactly {len(rules_seq)} lines, one per rule, following the constraints.
'''

    # Опитваме primary client първо
    last_exception = None
    for client in (primary_client, backup_client):
        if client is None:
            continue
        try:
            resp = _call_client_chat(client, model=native_str(model:=model),
                                     system=narrative_system_prompt, user=user_prompt,
                                     temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            # parse text safely
            content = ""
            # handle possible response shape
            try:
                content = resp.choices[0].message.content
            except Exception:
                # some clients return a different structure
                content = getattr(resp, "content", "") or str(resp)
            lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
            # remove trailing rule markers from cleaned lines if needed (we keep original lines)
            return lines
        except Exception as e:
            last_exception = e
            # опитваме следващия клиент
            continue

    # Ако стигнем тук — и primary, и backup са пробвани и са дали грешка
    raise RuntimeError("All clients failed. Last error: " + (str(last_exception) if last_exception else "unknown"))

# helper for Python <-> f-string weirdness in some runtimes
def native_str(x):
    return x
        

if __name__ == "__main__":
    # fallback за команден ред/бърз тест (не включва ключове в кода)
    try:
        initial = input("Initial phrase: ")
        rules_type = int(input("Choose rules type [1-8]: "))
        mapping = {
            1: ['B','D','A','C','D','B','C'],
            2: ['D','E','A','E','C','B','C'],
            3: ['B','C','B','C','A','C'],
            4: ['A','E','D','B','D','C'],
            5: ['E','D','E','B','A','C'],
            6: ['C','C','A','B','C','B','A'],
            7: ['D','E','B','E','C','A'],
            8: ['C','A','B','C','A','D']
        }
        rules_seq = mapping.get(rules_type)
        if not rules_seq:
            print("Invalid type")
        else:
            lines = generate_narrative(initial, rules_seq)
            print("\n".join(lines))
    except Exception as exc:
        print("Error:", exc)
