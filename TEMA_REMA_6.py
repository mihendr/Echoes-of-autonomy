#!/usr/bin/env python
# coding: utf-8

from openai import OpenAI
import streamlit as st

backup_client = OpenAI(
    api_key=st.secrets["OPENAI_KEY_FALLBACK"],
    base_url="https://api.groq.com/openai/v1"
)

client = OpenAI(
    api_key=st.secrets["OPENAI_KEY_PRIMARY"],
    base_url="https://api.together.xyz/v1"
)

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

st.title("Narrative Generator")
st.markdown("""
**Type [1–8]:**__1. Cathartic Cycle__2. Existential Spiral__3. Harmonic Duo-motif__4. Heroic Rise  
5. Tragic Counterpoint__6. Meditative Cycle__7. Introspective Fold__8. Humoristic Effect
""")
initial_sentence = st.text_input("Initial phrase:", value="")



rules_type_raw = st.text_input("")
rules_seq = None

if rules_type_raw:
    try:
        rules_type = int(rules_type_raw)
        if rules_type not in range(1, 9):
            st.error("❌ Невалидно число. Моля въведи стойност от 1 до 8.")
        else:
            rules_seq = {
                1: ['B','D','A','C','D','B','C'],
                2: ['D','E','A','E','C','B','C'],
                3: ['B','C','B','C','A','C'],
                4: ['A','E','D','B',"D",'C'],
                5: ['E','D','E','B','A','C'],
                6: ['C','C','A','B','C','B','A'],
                7: ['D','E','B','E','C','A'],
                8: ['C','A','B','C','A','D']
            }[rules_type]
    except ValueError:
        st.error("❌ Моля въведи цяло число между 1 и 8.")

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

Example (illustrative only):
The message unsettled the crew (C)

Hard prohibitions:
No extra commentary, no explanations, no metadata.
No missing rule markers.
No ambiguous pronouns when multiple antecedents are possible.
"""

if st.button("Generate") and rules_seq:
    user_prompt = f'''
    Initial sentence: "{initial_sentence}"
    RULES sequence: {rules_seq}
    Generate exactly {len(rules_seq)} lines, one per rule, following the constraints.
    '''

    try:
        with st.spinner("Генериране на разказ..."):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": narrative_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.0,
                max_tokens=220
            )
    except:
        with st.spinner("Генериране на разказ (backup)..."):
            resp = backup_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": narrative_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.0,
                max_tokens=220
            )

    lines = [ln.strip() for ln in resp.choices[0].message.content.strip().splitlines() if ln.strip()]
    import re
    cleaned_lines = [re.sub(r"\s*\((?:A|B|C|D|E)\)\s*$", "", ln) for ln in lines]
    st.session_state["narrative"] = "\n".join(lines)

if "narrative" in st.session_state:
    st.text(st.session_state["narrative"])

if st.button("Music") and rules_seq:
    audio_path = f"audio/S{rules_type}_full_instruments.mp3"
    try:
        st.markdown(
            f"""
            <audio autoplay>
                <source src="{audio_path}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.warning(f"Audio file not found at {audio_path}. Make sure the file exists.")

