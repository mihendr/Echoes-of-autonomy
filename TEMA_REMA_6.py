import spacy
import google.generativeai as genai
import re
import music21
from typing import List, Tuple
from openai import OpenAI

from music21 import stream, harmony, key, interval, note

# --- minimal Streamlit preparation ---
try:
    import streamlit as st
except Exception:
    st = None




# =========================
# === ORIGINAL CONSTANTS ==
# =========================
ADVERB_GUIDELINES_FOR_MAJOR = {
    "triola min": "Your reply must include at least one adverb of MANNER (e.g. well, quickly, slowly, carefully, badly, easily, loudly, happily, quietly, clearly, honestly, confidently, sadly, successfully, angrily, accidentally, gently, politely, perfectly, softly).",
    "m7": "Your reply must include at least one adverb of DEGREE (e.g. very, too, quite, rather, so, such, extremly, fairly, somewhat, barely, completely, entirely, almost).",
    "Gb": "Your reply must include at least one adverb of PLACE (e.g. here, there, inside, outsude, everywhere, somewhere, anywhere, nowhere, upstairs, downstairs, around, near, far).",
    "pause min": "Your reply must include at least one adverb of FREQUENCY (e.g. always, often, rarely, sometimes, never, usually, daily).",
    "Bb": "Your reply must include at least one INTERROGATIVE word (e.g. what, who, why, how, where, when, which).",
    "Eb": "Your reply must include at least one adverb of TIME (e.g. later, now, then, today, tomorrow, yesterday, soon, immediately, already, finally).",
    "Ab": "Your reply must include at least one adverb of REASON (e.g. therefore, thus, hence, so, accordinglyq since).",
    "Db": "Your reply must include at least one adverb of CERTAINTY (e.g. definitely, certainly, surely, probably, possibly, perhaps, obvously, undoubtedly, clearly)"
}


CONJUNCTION_GUIDELINES_FOR_MAJOR = {
    "F#": "Your reply must include at least one COORDINATING conjunction (e.g. and, but, or, yet, so).",
    "C#": "Your reply must include at least one SUBORDINATING conjunction for CONCESSION (e.g. although, even though, while).",
    "G#": "Your reply must include at least one SUBORDINATING conjunction for PLACE (e.g. where, wherever).",
    "pause maj": "Your reply must include at least one SUBORDINATING conjunction for CONDITION (e.g. if, unless, in case, otherwise, supposing, even if).",
    "triola maj": "Your reply must include at least one SUBORDINATING conjunction for PURPOSE (e.g. so that, in order that).",
    "maj 7": "Your reply must include at least one SUBORDINATING conjunction for TIME (e.g. when, after, before, while, until, since, as soon as).",
    "A#": "Your reply must include at least one SUBORDINATING conjunction for REASON/CAUSE (e.g. because, since, as).",
    "D#": "Your reply must include at least one CORRELATIVE conjunction (e.g. either...or, neither...nor, both...and)."
}

ADVERB_GUIDELINES_FOR_MINOR = {
    "Db": "Your reply must include at least one adverb of MANNER (e.g. well, quickly, slowly, carefully, badly, easily, loudly, happily, quietly, clearly, honestly, confidently, sadly, successfully, angrily, accidentally, gently, politely, perfectly, softly).",
    "Ab": "Your reply must include at least one adverb of DEGREE (e.g. very, too, quite, rather, so, such, extremly, fairly, somewhat, barely, completely, entirely, almost).",
    "Eb": "Your reply must include at least one adverb of PLACE (e.g. here, there, inside, outsude, everywhere, somewhere, anywhere, nowhere, upstairs, downstairs, around, near, far).",
    "pause min": "Your reply must include at least one adverb of FREQUENCY (e.g. always, often, rarely, sometimes, never, usually, daily).",
    "Gb": "Your reply must include at least one INTERROGATIVE word (e.g. what, who, why, how, where, when, which).",
    "m7": "Your reply must include at least one adverb of TIME (e.g. later, now, then, today, tomorrow, yesterday, soon, immediatelyq already, finally).",
    "triola min": "Your reply must include at least one adverb of REASON (e.g. therefore, thus, hence, so, accordinglyq since).",
    "Bb": "Your reply must include at least one adverb of CERTAINTY (e.g. definitely, certainly, surely, probably, possibly, perhaps, obvously, undoubtedly, clearly)"
}


CONJUNCTION_GUIDELINES_FOR_MINOR = {
    "D#": "Your reply must include at least one COORDINATING conjunction (e.g. and, but, or, yet, so).",
    "A#": "Your reply must include at least one SUBORDINATING conjunction for CONCESSION (e.g. although, even though, while).",
    "maj7": "Your reply must include at least one SUBORDINATING conjunction for PLACE (e.g. where, wherever).",
    "pause maj": "Your reply must include at least one SUBORDINATING conjunction for CONDITION (e.g. if, unless, in case, otherwise, supposing, even if).",
    "G#": "Your reply must include at least one SUBORDINATING conjunction for PURPOSE (e.g. so that, in order that).",
    "C#": "Your reply must include at least one SUBORDINATING conjunction for TIME (e.g. when, after, before, while, until, since, as soon as).",
    "F#": "Your reply must include at least one SUBORDINATING conjunction for REASON/CAUSE (e.g. because, since, as).",
    "triola maj": "Your reply must include at least one CORRELATIVE conjunction (e.g. either...or, neither...nor, both...and)."
}


# =========================
# === МОДУЛ 1: SYNTAX ====
# =========================
nlp = spacy.load("en_core_web_sm")

def extract_last_subj_obj_with_clauses(doc):
    object_like_roles = {"dobj", "iobj", "attr", "oprd", "pobj", "appos"}

    last_main_subject = None
    last_sub_clause_subject = None
    last_object = None

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            if token.head.dep_ == "ROOT":
                last_main_subject = token.text
            else:
                last_sub_clause_subject = token.text

        if token.dep_ in object_like_roles:
            last_object = token.text

    last_subject = last_main_subject or last_sub_clause_subject

    return last_subject, last_object


# =========================
# === МОДУЛ 2: GEMINI ====
# =========================

def pronoun_to_object(word):
    pronoun_map = {
        "I": "me", "you": "you", "he": "him", "she": "her",
        "it": "it", "we": "us", "they": "them"
    }
    return pronoun_map.get(word, word)


def object_to_pronoun(word):
    pronoun_map = {
        "me": "I", "you": "you", "him": "he", "her": "she",
        "it": "it", "us": "we", "them": "they"
    }
    return pronoun_map.get(word, word)


# === Правила ===

def generate_sentence_rule_A(word_1, word_2, text_history, model_gemini, response_func, rich_rich):
    word_1 = pronoun_to_object(word_1)
    prompt = f"""
        Context:
        {text_history}

        Write one new short sentence where the direct object or indirect object
        or object of a preposition is '{word_1}' and the subject is neither '{word_1}' nor '{word_2}'.
        If it sounds naturally {rich_rich}
        Only final sentence.
       
    """
    #if rich_rich:
       #print(rich_rich)
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text

def generate_sentence_rule_B(word_1, word_2, text_history, model_gemini, response_func, rich_rich):
    word_2 = pronoun_to_object(word_2)
    #print(word_1, word_2)
    prompt = (
        f"""
        Context:\n{text_history}\n\n
        Write one new short sentence where the subject is '{word_2}'
        and the direct object is neither '{word_1}' nor '{word_2}'.
        If it sounds naturally {rich_rich}
        Only final sentence.
        
        """
    )
    #if rich_rich:
        #print(rich_rich)
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text


def generate_sentence_rule_C(word_1, word_2, text_history, model_gemini, response_func, rich_rich):
    prompt = (
        f"""
        Context:\n{text_history}\n\n
        Write one new short sentence where the subject is '{word_1}'
        and do not include '{word_2}'.
        If it sounds naturally {rich_rich}
        Only final sentence.
        
        """
    )
    #if rich_rich:
        #print(rich_rich)
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text


def generate_sentence_rule_D(word_1, word_2, text_history, model_gemini, response_func, rich_rich):
    prompt = (
        f"""
        Context:\n{text_history}\n\n
        Write one new short sentence where neither the subject
        nor the object is '{word_1}' or '{word_2}'.
        If it sounds naturally {rich_rich}
        Only final sentence.
        
        """
    )
    #if rich_rich:
        #print(rich_rich)
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text


def generate_sentence_rule_E(word_1, word_2, text_history, model_gemini, response_func, rich_rich):
    word_2 = pronoun_to_object(word_2)
    word_1 = pronoun_to_object(word_1)
    prompt = (
        f"""
        Context:\n{text_history}\n\n
        Write one new short sentence where the subject is '{word_2}'
        and the object is '{word_1}'.
        If it sounds naturally {rich_rich}
        Only final sentence.
        
        """
    )
    #if rich_rich:
        #print(rich_rich)
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text


def generate_sentence_rule_F(word_1="", word_2="", text_history="", model_gemini=None, response_func=None, rich_rich=""):
    if word_1 == "minor":
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short exclamatory phrase."
            f"Never repeat existing in {text_history} phrase!"
            f"DO NOT SHOW your toughts or notes!!!"
            f"Only final sentence!!!"
        )
    else:
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short imperative phrase."
            f"Never repeat existing in {text_history} phrase!"
            f"DO NOT SHOW your toughts or notes!!!"
            f"Only final sentence!!!"
        )
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text

def generate_sentence_rule_G(word_1="", word_2="", text_history="", model_gemini=None, response_func=None, rich_rich=""):
    
    if word_1 == "minor":
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short interrogative sentence that begins with an auxiliary or modal verb (is, are, do, can, will, etc.)"
            f"Never repeat existing in {text_history} phrase!"
            f"Only final sentence!!!"
        )
    else:
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short Wh-question phrase."
            f"Never repeat existing in {text_history} phrase!"
            f"Only final sentence!!!"
        )
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text
    
def generate_sentence_rule_H(word_1="", word_2="", text_history="", model_gemini=None, response_func=None, rich_rich=""):
    
    if word_1 == "minor":
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short phrase related with previous line using synonyms or tense shift. Do not invent new entities or events."
            f"Never repeat existing in {text_history} phrase!"
            f"If it sounds naturally {rich_rich}"
            f"Only final sentence!!!"
        )
    else:
        prompt = (
            f"Context:\n{text_history}\n\n"
            f"Write one new short phrase related with previous line using synonyms or tense shift. Do not invent new entities or events."
            f"Never repeat existing in {text_history} phrase!"
            f"If it sounds naturally {rich_rich}"
            f"Only final sentence!!!"
        )
    #if rich_rich:
        #print(rich_rich)    
    if response_func:
        return response_func(prompt)
    else:
        response = model_gemini.generate_content(prompt)
        return response.text    


# =========================
# ==== INPUT to RULES =====
# =========================





    
    
    
def extract_root_letters_from_transposed(transposed_chords):
   
    chord_list_roots = []
    for ch in transposed_chords:
        try:
            # Уверяваме се, че имаме стринг
            if not isinstance(ch, str):
                chord_list_roots.append('?')
                continue

            # Търсим първата буква A-G или P (case-insensitive)
            #print("DEBUG:", repr(ch))
            m = re.search(r'([A-Ga-gPp])', ch)
            m = re.search(r'([A-Ga-gPp])', ch.strip())

            if m:
                chord_list_roots.append(m.group(1).upper())
            else:
                chord_list_roots.append('?')
        except Exception:
            chord_list_roots.append('?')
    return chord_list_roots    

def normalize_chord_name(ch):
    
    if ch is None:
        return ch
    ch = ch.strip()
    if len(ch) >= 2 and ch[1] in ('b', '♭'):
        return ch[0] + '-' + ch[2:]
    return ch

def chords_to_list(input_string):
   
    if not input_string:
        return []
    if ' - ' in input_string:
        print("Предупреждение: Тире ('-') означава бемол в music21. Моля използвайте запетая за разделяне на акорди.")
    parts = [p.strip() for p in input_string.split(',') if p.strip()]
    return [normalize_chord_name(p) for p in parts]

def classify_chords(chords):
   
    list_major_minor = []
    list_seventh = []
    list_accidentals = []

    for ch_str in chords:
        try:
            cs = harmony.ChordSymbol(ch_str)
        except Exception:
            list_major_minor.append(-1)
            list_seventh.append(-1)
            list_accidentals.append(-1)
            continue

        # Quality
        quality = getattr(cs, 'quality', None)
        try:
            if quality and 'minor' in quality:
                list_major_minor.append(2)
            elif quality and ('major' in quality or 'dominant' in quality):
                list_major_minor.append(1)
            else:
                fig = (cs.figure or '').lower()
                if 'm' in fig and 'maj' not in fig:
                    list_major_minor.append(2)
                elif 'maj' in fig or 'M' in fig:
                    list_major_minor.append(1)
                else:
                    list_major_minor.append(0)
        except Exception:
            list_major_minor.append(0)

        # Seventh
        try:
            contains7 = False
            if hasattr(cs, 'containsSeventh'):
                contains7 = cs.containsSeventh()
            else:
                fig = cs.figure or ''
                contains7 = '7' in fig
            list_seventh.append(7 if contains7 else 0)
        except Exception:
            list_seventh.append(0)

        # Accidentals on root
        try:
            root = cs.root()
            root_name = root.name if root is not None else ''
            if '#' in root_name or '♯' in root_name:
                list_accidentals.append(11)
            elif '-' in root_name or 'b' in root_name or '♭' in root_name:
                list_accidentals.append(22)
            else:
                list_accidentals.append(0)
        except Exception:
            list_accidentals.append(0)

    return list_major_minor, list_seventh, list_accidentals

def simple_key_estimate(chord_list):
    
    s = stream.Stream()
    for ch in chord_list:
        s.append(harmony.ChordSymbol(ch))
    analysis_results = {
        'default': s.analyze('key'),
        'Krumhansl': s.analyze('Krumhansl'),
        'AardenEssen': s.analyze('AardenEssen'),
        'SimpleWeights': s.analyze('SimpleWeights')
    }

    best_key = None
    max_certainty = -1
    best_method = ""
    for method_name, key_object in analysis_results.items():
        certainty = key_object.tonalCertainty()

        if certainty > max_certainty:
            max_certainty = certainty
            best_key = key_object
            best_method = method_name
    return best_key.tonic.name, best_key.mode, best_key, s

def parse_user_key(user_key_str):
    
    s = user_key_str.strip()
    is_minor = s.lower().endswith('m')
    mode_str = 'minor' if is_minor else 'major'
    tonic_part = s[:-1] if is_minor else s
    tonic_part = tonic_part.replace('b', '-')  # нормализация на бемол
    return tonic_part, mode_str




def transpose_chords_to_reference(chords, user_key_str):
    tonic_str, mode_str = parse_user_key(user_key_str)

    user_key = key.Key(f"{tonic_str}m") if mode_str == 'minor' else key.Key(tonic_str)
    target_key = key.Key('a') if mode_str == 'minor' else key.Key('C')
    transposition_interval = interval.Interval(user_key.tonic, target_key.tonic)

    result = []
    for ch in chords:
        try:
            cs = harmony.ChordSymbol(ch)
            cs_t = cs.transpose(transposition_interval)
            result.append(cs_t.figure)  # запазва минор/мажор, добавки
        except Exception:
            result.append(ch)

    return result, tonic_str, mode_str



def analyze_and_print(chords, key_tonality):
    
    if not chords:
        print("NO CHORDS")
        return [], [], [], [], None, None
    
    
    transposed_chords = []
    if key_tonality:
        transposed_chords, tonic_name, mode = transpose_chords_to_reference(chords, key_tonality)
    else:    
        tonic_name, mode, k, s = simple_key_estimate(chords)
    
    

        if mode == 'major':
            # интервал към C major
            interval_to_c = k.intervalToC()
            s_transposed = s.transpose(interval_to_c)
            #print("Транспонирани акорди към C major:")
        elif mode == 'minor':
             # интервал към A minor
            interval_to_a = interval.Interval(note.Note(k.tonic), note.Note('A'))
            s_transposed = s.transpose(interval_to_a)
            #print("Транспонирани акорди към A minor:")
        else:
            # Ако mode не е нито 'major', нито 'minor', не транспонираме
            s_transposed = s

    
        for cs in s_transposed.getElementsByClass(harmony.ChordSymbol):
            try:
                transposed_chords.append(cs.figure or str(cs))
            except Exception:
                transposed_chords.append(str(cs))

    # Ако няма ChordSymbol елементи (напр. при грешка), опитваме да използваме оригиналните входни имена
    if not transposed_chords:
        transposed_chords = chords.copy()
        
    #print(f"Estimated key: {tonic_name} {mode}")
    
    
    
    
    transposed_chords = extract_root_letters_from_transposed(transposed_chords)
    
    chord_list_roots = []
    for ch in transposed_chords:
        # Ако е пауза - запазваме 'P' директно (ще се картографира към 'H' във MAP_*)
        if ch == 'P':
            chord_list_roots.append('P')
            continue
        try:
            cs = harmony.ChordSymbol(ch)
            root = cs.root()
            root_name = root.name if root is not None else ''
            root_name = root_name.replace('-', 'b')
            chord_list_roots.append(root_name or '?')
        except Exception:
            chord_list_roots.append('?')
            
   
    list_major_minor, list_seventh, list_accidentals = classify_chords(chords)

    
    #print("Transposed chords:", transposed_chords)
    #print("Chord roots:", chord_list_roots)
    #print("Major/Minor classification:", list_major_minor)
    #print("Seventh chords:", list_seventh)
    #print("Accidentals:", list_accidentals)
    
    return chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic_name, mode

# =========================
# === STREAMLIT-READY INPUT
# =========================
def user_prompt():
    
    if st:
        # initialize session state keys
        if 'init_confirm' not in st.session_state:
            st.session_state['init_confirm'] = False
        if 'chords_confirm' not in st.session_state:
            st.session_state['chords_confirm'] = False
        if 'key_confirm' not in st.session_state:
            st.session_state['key_confirm'] = False

        if 'init_phrase' not in st.session_state:
            st.session_state['init_phrase'] = ""
        if 'chords_raw' not in st.session_state:
            st.session_state['chords_raw'] = ""
        if 'key_tonality' not in st.session_state:
            st.session_state['key_tonality'] = ""

        st.markdown("## Harmony to text Generator")
        # Initial phrase input + confirm button
        init_col, init_btn_col = st.columns([4,1])
        with init_col:
            init_phrase_val = st.text_input("Initial Phrase :", value=st.session_state['init_phrase'])
        with init_btn_col:
            if st.button("Confirm Initial Phrase"):
                st.session_state['init_phrase'] = init_phrase_val
                st.session_state['init_confirm'] = True

        if st.session_state['init_confirm']:
            st.success("Initial Phrase confirmed.")
        else:
            st.info("Initial Phrase not confirmed.")

        # Chords input + confirm button
        chords_col, chords_btn_col = st.columns([4,1])
        with chords_col:
            chords_val = st.text_input("Enter chords separated by commas, \n        `P` for `pause` :", value=st.session_state['chords_raw'])
        with chords_btn_col:
            if st.button("Confirm chords"):
                st.session_state['chords_raw'] = chords_val
                st.session_state['chords_confirm'] = True

        if st.session_state['chords_confirm']:
            st.success("Chords confirmed.")
        else:
            st.info("Chords not confirmed.")

        # Key input + confirm button
        key_col, key_btn_col = st.columns([4,1])
        with key_col:
            key_val = st.text_input("Enter main key ( e.g. C, e.g Dm ): ", value=st.session_state['key_tonality'])
        with key_btn_col:
            if st.button("Confirm Key"):
                st.session_state['key_tonality'] = key_val
                st.session_state['key_confirm'] = True

        if st.session_state['key_confirm']:
            st.success("key confirmed.")
        else:
            st.info("key not confirmed.")

        # If not all confirmed, stop execution here in Streamlit
        if not (st.session_state['init_confirm'] and st.session_state['chords_confirm'] and st.session_state['key_confirm']):
            st.warning("Please confirm all fields using the buttons to the right of each entry to continue.")
            st.stop()

        # All confirmed: proceed with the same processing as original CLI
        init_phrase = st.session_state['init_phrase']
        user_input = st.session_state['chords_raw']
        key_tonality = st.session_state['key_tonality']

        chords = chords_to_list(user_input)
        chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic, mode = analyze_and_print(chords, key_tonality)
        return init_phrase, chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic, mode

    # Fallback: original CLI behavior
    init_phrase = input("Initial Phrase :  ")
    user_input = input("Enter chords separated by commas: ")
    key_tonality = input("Enter the key ( e.g. C, e.g Dm ): ")
    
    
    chords = chords_to_list(user_input)
    chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic, mode = analyze_and_print(chords, key_tonality)
    return init_phrase, chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic, mode

# =========================
# =========================
# ===  MAIN ======
# =========================

def main():
    MAP_C_MAJOR = {'C': 'A', 'D': 'E', 'E': 'F', 'F': 'B', 'G': 'C', 'A': 'D', 'B': 'G',"P": "H"}
    MAP_A_MINOR = {'A': 'A', 'B': 'E', 'C': 'F', 'D': 'B', 'E': 'C', 'F': 'D', 'G': 'G',"P": "H"}
    init_phrase, chord_list_roots, list_major_minor, list_seventh, list_accidentals, tonic, mode = user_prompt()
    rules = []
   
   
    for chord in chord_list_roots:
        # We retrieve the coded symbol for each chord
        if mode == "major":
            rules.append(MAP_C_MAJOR[chord])
            #print(mode)
        if mode == "minor":
            rules.append(MAP_A_MINOR[chord])
            #print(mode)
   
    text_history = [(init_phrase, "INIT")]
    # --- SETTINGS (без repetition_penalty) ---
    settings = {
        "temperature": 1.7,
        "top_p": 0.9,
        "max_tokens": 50
    }

    # === HELICONE ИНТЕГРАЦИЯ – ТОВА СА ЕДИНСТВЕНИТЕ НОВИ РЕДОВЕ ===
    from openai import OpenAI

    # ← ТУК СЛАГАШ СВОЯ HELICONE КЛЮЧ (от helicone.ai → API Keys)
    HELICONE_KEY = "sk-helicone-abcdef123456789"   # ← СМЕНИ С ТВОЯ!

    client_helicone = OpenAI(
        api_key=HELICONE_KEY,
        base_url="https://ai-gateway.helicone.ai/v1"
    )

    def call_gemini(prompt):
        response = client_helicone.chat.completions.create(
            model="gemini-2.5-flash-lite",   # ← ТОЧНО ТВОЯ МОДЕЛ, НЕПОКЪТНАТ
            messages=[{"role": "user", "content": prompt}],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_output_tokens=settings["max_tokens"]
        )
        return response.choices[0].message.content.strip()
    # ==========================================================

    # --- Gemini модел с правилна конфигурация – ИЗТРИТО ЦЯЛОТО, ЗАЩОТО ВЕЧЕ НЕ ТРЯБВА ---
    GOOGLE_API_KEY = None
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif "google" in st.secrets and isinstance(st.secrets["google"], dict) and "api_key" in st.secrets["google"]:
        GOOGLE_API_KEY = st.secrets["google"]["api_key"]
    if not GOOGLE_API_KEY:
        st.error("Google API key not found. Please set it in Streamlit secrets.")
        return

    # === КРАЙ НА HELICONE ЧАСТТА ===

    GROQ_KEY = None
    if "GROQ_API_KEY" in st.secrets:
        GROQ_KEY = st.secrets["GROQ_API_KEY"]
    elif "groq" in st.secrets and isinstance(st.secrets["groq"], dict) and "api_key" in st.secrets["groq"]:
        GROQ_KEY = st.secrets["groq"]["api_key"]
    if not GROQ_KEY:
        st.error("No GROQ API key found. Please set it in Streamlit secrets.")
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    GROQ_MODEL = "llama-3.1-8b-instant"
    rule_map = {
        "A": generate_sentence_rule_A,
        "B": generate_sentence_rule_B,
        "C": generate_sentence_rule_C,
        "D": generate_sentence_rule_D,
        "E": generate_sentence_rule_E,
        "F": generate_sentence_rule_F,
        "G": generate_sentence_rule_G,
        "H": generate_sentence_rule_H,
       
    }
   
    if mode == 'major':
        adv_dict = ADVERB_GUIDELINES_FOR_MAJOR
        conj_dict = CONJUNCTION_GUIDELINES_FOR_MAJOR
    else:
        adv_dict = ADVERB_GUIDELINES_FOR_MINOR
        conj_dict = CONJUNCTION_GUIDELINES_FOR_MINOR
   
    request_count = 0
    a =0
    for chord in rules:
        full_text = " ".join([t[0] for t in text_history])
        doc = nlp(full_text)
        last_subj, last_obj = extract_last_subj_obj_with_clauses(doc)
        rule_func = rule_map.get(chord)
       
         
        if rule_func:
            request_count += 1
           
           
            root = chord_list_roots[a]
            rich_rich = None
            if list_accidentals[a] == 22:
                key = root + "b"
                rich_rich= adv_dict.get(key)
           
            if list_accidentals[a] == 11:
                key = root + "#"
                rich_rich = conj_dict.get(key)
           
            if list_seventh[a] == 7:
                if list_major_minor[a] == 2:
                    rich_rich = adv_dict.get("m7")
                else:
                    rich_rich = conj_dict.get("maj 7")
               
            if chord_list_roots[a] == "H":
                if list_major_minor[a-1] == 2:
                    rich_rich = adv_dict.get("pause min")
                else:
                    rich_rich = conj_dict.get("pause maj")
               
            if request_count <= 14:
                if chord == "F" or chord == "G" or chord == "H":
                    result = rule_func(mode, last_obj, full_text, call_gemini, None, rich_rich)
                else:
                    result = rule_func(last_subj, last_obj, full_text, call_gemini, None, rich_rich)
            else:
                def llama_generate(prompt):
                    response = client.chat.completions.create(
                    model=GROQ_MODEL,
                        messages=[
                            {
                                "role": "system",
                               "content": "You are a creative poet and your style is concise. You strictly follow the rules described in the prompt."
                           },
                           {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        max_tokens=settings["max_tokens"]
                    )
                    return response.choices[0].message.content.strip()
                if chord == "F" or chord == "G" or chord == "H":
                    result = rule_func(mode, last_obj, full_text, None, llama_generate, rich_rich)
                else:
                    result = rule_func(last_subj, last_obj, full_text, None, llama_generate, rich_rich)
            a +=1
        clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", result)
        text_history.append((clean_text, chord))
   
    a = 0
    st.write("--- Final text history ---")
    buf = []
    for a, (sentence, chord) in enumerate(text_history, start=1):
        buf.append(f"{a}. {sentence.strip()} [{chord}]")
    st.text("\n".join(buf))



if __name__ == "__main__":
    main()
