import xml.etree.ElementTree as ET
import json
import emoji
from googletrans import Translator
import os

# ---------- Utility ----------
def normalize(text: str) -> str:
    return text.lower().replace("-", " ").strip()

def parse_cldr_annotations(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mapping = {}
    for annotation in root.findall(".//annotation"):
        emoji_char = annotation.attrib.get('cp')
        keywords = annotation.text
        if emoji_char and keywords:
            for kw in keywords.split('|'):
                mapping[normalize(kw)] = emoji_char
    return mapping

def translate_to_english(text: str, translator: Translator) -> str:
    try:
        translated = translator.translate(text, src='nl', dest='en').text
        return normalize(translated)
    except Exception as e:
        print(f"[WARN] Translation failed for {text}: {e}")
        return None

# ---------- Main Mapping Logic ----------
def build_dutch_to_emoji(input_names, cldr_dict_en, cldr_dict_nl, translator):
    dutch_to_emoji = {}
    for name in input_names:
        norm_name = normalize(name)
        emoji_char = None

        # 1️⃣ Translate -> English -> check English CLDR
        eng_name = translate_to_english(norm_name, translator)
        if eng_name:
            emoji_char = cldr_dict_en.get(eng_name)
            if emoji_char:
                dutch_to_emoji[name] = emoji_char
                continue

            # 2️⃣ If not found, try emoji.emojize on the English name
            emj = emoji.emojize(f":{eng_name}:", language='en')
            if emj != f":{eng_name}:":
                dutch_to_emoji[name] = emj
                continue

        # 3️⃣ If still not found, check Dutch CLDR
        emoji_char = cldr_dict_nl.get(norm_name)
        if emoji_char:
            dutch_to_emoji[name] = emoji_char
            continue

        # 4️⃣ Nothing worked
        dutch_to_emoji[name] = "❓"

    return dutch_to_emoji

# ---------- Entry point ----------
if __name__ == "__main__":
    # Input file with Dutch emoji names (one per line)
    input_file = "input/emoji_names.txt"
    output_file = "output/dutch_to_emoji.json"

    # Load emoji names
    with open(input_file, "r", encoding="utf-8") as f:
        emoji_names = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Loaded {len(emoji_names)} emoji names from {input_file}")

    # Parse CLDR annotation files
    cldr_en_path = os.path.join("input", "en.xml")
    cldr_nl_path = os.path.join("input", "nl.xml")

    print("[INFO] Parsing CLDR English annotations…")
    cldr_dict_en = parse_cldr_annotations(cldr_en_path)
    print(f"[INFO] Loaded {len(cldr_dict_en)} English annotation entries")

    print("[INFO] Parsing CLDR Dutch annotations…")
    cldr_dict_nl = parse_cldr_annotations(cldr_nl_path)
    print(f"[INFO] Loaded {len(cldr_dict_nl)} Dutch annotation entries")

    # Translator for fallback
    translator = Translator()

    # Build mapping
    print("[INFO] Building Dutch→Emoji mapping…")
    dutch_to_emoji = build_dutch_to_emoji(emoji_names, cldr_dict_en, cldr_dict_nl, translator)

    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dutch_to_emoji, f, ensure_ascii=False, indent=2)

    print(f"[✅] Mapping complete. Saved to {output_file}")
