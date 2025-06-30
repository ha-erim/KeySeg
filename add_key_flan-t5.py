import json
import os
import re
from pathlib import Path
from transformers import pipeline

BASE_DIR = Path("/home/hrkim/dataset/reason_seg/ReasonSeg")
TRAIN_FILE = BASE_DIR / "explanatory" / "train.json"
ANNOTATION_DIR = BASE_DIR / "train"
OUTPUT_DIR = BASE_DIR / "train_updated"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "keyword_outputs.json" #save path

print("FLAN-T5-XL loading...")
generator = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
print("complete!")

def build_prompt(query, outputs):
    return (
        f"From the following text, extract the most important object mentioned. "
        f"It should be one word or a short noun phrase.\n\n"
        f"{query}\n{outputs}\n\n"
        f"Return only the object's name."
    )

def extract_phrases(text):
    words = re.findall(r"\b\w+\b", text.lower())
    phrases = set()
    for i in range(len(words)):
        phrases.add(words[i])
        if i + 1 < len(words):
            phrases.add(f"{words[i]} {words[i + 1]}")
    return phrases

def extract_keyword(prompt, valid_phrases):
    response = generator(prompt, max_new_tokens=10, do_sample=False)
    text = response[0]["generated_text"].strip().lower()
    clean_text = re.sub(r"[^\w\s]", "", text)
    if clean_text in valid_phrases:
        return clean_text
    else:
        return f"(not found) {clean_text}"

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

results = []

for i, entry in enumerate(train_data):
    query = entry["query"]
    outputs = entry["outputs"]
    json_filename = entry["json"]
    image_path = entry["image"]
    json_path = ANNOTATION_DIR / json_filename

    if not json_path.exists():
        print(f"missing json file: {json_filename}")
        continue

    prompt = build_prompt(query, outputs)
    valid_phrases = extract_phrases(query + " " + outputs)
    keyword = extract_keyword(prompt, valid_phrases)

    result = {
        "query": query,
        "outputs": outputs,
        "image":image_path,
        "json": json_filename,
        "keyword": keyword,
    }
    results.append(result)

    # 콘솔 출력
    print(f"\n[count] {i+1}")
    print(f"Query: {query}")
    print(f"Output: {outputs}")
    print(f"Keyword: {keyword}")

# === 파일로 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n complete keyword extraction & saved in : {OUTPUT_JSON}")
