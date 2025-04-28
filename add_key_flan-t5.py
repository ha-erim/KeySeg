import json
import os
import re
from pathlib import Path
from transformers import pipeline

# === 경로 설정 ===
BASE_DIR = Path("/home/hrkim/dataset/reason_seg/ReasonSeg")
TRAIN_FILE = BASE_DIR / "explanatory" / "train.json"
ANNOTATION_DIR = BASE_DIR / "train"
OUTPUT_DIR = BASE_DIR / "train_updated"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "keyword_outputs.json"  # 저장 경로

# === 모델 로딩 ===
print("🔧 FLAN-T5-XL 모델 로딩 중...")
generator = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
print("✅ 모델 로딩 완료")

# === 프롬프트 구성 ===
def build_prompt(query, outputs):
    return (
        f"From the following text, extract the most important object mentioned. "
        f"It should be one word or a short noun phrase.\n\n"
        f"{query}\n{outputs}\n\n"
        f"Return only the object's name."
    )

# === 텍스트에서 단어 및 명사구 추출
def extract_phrases(text):
    words = re.findall(r"\b\w+\b", text.lower())
    phrases = set()
    for i in range(len(words)):
        phrases.add(words[i])
        if i + 1 < len(words):
            phrases.add(f"{words[i]} {words[i + 1]}")
    return phrases

# === 키워드 추출 및 검증
def extract_keyword(prompt, valid_phrases):
    response = generator(prompt, max_new_tokens=10, do_sample=False)
    text = response[0]["generated_text"].strip().lower()
    clean_text = re.sub(r"[^\w\s]", "", text)
    if clean_text in valid_phrases:
        return clean_text
    else:
        return f"(not found) {clean_text}"

# === train.json 로드
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# === 결과 저장을 위한 리스트
results = []

# === 추출 및 출력 + 저장
for i, entry in enumerate(train_data):
    query = entry["query"]
    outputs = entry["outputs"]
    json_filename = entry["json"]
    image_path = entry["image"]
    json_path = ANNOTATION_DIR / json_filename

    if not json_path.exists():
        print(f"⚠️ 누락된 JSON 파일: {json_filename}")
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
    print(f"🗨️ Query: {query}")
    print(f"📝 Output: {outputs}")
    print(f"🔑 Keyword: {keyword}")

# === 파일로 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 키워드 추출 완료 및 저장됨: {OUTPUT_JSON}")
