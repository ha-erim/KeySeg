import os
from tqdm import tqdm
from utils.refer import REFER

# used for extract image_path & question from refer_seg datasets

BASE_IMAGE_DIR = "/home/hrkim/dataset/refer_seg"
DATASET = "refcocog"
SPLITBY = "umd"
SPLIT = "test"
OUTPUT_TXT = "refcocog_umd_test_questions.txt"

refer = REFER(BASE_IMAGE_DIR, DATASET, SPLITBY)

refs = refer.loadRefs(ref_ids=refer.getRefIds(split=SPLIT))
imgs = {img["id"]: img for img in refer.loadImgs(image_ids=[ref["image_id"] for ref in refs])}
valid_refs = [ref for ref in refs if ref["image_id"] in imgs]

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for ref in tqdm(valid_refs, desc="Saving questions"):
        image_info = imgs[ref["image_id"]]
        image_path = os.path.join(
            BASE_IMAGE_DIR,
            "images/mscoco/images/train2014",
            image_info["file_name"]
        )
        sents = [s["sent"].strip().lower() for s in ref["sentences"]]
        if sents:
            question = sents[0]
            f.write(f"{image_path}\t{question}\n")

print(f"Saved to {OUTPUT_TXT}")
