import json

input_path = "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.json"
output_path = "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"

with open(input_path, "r") as f:
    raw_data = json.load(f)["data"]

with open(output_path, "w") as fout:
    for entry in raw_data:
        line = {
            "question_id": entry["question_id"],
            "image": entry["image_id"] + ".jpg",
            "text": entry["question"]
        }
        fout.write(json.dumps(line) + "\n")

print(f"Saved {len(raw_data)} items to {output_path}")
