"""Step 3: ask MedGemma a yes/no question about each test image.

Saves each raw answer to results/answers.csv so you can read them yourself.

Before running:
    1. Accept the terms on https://huggingface.co/google/medgemma-4b-it
    2. Run `huggingface-cli login` and paste your access token

Example:
    python 02_ask_medgemma.py --finding "a pleural effusion (fluid around the lungs)"
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline


def pick_device_and_dtype():
    """Use the GPU if there is one; bfloat16 where the GPU supports it."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        print("Note: this GPU has no bfloat16 support (e.g. a T4), so using float16. "
              "If answers look garbled, switch to an L4/A100 GPU.")
        return "cuda", torch.float16
    print("Note: no GPU found. This will work but be very slow.")
    return "cpu", torch.float32


def to_yes_no(text):
    """Turn the model's free-text reply into 'yes', 'no' or 'unclear'."""
    first_word = text.strip().lower().lstrip("*\"' ").split()[0] if text.strip() else ""
    first_word = first_word.strip(".,!:;*\"'")
    if first_word in ("yes", "no"):
        return first_word
    return "unclear"


def main():
    parser = argparse.ArgumentParser(description="Ask MedGemma about each test image.")
    parser.add_argument("--test-set", default="data/test_set.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--finding", default="a pleural effusion (fluid around the lungs)",
                        help="what to ask about, in plain words")
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--out", default="results/answers.csv")
    args = parser.parse_args()

    # The exact wording matters. Changing it is "step 2: better instructions".
    question = (f"Is there {args.finding} in this chest X-ray? "
                "Answer with only one word: yes or no.")

    device, dtype = pick_device_and_dtype()
    print(f"Loading {args.model} on {device} ...")
    pipe = pipeline("image-text-to-text", model=args.model, torch_dtype=dtype, device=device)

    with open(args.test_set, newline="") as f:
        test_rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(test_rows, start=1):
        image = Image.open(Path(args.images) / row["image"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
            ]},
        ]
        output = pipe(text=messages, max_new_tokens=20)
        reply = output[0]["generated_text"][-1]["content"]
        answer = to_yes_no(reply)
        results.append([row["image"], row["truth"], answer, reply.strip()])
        print(f"[{i}/{len(test_rows)}] {row['image']}: truth={row['truth']} model={answer}  ({reply.strip()!r})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "truth", "answer", "raw_reply"])
        writer.writerows(results)
    print(f"Saved answers to {out}")


if __name__ == "__main__":
    main()
