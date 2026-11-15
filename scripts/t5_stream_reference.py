"""Ground truth for flan-t5's token stream (`test_e2e_t5_token_stream.cpp`, ADR-130): transformers' greedy
`generate` on the checkpoint, the ids a `TextIteratorStreamer` would hand out one by one -- a streamer only
watches the loop, so the stream's ids are the one pass's, and those are what is recorded.

Per prompt, two lines in `<out>`:
    src <ids>   -- the tokenizer's ids for the prompt, `</s>` appended (what the encoder reads)
    out <ids>   -- the generated ids after the decoder start token, the final `</s>` included when the
                   loop stopped on it

    python scripts/t5_stream_reference.py <flan-t5 dir> <out.txt> [--max-new-tokens 48]
"""
import argparse

import torch

PROMPTS = [
    "Translate English to German: How old are you?",
    "Answer the following question. What is the capital of France?",
    "Summarize: The city council met on Tuesday to discuss the new budget. After a long debate, members "
    "agreed to increase spending on public transport and parks, while cutting administrative costs.",
    "Write a short story about a robot who learns to paint.",
    "Q: Can a dog drive a car? Give the rationale before answering.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    with open(args.out, "w") as f, torch.no_grad():
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)[0]
            gen = out[1:].tolist()  # after the decoder start token
            f.write("src " + " ".join(map(str, ids[0].tolist())) + "\n")
            f.write("out " + " ".join(map(str, gen)) + "\n")
            print(f"{len(gen):3d} ids: {tok.decode(gen, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
