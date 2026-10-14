if __name__ == "__main__":
    from tests.adapters import run_train_bpe        # ← import，不是复制
    from tests.common import gpt2_bytes_to_unicode
import json

if __name__ == "__main__":                       # ← 入口脚本必须有守卫
    vocab, merges = run_train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    gpt2_dict = gpt2_bytes_to_unicode();
    with open("my_result/TinyStoriesV2-GPT4-train_vocab.json", "w", encoding="utf-8") as f:
        json.dump({ "".join([ gpt2_dict[c] for c in bytes ]): id for id, bytes in vocab.items() }, f, ensure_ascii=False);
    with open("my_result/TinyStoriesV2-GPT4-train_merges.txt", "w", encoding="utf-8") as f:
        f.write( "\n".join([" ".join(["".join([gpt2_dict[c] for c in item]) for item in merge]) for merge in merges]));
    print(len(merges), len(vocab))
