from tests.adapters import run_train_bpe        # ← import，不是复制

if __name__ == "__main__":                       # ← 入口脚本必须有守卫
    vocab, merges = run_train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    print(len(merges), len(vocab))
