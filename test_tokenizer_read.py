if __name__ == "__main__":
    from cs336_basics.tokenizer import Tokenizer
    from tests.adapters import run_train_bpe

if __name__ == "__main__":
    tok_files = Tokenizer.from_files("my_result/TinyStoriesV2-GPT4-train_vocab.json", "my_result/TinyStoriesV2-GPT4-train_merges.txt", ["<|endoftext|>"])
    tok_mem   = Tokenizer( *run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"]), ["<|endoftext|>"] )
    test_strs = [
                    "test test test sentence",
                    "Hello, world! 123",
                    "你好，世界",
                    "abc<|endoftext|>def",
                    " a b "
                ]

    assert tok_files._vocab      == tok_mem._vocab
    assert tok_files._pair_rank  == tok_mem._pair_rank
    assert tok_files._bytes_to_id == tok_mem._bytes_to_id
    for str in test_strs:
        assert tok_files.encode(str) == tok_mem.encode(str), str + "有问题"
