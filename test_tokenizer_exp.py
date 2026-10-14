if __name__ == "__main__":
    from cs336_basics.tokenizer import Tokenizer
    import time
    import numpy as np

    def problem(
        doc_path: str,
        doc_num: int,
        doc_name: str,
        tokenizer: tuple[str, Tokenizer]
    ) -> tuple[str, (int, int)]:
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
        docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()][:doc_num]
        bytes = 0
        token = 0
        for doc in docs:
            bytes_len = len(doc.encode("utf-8"))
            token_len = len(tokenizer[1].encode(doc))
            bytes += bytes_len
            token += token_len
        return ("{} {} docs: {} bytes ==={}==> {} tokens, compression ratio: {}".format(doc_name, doc_num, bytes, tokenizer[0], token, bytes / token), (bytes, token))

    def file_to_ids(
        file_path: str,
        tokenizer: Tokenizer,
        target_path: str
    ) -> str:
        time_point = time.perf_counter();
        with open(file_path, encoding="utf-8", newline="") as read:
            data = np.fromiter(tokenizer.encode_iterable(read), dtype=np.uint16)
            np.save(target_path, data)
        return "{} finished, spend {} sec".format(file_path, time.perf_counter() - time_point)

    tok_ts  = Tokenizer.from_files("my_result/TinyStoriesV2-GPT4-train_vocab.json", "my_result/TinyStoriesV2-GPT4-train_merges.txt", ["<|endoftext|>"])
    tok_owt = Tokenizer.from_files("my_result/owt_vocab.json", "my_result/owt_merges.txt", ["<|endoftext|>"])

    ts_valid = "data/TinyStoriesV2-GPT4-valid.txt"
    ts_train = "data/TinyStoriesV2-GPT4-train.txt"
    owt_valid = "data/owt_valid.txt"
    owt_train = "data/owt_train.txt"

    ts_valid_np = "my_result/ts_valid_np.npy"
    ts_train_np = "my_result/ts_train_np.npy"
    owt_valid_np = "my_result/owt_valid_np.npy"
    owt_train_np = "my_result/owt_train_np.npy"

    print("---------- problem a ----------")
    print(problem(ts_valid, 10, "ts", ("ts", tok_ts))[0])
    print(problem(owt_valid, 10, "owt", ("owt", tok_owt))[0])

    print("---------- problem b ----------")
    print(problem(owt_valid, 10, "owt", ("ts", tok_ts))[0])

    print("---------- problem c ----------")
    time_point = time.perf_counter();
    bytes = problem(ts_valid, -1, "ts", ("ts", tok_ts))[1][0]
    time_used = time.perf_counter() - time_point
    throughput = bytes / time_used
    print("get {} bytes, ues {} sec ( use encode func ) => {} bytes/second).".format(bytes, time_used, throughput))
    print(" 825 GB text will use {} sec".format( 825 * 1000 * 1000 * 1000 / throughput))

    print("---------- problem d ----------")
    assert max(tok_ts._vocab) < 65536
    assert max(tok_owt._vocab) < 65536

    print(file_to_ids(ts_valid, tok_ts, ts_valid_np))
    print(file_to_ids(ts_train, tok_ts, ts_train_np))
    print(file_to_ids(owt_valid, tok_owt, owt_valid_np))
    print(file_to_ids(owt_train, tok_owt, owt_train_np))