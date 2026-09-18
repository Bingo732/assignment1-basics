import os
from typing import BinaryIO

from collections import Counter
import regex

# 官方给的文件切块函数
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 并行化处理部分
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
WORDING_PAT = regex.compile(PAT);
## 初始化文件分块
def get_chunk_boundiers_initial(
    input_path: str | os.PathLike,  # 输入语料库的文件路径，可以是字符串或os.PathLike对象
    chunk_num: int, # 期望初始化分块数量
    special_tokens: list[str],  # 需要添加到词汇表中的特殊标记列表，这些标记不会被拆分
) -> list[int]:
    with open(input_path, "rb") as file:
        file.seek(0);
        if not special_tokens:
            file.seek(0, os.SEEK_END);
            boundaries = [0, file.tell()];
        else:
            num_chunks = chunk_num;
            boundaries = [];
            for special_token in special_tokens:
                boundaries += find_chunk_boundaries(file, num_chunks, special_token.encode("utf-8"));
    boundaries = sorted(set(boundaries));
    return boundaries;
## 将初始化的分块进一步分到特殊点位 - worker 函数
def get_vocab_from_chunk(
    input_path: str | os.PathLike,  # 输入语料库的文件路径，可以是字符串或os.PathLike对象
    chunking_pattern: str,  # 处理后的特殊token对应的切分规则
    span: tuple[int, int], # 开始点和结束点
) -> Counter[tuple[int, ...]]: # 返回区间内语料对应的词表
    start, end = span;
    words_freq = Counter();
    with open(input_path, "rb") as file:
        file.seek(start);
        chunks = file.read(end - start).decode("utf-8");
        if chunking_pattern: chunks = regex.split(chunking_pattern, chunks);
        else: chunks = [chunks];
        for chunk in chunks:
            words = WORDING_PAT.findall(chunk);
            for word in words:
                words_freq[tuple(word.encode("utf-8"))] += 1;
    return words_freq;