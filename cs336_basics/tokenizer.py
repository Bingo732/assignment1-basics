from collections.abc import Iterable, Iterator
from cs336_basics.my_utils import WORDING_PAT
from cs336_basics.my_utils import gpt2_bytes_to_unicode
import regex
import json

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._vocab = dict(vocab)
        self._bytes_to_id = { value: key for key, value in vocab.items() }
        self._pair_rank = { pair: rank for rank, pair in enumerate(merges) }
        self._special_tokens = []
        if special_tokens:
            self._special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in self._special_tokens:
                if special_token.encode("utf-8") not in self._bytes_to_id:
                    self._vocab[len(self._vocab)] = special_token.encode("utf-8")
                    self._bytes_to_id[special_token.encode("utf-8")] = len(self._bytes_to_id)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        gpt2_dict_reverse = { value: key for key, value in gpt2_bytes_to_unicode().items() }

        vocab = {}
        with open(vocab_filepath, encoding="utf-8") as f:
            data = json.load(f)
            for key, value in data.items():
                vocab[value] = bytes([ gpt2_dict_reverse[byte] for byte in key ])

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    a, b = line.split(" ")
                    bytes_a = bytes([ gpt2_dict_reverse[char] for char in a ])
                    bytes_b = bytes([ gpt2_dict_reverse[char] for char in b ])
                    merges.append((bytes_a, bytes_b))
        return cls(vocab, merges, special_tokens)

    def _text_to_pieces(self, text: str) -> list[str]:
        if self._special_tokens:
            chunking_pattern = "|".join( regex.escape(token) for token in self._special_tokens )
            parts = regex.split(f"({chunking_pattern})", text)
        else:
            parts = [text]
        pieces = []
        for part in parts:
            if part in self._special_tokens:
                pieces.append(part)
            else:
                pieces.extend(WORDING_PAT.findall(part))
        return pieces

    def _word_to_ids(self, word: str) -> list[int]:
        bytes_list = [ bytes([byte]) for byte in word.encode("utf-8") ]
        result = []
        while True:
            choose_pos = min(
                ( i for i in range(len(bytes_list) - 1) if (bytes_list[i], bytes_list[i + 1]) in self._pair_rank),
                key = lambda i: self._pair_rank[(bytes_list[i], bytes_list[i + 1])],
                default = -1
            )
            if choose_pos == -1: break
            bytes_list[choose_pos : choose_pos + 2] = [ bytes_list[choose_pos] + bytes_list[choose_pos + 1] ]
        for item in bytes_list:
            result.append(self._bytes_to_id[item])
        return result
        
    def encode(self, text: str) -> list[int]:
        pieces = self._text_to_pieces(text)

        result = []

        for piece in pieces:
            if piece in self._special_tokens:
                result.append(self._bytes_to_id[piece.encode("utf-8")])
            else:
                result.extend(self._word_to_ids(piece))

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            pieces = self._text_to_pieces(pending + chunk)
            pending = ""
            if pieces:
                pending = pieces.pop()

            for piece in pieces:
                if piece in self._special_tokens:
                    yield self._bytes_to_id[piece.encode("utf-8")]
                else:
                    yield from self._word_to_ids(piece)

        if pending:
            if pending in self._special_tokens:
                yield self._bytes_to_id[pending.encode("utf-8")]
            else:
                for i in self._word_to_ids(pending):
                    yield i

    def decode(self, ids: list[int]) -> str:
        return b"".join([ self._vocab[id] for id in ids ]).decode("utf-8", errors="replace")

