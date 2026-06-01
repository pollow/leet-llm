"""Static typing facade for dynamically loaded task exports.

Runtime symbols are still resolved by ``leet_llm.__getattr__`` from the
registry. This stub exists for IDE and type checker support.
"""

from typing import Any

# L1 typed exports (current tokenizer tasks)
def build_char_vocab(text: str) -> tuple[dict[str, int], list[str]]: ...
def char_encode(text: str, stoi: dict[str, int]) -> list[int]: ...
def char_decode(ids: list[int], itos: list[str]) -> str: ...
def text_to_byte_ids(text: str) -> list[int]: ...
def byte_ids_to_text(ids: list[int]) -> str: ...
def count_pairs(seq: list[int]) -> dict[tuple[int, int], int]: ...
def apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]: ...
def bpe_train(text: str, vocab_size: int) -> tuple[list[str], list[float]]: ...

# Remaining exports are dynamic; keep them importable for static tools.
group_last_axis: Any
ungroup_last_axis: Any
add_bias: Any
standardize: Any
affine: Any
batched_matmul: Any
outer_product: Any
batched_trace: Any
softmax: Any
logsumexp: Any
log_softmax: Any
top_k: Any
argmax: Any
gather_rows: Any
one_hot: Any
masked_fill: Any
triangular_mask: Any
sample_categorical: Any
interleave: Any
deinterleave: Any
split_halves: Any
join_halves: Any
save_tokenizer: Any
load_tokenizer: Any
bpe_encode: Any
bpe_decode: Any
add_special_tokens: Any
strip_special_tokens: Any
regex_split: Any
tiktoken_encode: Any
tiktoken_decode: Any
pad_batch: Any
padding_mask: Any
position_ids: Any
build_batch: Any
embedding: Any
gelu: Any
silu: Any
layer_norm: Any
sinusoidal_pe: Any
sdpa: Any
mha: Any
AttnParams: Any
ffn: Any
FFNParams: Any
add_residual: Any
encoder_block: Any
EncoderBlockParams: Any
decoder_block: Any
DecoderBlockParams: Any
gpt_block: Any
GPTBlockParams: Any
rms_norm: Any
rope_interleaved: Any
rope_half: Any
rope_qk_dot: Any
swiglu_ffn: Any
SwiGLUParams: Any
gqa: Any
llama_decoder_block: Any
LlamaBlockParams: Any

def __getattr__(name: str) -> Any: ...
def __dir__() -> list[str]: ...
