from typing import List, Tuple, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

CLASS_START_TOKEN = "<classify>"
CLASS_END_TOKEN = "</classify>"
GLOSS_START_TOKEN = "<g>"
GLOSS_END_TOKEN = "</g>"


def get_tokenizer(transformer_model: str, use_special_tokens: bool) -> "DefinitionsTokenizer":
    return AutoDefinitionsTokenizer(transformer_model, use_special_tokens)


class DefinitionsTokenizer:

    transformer_model: str
    tokenizer: PreTrainedTokenizerFast
    use_special_tokens: bool
    space_token_id: int
    gloss_start_token_id: int
    gloss_end_token_id: int

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @staticmethod
    def insert_classify_tokens(sentence_tokens: List[str], index: int) -> str:
        sent_tokens_copy = sentence_tokens.copy()
        sent_tokens_copy.insert(index, CLASS_START_TOKEN)
        sent_tokens_copy.insert(index + 2, CLASS_END_TOKEN)
        return " ".join(sent_tokens_copy)

    def encode_pair(
        self, sequence1: str, sequence2: str
    ) -> Tuple[torch.LongTensor, Optional[List[Tuple[int, int]]], Optional[torch.LongTensor]]:
        encoding_out = self.tokenizer.encode_plus(sequence1, sequence2, return_tensors="pt", return_token_type_ids=True)

        token_type_ids = getattr(encoding_out, "token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze()

        return encoding_out.input_ids.squeeze(), encoding_out.encodings[0].offsets, token_type_ids

    def prepare_sample(
        self, context_sentence: str, definitions: List[str]
    ) -> Tuple[torch.LongTensor, List[Tuple[int, int]], Optional[torch.LongTensor]]:
        if self.use_special_tokens:
            return self.prepare_sample_with_st(context_sentence, definitions)
        else:
            return self.prepare_sample_without_st(context_sentence, definitions)

    def prepare_sample_without_st(
        self, context_sentence: str, definitions: List[str]
    ) -> Tuple[torch.LongTensor, List[Tuple[int, int]], Optional[torch.LongTensor]]:

        definitions_seq = " ".join(definitions)
        definitions_offsets = []

        for i in range(len(definitions)):
            if i == 0:
                definitions_offsets.append((0, len(definitions[i])))
            else:
                last_definition_start = definitions_offsets[i - 1][1]
                start = last_definition_start + 1
                definitions_offsets.append((start, start + len(definitions[i])))

        encoded_final_sequence, offsets, token_type_ids = self.encode_pair(context_sentence, definitions_seq)

        context_start_position = [i for i, elem in enumerate(offsets[:-1]) if sum(elem) + sum(offsets[i + 1]) == 0][0]
        context_offset = context_start_position + 2
        offsets = offsets[context_offset:-1]

        si2ti = dict()
        ei2ti = dict()
        for i, (si, ei) in enumerate(offsets):
            si2ti[si] = i
            ei2ti[ei] = i

        definitions_positions = [
            (context_offset + si2ti[gsi], context_offset + ei2ti[gei]) for gsi, gei in definitions_offsets
        ]

        return encoded_final_sequence, definitions_positions, token_type_ids

    def prepare_sample_with_st(
        self, context_sentence: str, definitions: List[str]
    ) -> Tuple[torch.LongTensor, List[Tuple[int, int]], Optional[torch.LongTensor]]:

        definitions_seq = "".join([f"{GLOSS_START_TOKEN} {_def}{GLOSS_END_TOKEN}" for _def in definitions])
        encoded_final_sequence, _, token_typed_ids = self.encode_pair(context_sentence, definitions_seq)

        # removing spaces created with the addition of <classify> and </classify>
        encoded_final_sequence = encoded_final_sequence[encoded_final_sequence != self.space_token_id]

        definitions_positions = zip(
            (i for i, wid in enumerate(encoded_final_sequence) if wid == self.gloss_start_token_id),
            (i for i, wid in enumerate(encoded_final_sequence) if wid == self.gloss_end_token_id),
        )

        return encoded_final_sequence, list(definitions_positions), token_typed_ids

    def save(self, dir_path: str) -> None:
        import os

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            self.tokenizer.save_pretrained(dir_path)

    def __len__(self):
        return len(self.tokenizer)

    def decode(self, sequence: Union[int, List[int]]) -> str:
        if type(sequence) == int:
            sequence = [sequence]
        return self.tokenizer.decode(sequence)


class AutoDefinitionsTokenizer(DefinitionsTokenizer):
    def __init__(self, transformer_model: str, use_special_tokens: bool):
        self.transformer_model = transformer_model
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True, add_prefix_space=True)
        self.use_special_tokens = use_special_tokens
        if self.use_special_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<classify>", "</classify>", "<g>", "</g>"]}
            )
            self.space_token_id = -1
            self.gloss_start_token_id = self.tokenizer.encode("<g>", add_special_tokens=False)[0]
            self.gloss_end_token_id = self.tokenizer.encode("</g>", add_special_tokens=False)[0]
