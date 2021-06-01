import collections
import random
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Any, Optional, NamedTuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributions import poisson

from esc.utils.definitions_tokenizer import DefinitionsTokenizer
from esc.utils.commons import flatten, chunks, batch_data
from esc.utils.wordnet import wn_offsets_from_lemmapos, synset_from_offset, wn_offset_from_sense_key
from esc.utils.wsd import expand_raganato_path, read_from_raganato, pos_map, WSDInstance

ROBERTA_TOKENIZER_MODELS = {"roberta", "bart", "longformer"}


class DataElement(NamedTuple):
    encoded_final_sequence: torch.LongTensor
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    possible_offsets: Optional[List[str]] = None
    gold_labels: Optional[List[str]] = None
    gloss_positions: Optional[List[Tuple[int, int]]] = None
    token_type_ids: Optional[torch.LongTensor] = None
    wsd_instance: Optional[WSDInstance] = None


class QAExtractiveDataset(IterableDataset, ABC):

    dataset_id: str

    def __init__(
        self, tokenizer: DefinitionsTokenizer, tokens_per_batch: int, re_init_on_iter: bool, is_test: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.re_init_on_iter = re_init_on_iter
        self.is_test = is_test
        self.data_store: List[DataElement] = []
        self.is_dataset_used = False

    @abstractmethod
    def init_dataset(self):
        raise NotImplementedError

    def clean_dataset(self):

        old_len = len(self.data_store)

        self.data_store = [
            elem
            for elem in self.data_store
            if elem.encoded_final_sequence.size(0) <= self.tokens_per_batch
            and elem.encoded_final_sequence.size(0) <= self.tokenizer.model_max_length
        ]

        new_len = len(self.data_store)

        if old_len != new_len:
            print(
                "Removed {} instances from the {} data due to their len. Old-len: {}, New-len: {}".format(
                    old_len - new_len, self.__class__.__name__, old_len, new_len
                )
            )

    def shuffle_dataset(self) -> None:
        if not self.is_test:
            self.data_store = sorted(
                self.data_store, key=lambda x: x.encoded_final_sequence.size(0) + torch.randint(0, 10, (1,)) + 1
            )
            self.data_store = list(chunks(self.data_store, 2048))
            random.shuffle(self.data_store)
            self.data_store = flatten(self.data_store)
        else:
            self.data_store = sorted(self.data_store, key=lambda x: x.encoded_final_sequence.size(0))

    def build_dataset(self):
        self.init_dataset()
        self.shuffle_dataset()
        self.clean_dataset()
        self.is_dataset_used = False

    def output_batch(self, current_batch: list) -> Dict[str, Any]:
        batched_sequences = batch_data([be.encoded_final_sequence for be in current_batch], self.tokenizer.pad_token_id)
        attention_masks = [torch.ones_like(be.encoded_final_sequence) for be in current_batch]
        batched_attention_masks = batch_data(attention_masks, pad_token_id=0)

        batch_dict = {
            "dataset_identifier": self.dataset_id,
            "sequences": batched_sequences,
            "attention_masks": batched_attention_masks,
        }

        rep_elem = current_batch[0]

        if rep_elem.start_position is not None:
            batch_dict["start_positions"] = torch.tensor([be.start_position for be in current_batch])

        if rep_elem.end_position is not None:
            batch_dict["end_positions"] = torch.tensor([be.end_position for be in current_batch])

        if rep_elem.token_type_ids is not None:
            batch_dict["token_type_ids"] = batch_data([be.token_type_ids for be in current_batch], pad_token_id=0)

        if rep_elem.possible_offsets is not None:
            batch_dict["possible_offsets"] = [be.possible_offsets for be in current_batch]

        if rep_elem.gold_labels is not None:
            batch_dict["gold_labels"] = [be.gold_labels for be in current_batch]

        if rep_elem.gloss_positions is not None:
            batch_dict["gloss_positions"] = [be.gloss_positions for be in current_batch]

        if rep_elem.wsd_instance is not None:
            batch_dict["wsd_instances"] = [be.wsd_instance for be in current_batch]

        return batch_dict

    def __iter__(self):

        if (self.re_init_on_iter and self.is_dataset_used) or len(self.data_store) == 0:
            self.build_dataset()

        self.is_dataset_used = True

        print("Len datastore: {}".format(len(self.data_store)))

        current_batch = []

        for data_elem in self.data_store:

            batch_max_len = max([be.encoded_final_sequence.size(0) for be in current_batch], default=0)

            if (
                max(data_elem.encoded_final_sequence.size(0), batch_max_len) * (len(current_batch) + 1)
                > self.tokens_per_batch
            ):
                if len(current_batch) != 0:
                    yield self.output_batch(current_batch)
                current_batch = []

            current_batch.append(data_elem)

        if len(current_batch) != 0:
            yield self.output_batch(current_batch)

    def __len__(self) -> int:
        if len(self.data_store) == 0:
            self.build_dataset()
        return len(self.data_store)


class WordNetDataset(QAExtractiveDataset):

    dataset_id = "wsd"

    def __init__(
        self,
        raganato_path: Union[str, List[str]],
        tokenizer: DefinitionsTokenizer,
        tokens_per_batch: int,
        re_init_on_iter: bool,
        is_test: bool = False,
        add_glosses_noise: bool = False,
        fix_glosses: bool = False,
        kshot: int = -1,
        poisson_lambda: int = 1,
    ) -> None:

        super().__init__(tokenizer, tokens_per_batch, re_init_on_iter, is_test)

        self.data_paths, self.keys_paths = [], []
        if type(raganato_path) == str:
            dp, kp = expand_raganato_path(raganato_path)
            self.data_paths.append(dp)
            self.keys_paths.append(kp)
        else:
            for rp in raganato_path:
                dp, kp = expand_raganato_path(rp)
                self.data_paths.append(dp)
                self.keys_paths.append(kp)

        self.add_glosses_noise = add_glosses_noise
        self.offsets_frequencies = None  # only used if the parameter "add_glosses_noise" is > 0.0
        self.fix_glosses = fix_glosses
        self.kshot = kshot
        self.poisson = poisson.Poisson(poisson_lambda)

    def __init_offsets_frequencies(self) -> None:

        offsets_counter = collections.Counter()

        for data_path, keys_path in zip(self.data_paths, self.keys_paths):
            for _, _, wsd_sentence in read_from_raganato(data_path, keys_path):
                for wsd_instance in wsd_sentence:
                    if wsd_instance.labels is None:
                        continue
                    else:
                        wn_offsets = [wn_offset_from_sense_key(_l) for _l in wsd_instance.labels]
                        offsets_counter.update(wn_offsets)

        total_occurrences = sum(offsets_counter.values())

        self.offsets_frequencies = [], []

        for wn_offset, count in offsets_counter.items():
            self.offsets_frequencies[0].append(wn_offset)
            self.offsets_frequencies[1].append(count / total_occurrences)

    def init_dataset(self) -> None:

        self.data_store = []

        if self.add_glosses_noise and self.offsets_frequencies is None:
            self.__init_offsets_frequencies()

        if self.kshot > 0:
            senses_counter = collections.Counter()

        for data_path, keys_path in zip(self.data_paths, self.keys_paths):

            raganato_iterable = read_from_raganato(data_path, keys_path)
            for _, _, wsd_sentence in raganato_iterable:

                sentence_tokens = [wsd_instance.annotated_token.text for wsd_instance in wsd_sentence]

                for i, wsd_instance in enumerate(wsd_sentence):

                    if wsd_instance.instance_id is None:
                        continue

                    gold_labels, current_label = None, None
                    if wsd_instance.labels is not None:
                        gold_labels = [wn_offset_from_sense_key(_l) for _l in wsd_instance.labels]
                        current_label = random.choice(gold_labels)

                    possible_offsets = wn_offsets_from_lemmapos(
                        wsd_instance.annotated_token.lemma,
                        pos_map.get(wsd_instance.annotated_token.pos, wsd_instance.annotated_token.pos),
                    )

                    if len(possible_offsets) == 0:
                        print(
                            f"No synsets found in WordNet for instance {wsd_instance.instance_id}. "
                            f"Skipping this instance"
                        )
                        print(wsd_instance)
                        continue

                    if not self.is_test:

                        # kshot
                        if self.kshot > 0:
                            if any([senses_counter[sk] > self.kshot for sk in wsd_instance.labels]):
                                continue
                            else:
                                senses_counter.update(wsd_instance.labels)

                        # randomly add glosses from other senses
                        if self.add_glosses_noise:
                            n_offsets_to_add = self.poisson.sample().item()
                            offsets, frequencies = self.offsets_frequencies
                            random_offsets = np.random.choice(
                                offsets, size=int(n_offsets_to_add), replace=False, p=frequencies
                            )
                            possible_offsets += random_offsets.tolist()

                        # remove gold glosses for multilabel instances
                        possible_offsets = [
                            po for po in possible_offsets if po == current_label or po not in gold_labels
                        ]

                        # remove possible duplicates
                        possible_offsets = list(set(possible_offsets))

                        if not self.fix_glosses:
                            np.random.shuffle(possible_offsets)

                    curr_sentence = self.tokenizer.insert_classify_tokens(sentence_tokens, index=i)
                    possible_glosses = [
                        synset_from_offset(po).definition().capitalize() + "." for po in possible_offsets
                    ]

                    encoded_final_sequence, gloss_positions, token_type_ids = self.tokenizer.prepare_sample(
                        curr_sentence, possible_glosses
                    )

                    start_position, end_position = None, None
                    if current_label is not None:
                        try:
                            label_idx = possible_offsets.index(current_label)
                        except ValueError:
                            print(
                                f"Instance {wsd_instance.instance_id} label ({current_label}) is not a valid label "
                                f"for WordNet given its lemma ({wsd_instance.annotated_token.lemma}) "
                                f"and pos ({wsd_instance.annotated_token.pos}. "
                                f"Skipping this instance"
                            )
                            continue

                        start_position, end_position = gloss_positions[label_idx]

                    data_elem = DataElement(
                        encoded_final_sequence=encoded_final_sequence,
                        start_position=start_position,
                        end_position=end_position,
                        possible_offsets=possible_offsets,
                        gold_labels=gold_labels,
                        gloss_positions=gloss_positions,
                        token_type_ids=token_type_ids,
                        wsd_instance=wsd_instance,
                    )

                    self.data_store.append(data_elem)


class OxfordDictionaryDataset(QAExtractiveDataset):

    dataset_id = "oxford"

    def __init__(
        self,
        dataset_path: str,
        tokenizer: DefinitionsTokenizer,
        tokens_per_batch: int,
        re_init_on_iter: bool,
        is_test: bool = False,
    ) -> None:

        super().__init__(tokenizer, tokens_per_batch, re_init_on_iter, is_test)

        self.lemmapos2glosses = collections.defaultdict(set)
        self.dataset_sentences = list()
        self.__init_kb(dataset_path)

    def __init_kb(self, dataset_path: str) -> None:

        with open("data/other_kbs/preprocessed_data/oxford_kb.tsv") as oxf:
            for line in oxf:
                lemmapos, *definitions = line.strip().split("\t")
                self.lemmapos2glosses[lemmapos] = definitions

        with open(dataset_path) as f:

            for line in f:

                defined_token, context_tagged_tokens, definition = line.strip().split("\t")

                context_tagged_tokens = context_tagged_tokens.split(" ")
                context_tokens = [ctt.split("#$#")[0] for ctt in context_tagged_tokens]

                if defined_token not in context_tokens:
                    continue

                defined_token_index = context_tokens.index(defined_token)

                defined_token_lemmapos = "#".join(reversed(context_tagged_tokens[defined_token_index].split("#$#")[1:]))

                nice_definition = definition.capitalize() + "."
                self.dataset_sentences.append(
                    (defined_token_index, defined_token_lemmapos, context_tokens, nice_definition)
                )

    def init_dataset(self) -> None:

        self.data_store = []

        for defined_token_index, defined_token_lemmapos, context_tokens, definition in self.dataset_sentences:

            context_sentence = self.tokenizer.insert_classify_tokens(context_tokens, defined_token_index)

            possible_definitions = list(self.lemmapos2glosses[defined_token_lemmapos])

            if not self.is_test:
                np.random.shuffle(possible_definitions)

            label_idx = possible_definitions.index(definition)

            encoded_final_sequence, definitions_positions, token_type_ids = self.tokenizer.prepare_sample(
                context_sentence, possible_definitions
            )

            start_position, end_position = definitions_positions[label_idx]

            data_elem = DataElement(
                encoded_final_sequence=encoded_final_sequence,
                start_position=start_position,
                end_position=end_position,
                possible_offsets=possible_definitions,
                gold_labels=[definition],
                gloss_positions=definitions_positions,
                token_type_ids=token_type_ids,
            )

            self.data_store.append(data_elem)


class DatasetAlternator(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], is_infinite: bool = False):
        self.datasets = {dataset.__class__.__name__: dataset for dataset in datasets}
        self.is_infinite = is_infinite

    def __iter__(self):

        datasets_iterable = {
            dataset.__class__.__name__: [dataset.__iter__(), False] for dataset in self.datasets.values()
        }

        while not all([x[1] for x in datasets_iterable.values()]):

            for dataset_name, (dataset_iterator, is_ended) in datasets_iterable.items():

                if not is_ended:

                    next_batch = next(dataset_iterator, None)

                    if next_batch is None:
                        if not self.is_infinite:
                            datasets_iterable[dataset_name][-1] = True
                        else:
                            datasets_iterable[dataset_name][0] = self.datasets[dataset_name].__iter__()
                    else:
                        yield next_batch


class SmartDatasetAlternator(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], is_infinite: bool = False):
        self.datasets = {dataset.__class__.__name__: dataset for dataset in datasets}
        self.is_infinite = is_infinite
        self.datasets_probabilities = None
        self.init_datasets_probabilities()

    def compute_datasets_len(self) -> Dict[str, int]:
        return {dn: sum(1 for _ in diter.__iter__()) for dn, diter in self.datasets.items()}

    def init_datasets_probabilities(self):
        datasets_len = self.compute_datasets_len()
        total_len = sum(datasets_len.values())
        self.datasets_probabilities = {dn: dn_len / total_len for dn, dn_len in datasets_len.items()}

    def __iter__(self):
        datasets_iterable = {dataset.__class__.__name__: dataset.__iter__() for dataset in self.datasets.values()}

        datasets, probabilities = [], []
        for dataset, prob in self.datasets_probabilities.items():
            datasets.append(dataset)
            probabilities.append(prob)

        def get_batch() -> Optional[Dict[str, Any]]:
            if len(datasets) == 0:
                return None

            current_dataset = (
                datasets[0] if len(datasets) == 0 else np.random.choice(datasets, 1, p=probabilities).item()
            )

            dataset_batch = next(datasets_iterable[current_dataset], None)

            if dataset_batch is None:
                if self.is_infinite:
                    datasets_iterable[current_dataset][0] = self.datasets[current_dataset].__iter__()
                else:
                    current_dataset_idx = datasets.index(current_dataset)
                    del datasets[current_dataset_idx]
                    del probabilities[current_dataset_idx]
                    prob_sum = sum(probabilities)
                    for i in range(len(probabilities)):
                        probabilities[i] = probabilities[i] / prob_sum
                return get_batch()

            return dataset_batch

        while True:
            next_batch = get_batch()
            if next_batch is not None:
                yield next_batch
            else:
                break
