from typing import Any, Optional, Callable, List

from nltk.corpus import wordnet as wn
import collections

from esc.utils.wordnet import wn_offset_from_synset

from tqdm import tqdm


def write_dict_on_file(
    output_path: str,
    dict_to_write: dict,
    key_mapper: Optional[Callable[[Any], str]] = None,
    value_mapper: Optional[Callable[[Any], str]] = None,
    sep: str = "\t",
) -> None:
    with open(output_path, "w") as f:
        for key, value in dict_to_write.items():
            if key_mapper is not None:
                key = key_mapper(key)
            if value_mapper is not None:
                value = value_mapper(value)
            f.write(f"{key}{sep}{value}\n")


def build_wordnet_dict(output_dir_path: str) -> None:
    synset2gloss = dict()
    synset2examples = dict()
    lemmapos2synsets = collections.defaultdict(list)
    synset2sensekeys = collections.defaultdict(list)

    for synset in tqdm(wn.all_synsets()):
        synset_offset = wn_offset_from_synset(synset)
        synset2gloss[synset_offset] = synset.definition()
        synset2examples[synset_offset] = synset.examples()

        synset_pos = synset.pos()
        for lemma in synset.lemmas():
            lemmapos = f"{lemma.name()}#{synset_pos}"
            lemmapos2synsets[lemmapos].append(synset_offset)

            lemma_key = lemma.key()
            synset2sensekeys[synset_offset].append(lemma_key)

    write_dict_on_file(f"{output_dir_path}/synset2gloss.tsv", synset2gloss)
    write_dict_on_file(f"{output_dir_path}/synset2examples.tsv", synset2examples, value_mapper=lambda v: "\t".join(v))
    write_dict_on_file(f"{output_dir_path}/lemmapos2synsets.tsv", lemmapos2synsets, value_mapper=lambda v: "\t".join(v))
    write_dict_on_file(f"{output_dir_path}/synset2sensekeys.tsv", synset2sensekeys, value_mapper=lambda v: "\t".join(v))


class KBManager:
    def __init__(self, dictionaries_dir: str) -> None:
        self.sensekey2synset = dict()
        self.lemmapos2synsets = dict()
        self.synset2gloss = dict()
        self.__init_data_structures(dictionaries_dir)

    def __init_data_structures(self, dictionaries_dir: str) -> None:
        # self.sensekey2synset
        with open(f"{dictionaries_dir}/synset2sensekeys.tsv") as f:
            for line in f:
                synset, *sensekeys = line.strip().split("\t")
                for sk in sensekeys:
                    self.sensekey2synset[sk] = synset

        # self.lemmapos2synsets
        with open(f"{dictionaries_dir}/lemmapos2synsets.tsv") as f:
            for line in f:
                lemmapos, *synsets = line.strip().split("\t")
                self.lemmapos2synsets[lemmapos] = synsets

        # self.synset2gloss
        with open(f"{dictionaries_dir}/synset2gloss.tsv") as f:
            for line in f:
                synset, gloss = line.strip().split("\t")
                self.synset2gloss[synset] = gloss

    def synset_offset_from_sense_key(self, sense_key: str) -> str:
        return self.sensekey2synset[sense_key]

    def synset_offsets_from_lemmapos(self, lemma, pos) -> List[str]:
        return self.lemmapos2synsets[f"{lemma}#{pos}"]

    def gloss_from_offset(self, offset: str) -> str:
        return self.synset2gloss[offset]


def main():
    build_wordnet_dict("data/dictionaries")


if __name__ == "__main__":
    main()
