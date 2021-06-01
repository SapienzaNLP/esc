import collections
from enum import Enum
from typing import List, Tuple, Optional, Set, Iterable
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from esc.utils.wsd import read_from_raganato


def wn_offset_from_synset(synset: Synset) -> str:
    return str(synset.offset()).zfill(8) + synset.pos()


def wn_offset_from_sense_key(sense_key: str) -> str:
    synset = wn.lemma_from_key(sense_key).synset()
    return wn_offset_from_synset(synset)


def synset_from_offset(synset_offset: str) -> Synset:
    return wn.of2ss(synset_offset)


def gloss_from_sense_key(sense_key: str) -> str:
    return wn.lemma_from_key(sense_key).synset().definition()


def synsets_from_lemmapos(lemma: str, pos: str) -> List[Synset]:
    return wn.synsets(lemma, pos)


def wn_offsets_from_lemmapos(lemma: str, pos: str) -> List[str]:
    return [wn_offset_from_synset(syns) for syns in synsets_from_lemmapos(lemma, pos)]


class WNRelation(Enum):
    HYPONYMY = 0
    HOLONYMY = 1
    ANTONYMY = 2
    SIBLINGS = 3
    HYPERNYMY = 4
    MERONYMY = 5

    def inverse(self) -> Optional["WNRelation"]:
        if self == WNRelation.HYPONYMY:
            return WNRelation.HYPERNYMY
        elif self == WNRelation.HYPERNYMY:
            return WNRelation.HYPONYMY
        elif self == WNRelation.HOLONYMY:
            return WNRelation.MERONYMY
        elif self == WNRelation.HOLONYMY:
            return WNRelation.MERONYMY
        else:
            return None


def extract_from_relation(synset: Synset, relation: WNRelation) -> List[Synset]:
    if relation == WNRelation.HYPONYMY:
        return synset.hyponyms()
    elif relation == WNRelation.HOLONYMY:
        return synset.part_holonyms() + synset.member_holonyms() + synset.substance_holonyms()
    elif relation == WNRelation.ANTONYMY:
        return [_l.synset() for _l in synset.lemmas()[0].antonyms()]
    elif relation == WNRelation.HYPERNYMY:
        return synset.hypernyms()
    elif relation == WNRelation.MERONYMY:
        return synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms()
    else:
        raise NotImplementedError


def senses_from_relation(
    starting_synset: Synset, relations: List[WNRelation], with_inverse: bool, working_relations: Set[WNRelation]
) -> List[Tuple[str, str, WNRelation]]:
    starting_synset_offset = wn_offset_from_synset(starting_synset)
    edges = []

    for relation in relations:
        extracted_synsets = extract_from_relation(starting_synset, relation)

        for extracted_synset in extracted_synsets:
            extracted_synset_offset = wn_offset_from_synset(extracted_synset)
            if relation in working_relations:
                edges.append((starting_synset_offset, extracted_synset_offset, relation))

            if with_inverse and relation.inverse() is not None:
                if relation.inverse() in working_relations:
                    edges.append((extracted_synset_offset, starting_synset_offset, relation.inverse()))

    return edges


def one_hop_synsets(
    synset_offsets: Iterable[str],
    relations: List[WNRelation],
    with_inverse: bool,
    working_relations: Optional[Set[WNRelation]] = None,
) -> List[Tuple[str, str, WNRelation]]:
    synsets = [synset_from_offset(synset_offset) for synset_offset in synset_offsets]
    one_hop_edges = []
    for synset in synsets:
        one_hop_edges += senses_from_relation(synset, relations, with_inverse, working_relations)
    return one_hop_edges


def compute_gloss_vocab():
    offset2idx = dict()
    for synset in wn.all_synsets():
        offset2idx[wn_offset_from_synset(synset)] = len(offset2idx)
    return offset2idx


def main_compute_vocab():
    gloss_vocab = compute_gloss_vocab()
    with open("data/gloss_vocab.tsv", "w") as f:
        for offset, gloss_id in gloss_vocab.items():
            f.write(f"{offset}\t{gloss_id}\n")


def main():

    # semeval2007 = read_from_raganato(
    #     '/media/ssd/wsd-biencoders/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml',
    #     '/media/ssd/wsd-biencoders/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
    # )

    semeval2007 = read_from_raganato(
        "/media/ssd/wsd-biencoders/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
        "/media/ssd/wsd-biencoders/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    )

    semeval2007 = [x[2] for x in semeval2007]

    semeval_2007_senses = [
        wsd_instance.labels
        for wsd_sentence in semeval2007
        for wsd_instance in wsd_sentence
        if wsd_instance.labels is not None
    ]
    semeval_2007_senses = set([x for y in semeval_2007_senses for x in y])

    oh_synsets = one_hop_synsets(
        semeval_2007_senses,
        [WNRelation.HYPERNYMY, WNRelation.HYPONYMY, WNRelation.ANTONYMY, WNRelation.MERONYMY, WNRelation.HOLONYMY],
        with_inverse=True,
    )

    sources_dict = collections.defaultdict(list)
    target_dict = collections.defaultdict(list)

    for relation_triplet in oh_synsets:
        s, t, _ = relation_triplet
        sources_dict[s].append(relation_triplet)
        target_dict[t].append(relation_triplet)

    semeval_2007_synsets = set([wn.lemma_from_key(ss).synset() for ss in semeval_2007_senses])

    counter = []
    for s, t, rel in oh_synsets:
        if s in semeval_2007_synsets and t in semeval_2007_synsets:
            counter.append((s, t, rel))

    print(counter)


if __name__ == "__main__":
    main_compute_vocab()
