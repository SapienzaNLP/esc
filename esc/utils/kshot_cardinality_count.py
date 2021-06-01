import collections
import argparse


def count_cardinality(gold_keys_path: str, k: int) -> int:
    senses_counter = collections.Counter()
    instances_counter = 0
    with open(gold_keys_path) as f:
        for line in f:
            _, *gold_senses = line.strip().split(" ")
            if any(senses_counter[_gs] >= k for _gs in gold_senses):
                continue
            instances_counter += 1
            senses_counter.update(gold_senses)

    return instances_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-keys-path", required=True)
    parser.add_argument("--ks", required=True)
    return parser.parse_args()


def main(args):
    ks = map(int, args.ks.split(","))
    for k in ks:
        total_count = count_cardinality(args.gold_keys_path, k)
        print(f"Total instances count for k-{k} = {total_count}")


if __name__ == "__main__":
    main(parse_args())
