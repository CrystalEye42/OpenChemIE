import gzip
import json
import os
import sys
import time
from pebble import ProcessPool
# from rdchiral.template_extractor import extract_from_reaction
from rdkit import RDLogger
from scripts.convert_reaction_helper import extract_from_reaction_general
from tqdm import tqdm
from typing import Tuple

RDLogger.DisableLog("rdApp.*")


class BlockPrint:
    """Context manager to suppress printing from subprocesses"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _get_tpl_from_entry(task: Tuple[int, str]) -> Tuple[int, str]:
    i, reaction_smiles = task
    r_smi, _, p_smi = reaction_smiles.split()[0].split(">")
    reaction = {'_id': None, 'reactants': r_smi, 'products': p_smi}
    try:
        with BlockPrint():
            template = extract_from_reaction_general(reaction, super_general=True)
        reaction_smarts = template["reaction_smarts"]
    except:
        reaction_smarts = ""

    return i, reaction_smarts


def main():
    fn = "1976_Sep2016_USPTOgrants_smiles.rsmi"
    ofn = "data/db/historian/reactions.USPTO_FULL.json.gz"

    data = []
    all_reaction_smiles = []
    failed_count = 0
    start = time.time()
    with open(fn, "r") as f:
        for i, line in enumerate(tqdm(f)):
            if i == 0:
                continue
            (
                reaction_smiles,
                patent_number,
                paragraph_num,
                year,
                text_mined_yield,
                calculated_yield
            ) = line.strip("\n").split("\t")
            reaction_entry = {
                "_id": f"USPTO_FULL_{i}",
                "reaction_id": i,
                "template_set": "USPTO_FULL",
                "reaction_smiles": reaction_smiles,
                "patent_number": patent_number,
                "paragraph_num": paragraph_num,
                "year": year,
                "text_mined_yield": text_mined_yield,
                "calculated_yield": calculated_yield
            }

            data.append(reaction_entry)
            all_reaction_smiles.append(reaction_smiles)

    # test_count = 1000
    # data = data[:test_count]
    # all_reaction_smiles = all_reaction_smiles[:test_count]

    with ProcessPool() as pool:
        # Using pebble to add timeout, as rdchiral could hang
        future = pool.map(
            _get_tpl_from_entry,
            enumerate(tqdm(all_reaction_smiles)),
            timeout=10
        )
        iterator = future.result()

        # The while True - try/except/StopIteration is just pebble signature
        while True:
            try:
                i, reaction_smarts = next(iterator)
                if i > 0 and i % 100000 == 0:
                    print(f"Processing {i}th reaction, "
                          f"elapsed time: {time.time() - start: .0f} s")
                if not reaction_smarts:
                    failed_count += 1
                data[i]["reaction_smarts"] = reaction_smarts
            except StopIteration:
                break
            except TimeoutError as error:
                print(f"i: {i}. get_tpl() call took more than {error.args} seconds.")
                failed_count += 1
            except:
                print(f"i: {i}. Unknown error for getting template.")
                failed_count += 1

    print(f"Failed count: {failed_count}")

    with gzip.open(ofn, "wt", encoding="UTF-8") as zipfile:
        json.dump(data, zipfile)


if __name__ == "__main__":
    main()
