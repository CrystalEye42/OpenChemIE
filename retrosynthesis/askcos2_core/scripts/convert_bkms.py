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


def _get_tpl_from_entry(task: Tuple[int, dict]) -> Tuple[int, str]:
    i, reaction_entry = task
    reaction_smiles = reaction_entry['reaction_smiles']
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
    fn = "data/db/historian/reactions.bkms_metabolic.json.gz.old"
    ofn = "data/db/historian/reactions.bkms_metabolic.json.gz"

    failed_count = 0
    start = time.time()

    with gzip.open(fn, "rt", encoding="UTF-8") as zipfile:
        data = json.load(zipfile)

    with ProcessPool() as pool:
        # Using pebble to add timeout, as rdchiral could hang
        future = pool.map(
            _get_tpl_from_entry,
            enumerate(tqdm(data)),
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
