import csv
import requests
from rdkit import Chem
from tqdm import tqdm


def extract_and_resolve():
    print("Extracting and resolving names from raw data...")

    names_smis = []
    invalid_names = []

    with open("compilation_of_cder_nme_and_new_biologic_approvals_1985-2023.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in tqdm(reader):
            if row["NDA/BLA"] == "BLA":
                continue
            if row["Approval Year"] not in ["2019", "2020", "2021", "2022", "2023"]:
                continue

            name = row["Active Ingredient/Moiety"]
            print(f"Processing {name}")
            try:
                resp = requests.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/txt",
                    timeout=20
                )
            except:
                print(f"Can't resolve name: {name}, probably timeout")
                continue

            if resp.status_code == 200:
                smi = resp.text.strip()
                names_smis.append((name, smi))
            else:
                print(f"Error resolving raw entry: {name}, dumped into invalid_names for further processing")
                invalid_names.append(name)
                pass

    # invalid_names contain contains names which are unresolvable as-is by NIH.
    # print(invalid_names)
    """
    invalid_names = [
        'imipenem, cilastatin, and relebactam',
        'elexacaftor, ivacaftor, tezacaftor; ivacaftor (co-packaged)',
        'air polymer-type A',
        'golodirsen', 'Brilliant Blue G Ophthalmic Solution', 'decitabine and cedazuridine',
        'lumasiran', 'cabotegravir; rilpivirine (co-packaged)', 'serdexmethylphenidate and dexmethylphenidate',
        'drospirenone and estetrol', 'pegcetacoplan', 'olanzapine and samidorphan',
        'lutetium Lu 177 vipivotide tetraxetan',
        'vonoprazan; amoxicillin; clarithromycin (co-packaged)', 'vutrisiran', 'sodium phenylbutyrate and taurursodiol',
        'tofersen', 'sulbactam; durlobactam (co-packaged)', 'nirmatrelvir; ritonavir (co-packaged)',
        'avacincaptad pegol', 'nedosiran', 'taurolidine and heparin', 'birch triterpenes', 'eplontersen'
    ]
    """

    # These were manually inspected and properly formated into processed_names
    # to be resolved again
    processed_names = [
        "imipenem",
        "cilastatin",
        "relebactam",
        "elexacaftor",
        "ivacaftor",
        "tezacaftor",
        "golodirsen",
        "decitabine",
        "cedazuridine",
        "lumasiran",
        "cabotegravir",
        "rilpivirine",
        "serdexmethylphenidate",
        "dexmethylphenidate",
        "drospirenone",
        "estetrol",
        "olanzapine",
        "samidorphan",
        "vonoprazan",
        "amoxicillin",
        "clarithromycin",
        "vutrisiran",
        "sodium phenylbutyrate",
        "taurursodiol",
        "tofersen",
        "sulbactam",
        "durlobactam",
        "nirmatrelvir",
        "ritonavir",
        "taurolidine",
        "heparin"
    ]

    for name in tqdm(processed_names):
        print(f"Processing manually formatted name: {name}")
        try:
            resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/txt",
                timeout=20
            )
        except:
            print(f"Can't resolve name: {name}, probably timeout")
            continue

        if resp.status_code == 200:
            smi = resp.text.strip()
            names_smis.append((name, smi))
        else:
            print(f"Can't resolve name: {name}")

    cano_smis = []
    with open("FDA_approved_resolved_19-23.csv", "w") as of:
        # deduplication
        for name, smi in names_smis:
            cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            if cano_smi not in cano_smis:
                cano_smis.append(cano_smi)
                of.write(f"{name}\t{cano_smi}\n")


def filter_buyables():
    print("Filtering out targets which are buyables...")

    s = requests.Session()
    buyable = 0

    with open("FDA_approved_resolved_19-23.csv", "r") as f, \
            open("FDA_approved_filtered_19-23.csv", "w") as of:
        for line in f:
            # filter out multi-components
            if "." in line:
                continue
            name, smi = line.strip().split("\t")

            # remove isotope label (by smi-hardcoding)
            if smi == "[129Xe]":
                print(f"Skipping {smi}..")
                continue
            if smi == "[2H]C([2H])([2H])NC(=O)c1nnc(NC(=O)C2CC2)cc1Nc1cccc(-c2ncn(C)n2)c1OC":
                smi = "CNC(=O)c1nnc(NC(=O)C2CC2)cc1Nc1cccc(-c2ncn(C)n2)c1OC"
            if "[18F]" in smi:
                print(f"Replacing [18F] with F for {smi}..")
                smi = smi.replace("[18F]", "F")

            # buyability check
            resp = s.post(
                url=f"http://{HOST}:{PORT}/api/buyables/lookup",
                json={"smiles": [smi]}
            ).json()
            if resp["result"]:
                print(f"Skipping buyable {smi}..")
                buyable += 1
                continue

            of.write(f"{name}\t{smi}\n")

    print(f"Buyable count: {buyable}")


def main():
    extract_and_resolve()
    filter_buyables()


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = "9100"
    HOST = "72.70.38.132"
    PORT = "9918"

    main()
