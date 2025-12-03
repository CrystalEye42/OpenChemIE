import copy
import pprint
import requests


query_template = {
    "backend": "wldn5",
    "model_name": "pistachio",
    "smiles": [],
    "reagents": "",
    "solvent": ""
}


def main():
    HOST = "0.0.0.0"
    PORT = "9100"

    test_smiles = [
        "CS(=N)(=O)Cc1cccc(Br)c1.Nc1cc(Br)ccn1",
        "CCO.CC(=O)O"
    ]
    data = copy.deepcopy(query_template)
    data["smiles"] = test_smiles

    resp = requests.post(
        url=f"http://{HOST}:{PORT}/api/forward/controller/call-sync",
        json=data
    ).json()

    for smi, res in zip(test_smiles, resp["result"]):
        print(f"Printing the first two results for SMILES {smi}")
        pprint.pprint(res[:2])
        print()


if __name__ == "__main__":
    main()
