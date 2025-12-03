import copy
import pprint
import requests


query_template = {
    "smiles": "",
    "reagents": [],
    "n_conditions": 10
}


def main():
    HOST = "0.0.0.0"
    PORT = "9100"

    test_smiles = "CC(=O)O.Fc1ccccc1Nc1cccc2ccccc12>>Cc1c2cccc(F)c2nc2c1ccc1ccccc12"
    test_reagents = ["CC(=O)O"]
    data = copy.deepcopy(query_template)
    data["smiles"] = test_smiles
    data["reagents"] = test_reagents

    resp = requests.post(
        url=f"http://{HOST}:{PORT}/api/context-recommender/v2/condition/FP/call-sync",
        json=data
    ).json()

    print(f"Printing the first two results for the fingerprint (FP) model")
    pprint.pprint(resp["result"][:2])
    print()

    resp = requests.post(
        url=f"http://{HOST}:{PORT}/api/context-recommender/v2/condition/GRAPH/call-sync",
        json=data
    ).json()

    print(f"Printing the first two results for the GRAPH model")
    pprint.pprint(resp["result"][:2])
    print()


if __name__ == "__main__":
    main()
