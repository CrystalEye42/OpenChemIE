import copy
import pprint
import requests


query_template = {
    "backend": "template_relevance",
    "model_name": "reaxys",
    "smiles": [],
    "max_num_templates": 1000,
    "max_cum_prob": 0.995,
    "attribute_filter": []
}


def main():
    HOST = "0.0.0.0"
    PORT = "9100"

    test_smiles = [
        "CS(=N)(=O)Cc1cccc(Br)c1",
        "CN(C)CCOC(c1ccccc1)c1ccccc1"
    ]
    data = copy.deepcopy(query_template)
    data["smiles"] = test_smiles

    resp = requests.post(
        url=f"http://{HOST}:{PORT}/api/retro/controller/call-sync",
        json=data
    ).json()

    for smi, res in zip(test_smiles, resp["result"]):
        print(f"Printing the first two results for SMILES {smi}")
        pprint.pprint(res[:2])
        print()


if __name__ == "__main__":
    main()
