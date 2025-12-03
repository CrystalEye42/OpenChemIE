import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class V1ReactionClassificationTest(unittest.TestCase):
    """Test class for V1 Reaction Classification"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.v1_username = os.environ.get("V1_USERNAME", "askcos")
        cls.v1_password = os.environ.get("V1_PASSWORD", "")

        cls.v1_reaction_classification_url = "https://askcos-demo.mit.edu/api/v2/reaction-classification/"
        cls.v1_celery_url = "https://askcos-demo.mit.edu/api/v2/celery/task"
        cls.legacy_adapter_url = f"http://{V2_HOST}:{V2_PORT}/api/legacy/reaction-classification/"
        cls.legacy_celery_url = f"http://{V2_HOST}:{V2_PORT}/api/legacy/celery/task"

    def get_result(self, task_id: str, celery_url: str, mode: str, timeout: int = 20):
        """Retrieve celery task output"""
        # Try to get result 10 times in 2 sec intervals
        for _ in range(timeout // 2):
            if mode == "v1":
                response = self.session.get(
                    url=f"{celery_url}/{task_id}/",
                    auth=(self.v1_username, self.v1_password)
                )
            else:
                response = self.session.get(f"{celery_url}/{task_id}/")

            result = response.json()
            if result.get("complete"):
                return result
            else:
                if result.get("failed"):
                    self.fail("Celery task failed.")
                else:
                    time.sleep(2)

    def test_1(self):
        case_file = "tests/adapters/v1_reaction_classification/v1_reaction_classification_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        task_ids = {}
        for mode, url in [("v1", self.v1_reaction_classification_url),
                          ("legacy", self.legacy_adapter_url)]:
            if mode == "v1":
                response = self.session.post(
                    url=url,
                    json=data,
                    auth=(self.v1_username, self.v1_password)
                )
            else:
                response = self.session.post(url, json=data)

            # Copied from askcos_site/api2/api_test.py
            self.assertEqual(response.status_code, 200)

            # Confirm that request was interpreted correctly
            result = response.json()
            request = result["request"]
            self.assertEqual(request["reactants"], data["reactants"])
            self.assertEqual(request["products"], data["products"])

            # Test that we got the celery task id
            self.assertIsInstance(result["task_id"], str)
            task_ids[mode] = result["task_id"]

        results = {}
        for mode, celery_url in [("v1", self.v1_celery_url),
                                 ("legacy", self.legacy_celery_url)]:
            task_id = task_ids[mode]

            # Try retrieving task output
            result = self.get_result(task_id=task_id, celery_url=celery_url, mode=mode)
            self.assertTrue(result["complete"])
            self.assertIsInstance(result["output"]["result"], list)
            #self.assertEqual(len(result["output"]), 123)
            results[mode] = result

        # Added for v2, consistency check
        v1_output = results["v1"]["output"]["result"]
        legacy_output = results["legacy"]["output"]["result"]

        # Get the length of the shorter list
        min_length = len(v1_output)

        # Truncate both lists to the length of the shorter list
        v1_output = v1_output[:min_length]
        legacy_output = legacy_output[:min_length]
        for r1, r2 in zip(v1_output, legacy_output, strict=True):
            self.assertEqual(r1["rank"], r2["rank"])
            self.assertEqual(r1["reaction_num"], r2["reaction_num"])
            self.assertEqual(r1["reaction_name"], r2["reaction_name"])
            self.assertEqual(r1["reaction_classnum"], r2["reaction_classnum"])
            self.assertEqual(r1["reaction_classname"], r2["reaction_classname"])
            self.assertEqual(r1["reaction_superclassnum"], r2["reaction_superclassnum"])
            self.assertEqual(r1["reaction_superclassname"], r2["reaction_superclassname"])
            self.assertAlmostEqual(r1["prediction_certainty"], r2["prediction_certainty"], places=4)
            # for s1, s2 in zip(r1["atom_scores"], r2["atom_scores"]):
            #     self.assertAlmostEqual(s1, s2, places=4)
