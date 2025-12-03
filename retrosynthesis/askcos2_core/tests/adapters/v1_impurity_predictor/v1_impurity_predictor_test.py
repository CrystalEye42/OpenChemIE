import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class V1SelectivityTest(unittest.TestCase):
    """Test class for V1 ImpurityPredictor adapter"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.v1_username = os.environ.get("V1_USERNAME", "askcos")
        cls.v1_password = os.environ.get("V1_PASSWORD", "")

        cls.v1_impurity_predictor_url = "https://askcos-demo.mit.edu/api/v2/impurity/"
        cls.v1_celery_url = "https://askcos-demo.mit.edu/api/v2/celery/task"
        cls.legacy_adapter_url = f"http://{V2_HOST}:{V2_PORT}/api/legacy/impurity/"
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

    @unittest.skip(
        reason="Test turned off for impurity consistency check "
               "as it takes a very long time (at least a few minutes)"
    )
    def test_1(self):
        case_file = "tests/adapters/v1_impurity_predictor/v1_impurity_predictor_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        task_ids = {}
        for mode, url in [("v1", self.v1_impurity_predictor_url),
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
            self.assertEqual(request["training_set"], data["training_set"])
            self.assertEqual(request["threshold"], data["threshold"])
            self.assertEqual(request["inspector"], data["inspector"])

            # Test that we got the celery task id
            self.assertIsInstance(result["task_id"], str)
            task_ids[mode] = result["task_id"]

        results = {}
        for mode, celery_url in [("v1", self.v1_celery_url),
                                 ("legacy", self.legacy_celery_url)]:
            task_id = task_ids[mode]

            # Try retrieving task output
            result = self.get_result(task_id=task_id, celery_url=celery_url, mode=mode, timeout=210)
            self.assertTrue(result["complete"])
            self.assertIsInstance(result["output"], dict)

            results[mode] = result
        v1_output = results["v1"]["output"]["predict_expand"]
        legacy_output = results["legacy"]["output"]["predict_expand"]

        # Get the length of the shorter list
        min_length = len(v1_output)

        # Truncate both lists to the length of the shorter list
        v1_output = v1_output[:min_length]
        legacy_output = legacy_output[:min_length]

        # Added for v2, consistency check
        # for r1, r2 in zip(v1_output, legacy_output, strict=True):
        #     print(r1)
        #     print(r2)
        #     self.assertEqual(r1["no"], r2["no"])
        #     self.assertEqual(r1["prd_smiles"], r2["prd_smiles"])
        #     self.assertAlmostEqual(r1["prd_mw"], r2["prd_mw"], places=4)
        #     for s1, s2 in zip(r1["rct_rea_sol"], r2["rct_rea_sol"], strict=True):
        #         self.assertEqual(s1["rct_smiles"], s2["rct_smiles"])
        #     for x1, x2 in zip(r1["insp_scores"], r2["insp_scores"], strict=True):
        #         self.assertAlmostEqual(x1, x2, places=4)
        #     self.assertAlmostEqual(r1["avg_insp_score"], r2["avg_insp_score"], places=4)
        #     self.assertEqual(r1["similarity_to_major"], r2["similarity_to_major"])
        #     self.assertEqual(r1["modes"], r2["modes"])
