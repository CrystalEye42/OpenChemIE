import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class V1ContextTest(unittest.TestCase):
    """Test class for V1 context adapter"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.v1_username = os.environ.get("V1_USERNAME", "askcos")
        cls.v1_password = os.environ.get("V1_PASSWORD", "")

        cls.v1_context_url = "https://askcos-demo.mit.edu/api/v2/context/"
        cls.v1_celery_url = "https://askcos-demo.mit.edu/api/v2/celery/task"
        cls.legacy_adapter_url = f"http://{V2_HOST}:{V2_PORT}/api/legacy/context/"
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
        case_file = "tests/adapters/v1_context/v1_context_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        task_ids = {}
        for mode, url in [("v1", self.v1_context_url),
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
            self.assertEqual(request["num_results"], data["num_results"])
            self.assertTrue(request["return_scores"])

            # Test that we got the celery task id
            self.assertIsInstance(result["task_id"], str)
            task_ids[mode] = result["task_id"]

        results = {}
        for mode, celery_url in [("v1", self.v1_celery_url),
                                 ("legacy", self.legacy_celery_url)]:
            task_id = task_ids[mode]
            print(task_id)

            # Try retrieving task output
            result = self.get_result(task_id=task_id, celery_url=celery_url, mode=mode)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["output"]), 5)
            o = result["output"][0]
            print(o)
            self.assertEqual(o["catalyst"], "")
            self.assertEqual(o["reagent"], "Cc1ccccc1.[H][N-][H].[Na+]")
            self.assertAlmostEqual(o["score"], 0.339, places=2)
            self.assertEqual(o["solvent"], "")
            self.assertAlmostEqual(o["temperature"], 94.48, places=1)

            results[mode] = result

        # Added for v2, consistency check
        self.assertAlmostEqual(
            results["v1"]["output"][0]["temperature"],
            results["legacy"]["output"][0]["temperature"],
            places=4
        )
