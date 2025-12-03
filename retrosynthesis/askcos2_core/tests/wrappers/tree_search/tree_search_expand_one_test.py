import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "http://0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class RetroATTest(unittest.TestCase):
    """Test class for Retro Augmented Transformer wrapper"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.base_url = f"{V2_HOST}:{V2_PORT}/api"
        cls.module_url = f"{V2_HOST}:{V2_PORT}/api/tree-search/expand-one"

    def get_async_result(self, task_id: str, timeout: int = 20):
        """Retrieve celery task output"""
        # Try to get result 10 times in per sec interval
        for _ in range(timeout):
            response = self.session.get(
                f"{self.base_url}/celery/task/get?task_id={task_id}",
            )
            response = response.json()
            if response.get("complete"):
                return response
            else:
                if response.get("failed"):
                    print("Celery task failed!")

                    return response
                else:
                    time.sleep(1)
        else:
            print("Celery task timeout!")

            return response

    def test_1(self):
        case_file = "tests/wrappers/tree_search/tree_search_expand_one_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        # get sync response
        response_sync = self.session.post(
            f"{self.module_url}/call-sync-without-token", json=data
        ).json()

        # only check sync, as async endpoint typically requires authentication
        for response in [response_sync]:
            self.assertEqual(response["status_code"], 200)
            self.assertIsInstance(response["result"], list)
            self.assertIsInstance(response["result"][0], dict)
            self.assertIsInstance(response["result"][0]["outcome"], str)
            self.assertIsInstance(response["result"][0]["model_score"], float)
            self.assertIsInstance(response["result"][0]["normalized_model_score"], float)
            self.assertIsInstance(response["result"][0]["template"], dict)
            self.assertIsInstance(response["result"][0]["retro_backend"], str)
            self.assertIsInstance(response["result"][0]["retro_model_name"], str)
            self.assertIsInstance(response["result"][0]["plausibility"], float)
            self.assertIsInstance(response["result"][0]["rms_molwt"], float)
            self.assertIsInstance(response["result"][0]["num_rings"], int)
            self.assertIsInstance(response["result"][0]["scscore"], float)
