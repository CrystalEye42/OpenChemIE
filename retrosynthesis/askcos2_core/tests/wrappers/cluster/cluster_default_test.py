import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "http://0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class ClusterTest(unittest.TestCase):
    """Test class for Cluster wrapper"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.base_url = f"{V2_HOST}:{V2_PORT}/api"
        cls.module_url = f"{V2_HOST}:{V2_PORT}/api/cluster"

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
        case_file = "tests/wrappers/cluster/cluster_default_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        # get sync response
        response_sync = self.session.post(
            f"{self.module_url}/call-sync", json=data
        ).json()

        # get async response
        task_id = self.session.post(
            f"{self.module_url}/call-async", json=data
        ).json()
        response_async = self.get_async_result(
            task_id=task_id,
            timeout=60
        )
        response_async = response_async["output"]

        for response in [response_sync, response_async]:
            self.assertEqual(response["status_code"], 200)
            self.assertIsInstance(response["result"], list)
            self.assertIsInstance(response["result"][0], list)
            self.assertIsInstance(response["result"][1], dict)
