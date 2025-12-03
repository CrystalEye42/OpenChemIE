import json
import os
import requests
import time
import unittest

V2_HOST = os.environ.get("V2_HOST", "0.0.0.0")
V2_PORT = os.environ.get("V2_PORT", "9100")


class V1SolubilityTest(unittest.TestCase):
    """Test class for V1 Selectivity adapter"""

    @classmethod
    def setUpClass(cls) -> None:
        """This method is run once before all tests in this class."""
        cls.session = requests.Session()
        cls.v1_username = os.environ.get("V1_USERNAME", "askcos")
        cls.v1_password = os.environ.get("V1_PASSWORD", "")

        cls.v1_solubility_url = "https://askcos-demo.mit.edu/api/v2/solubility/"
        cls.v1_celery_url = "https://askcos-demo.mit.edu/api/v2/celery/task"
        cls.legacy_adapter_url = f"http://{V2_HOST}:{V2_PORT}/api/legacy/solubility/"
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
        case_file = "tests/adapters/v1_solubility/v1_solubility_test_case_1.json"
        with open(case_file, "r") as f:
            data = json.load(f)

        task_ids = {}
        for mode, url in [("v1", self.v1_solubility_url),
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
            self.assertEqual(request["solvent"], data["solvent"])
            self.assertEqual(request["solute"], data["solute"])
            self.assertEqual(request["temp"], data["temp"])

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
            self.assertIsInstance(result["output"], list)
            #self.assertEqual(len(result["output"]), 123)

            results[mode] = result

        # Added for v2, consistency check
        for r1, r2 in zip(results["v1"]["output"], results["legacy"]["output"], strict=True):
            print("This is the v1 output:")
            print(r1)
            print("This is the v2 output:")
            print(r2)
            self.assertEqual(r1["Solvent"], r2["Solvent"])
            self.assertEqual(r1["Solute"], r2["Solute"])
            self.assertEqual(r1["Temp"], r2["Temp"])
            self.assertEqual(r1["logST (method1) [log10(mol/L)]"], r2["log_st_1"])
            self.assertEqual(r1["logST (method2) [log10(mol/L)]"], r2["log_st_2"])
            self.assertEqual(r1["dGsolvT [kcal/mol]"], r2["dg_solv_t"])
            self.assertEqual(r1["dHsolvT [kcal/mol]"], r2["dh_solv_t"])
            self.assertEqual(r1["dSsolvT [cal/K/mol]"], r2["ds_solv_t"])
            self.assertEqual(r1["Pred. Hsub298 [kcal/mol]"], r2["pred_hsub298"])
            self.assertEqual(r1["Pred. Cpg298 [cal/K/mol]"], r2["pred_cpg298"])
            self.assertEqual(r1["Pred. Cps298 [cal/K/mol]"], r2["pred_cps298"])
            self.assertEqual(r1["logS298 [log10(mol/L)]"], r2["log_s_298"])
            self.assertEqual(r1["uncertainty logS298 [log10(mol/L)]"], r2["uncertainty_log_s_298"])
            # self.assertEqual(r1["dGsolv298 [kcal/mol]"], r2[""])
            # self.assertEqual(r1["uncertainty dGsolv298 [kcal/mol]"], r2[""])
            self.assertEqual(r1["dHsolv298 [kcal/mol]"], r2["dh_solv_298"])
            self.assertEqual(r1["uncertainty dHsolv298 [kcal/mol]"], r2["uncertainty_dh_solv_298"])
            self.assertEqual(r1["E"], r2["E"])
            self.assertEqual(r1["S"], r2["S"])
            self.assertEqual(r1["A"], r2["A"])
            # self.assertEqual(r1[""], r2["B"])
            self.assertEqual(r1["L"], r2["L"])
            self.assertEqual(r1["V"], r2["V"])
            self.assertEqual(r1["ST (method1) [mg/mL]"], r2["st_1"])
            self.assertEqual(r1["ST (method2) [mg/mL]"], r2["st_2"])
            self.assertEqual(r1["S298 [mg/mL]"], r2["s_298"])
