import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_extractor import FeatureExtractor  # noqa: E402


class FeatureEncodingTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(fetch_html=False)

    def test_url_features_follow_uci_direction(self):
        self.assertEqual(self.extractor._f_having_IP_Address("192.0.2.1"), -1)
        self.assertEqual(self.extractor._f_having_IP_Address("example.com"), 1)
        self.assertEqual(self.extractor._f_URL_Length("x" * 53), 1)
        self.assertEqual(self.extractor._f_URL_Length("x" * 54), 0)
        self.assertEqual(self.extractor._f_URL_Length("x" * 76), -1)
        self.assertEqual(self.extractor._f_having_At_Symbol("https://a@b.example"), -1)
        self.assertEqual(self.extractor._f_Prefix_Suffix("safe-example.com"), -1)
        self.assertEqual(self.extractor._f_HTTPS_token("https-login.example"), -1)

    def test_html_features_follow_uci_direction(self):
        safe = BeautifulSoup("<html><body></body></html>", "html.parser")
        suspicious = BeautifulSoup(
            "<script>window.open('x'); document.addEventListener('contextmenu', e => e.preventDefault());</script>"
            "<iframe src='about:blank' frameborder='0'></iframe>",
            "html.parser",
        )
        self.assertEqual(self.extractor._f_Submitting_to_email(safe), 1)
        self.assertEqual(self.extractor._f_popUpWidnow(suspicious), -1)
        self.assertEqual(self.extractor._f_RightClick(suspicious), -1)
        self.assertEqual(self.extractor._f_Iframe(suspicious), -1)

    def test_redirect_values_stay_inside_arff_domain(self):
        self.assertEqual(self.extractor._f_Redirect(SimpleNamespace(history=[])), 1)
        self.assertEqual(self.extractor._f_Redirect(SimpleNamespace(history=[1, 2])), 0)
        self.assertEqual(self.extractor._f_Redirect(SimpleNamespace(history=[1] * 5)), 0)

    def test_ssl_state_uses_legitimate_positive_encoding(self):
        https = urlparse("https://example.com")
        http = urlparse("http://example.com")
        self.assertEqual(self.extractor._f_SSLfinal_State(https, "valid"), 1)
        self.assertEqual(self.extractor._f_SSLfinal_State(https, "invalid"), 0)
        self.assertEqual(self.extractor._f_SSLfinal_State(http, "none"), -1)


if __name__ == "__main__":
    unittest.main()
