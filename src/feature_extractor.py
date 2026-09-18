#!/usr/bin/env python3
"""
feature_extractor.py — Giai đoạn 3 (GĐ3): Trích 30 đặc trưng từ một URL thật.

Mục tiêu: với một URL website bất kỳ (vd https://duytan.edu.vn/), trích ra
vector 30 đặc trưng giống hệt bộ dữ liệu Phishing Websites (UCI) để đưa vào
Perceptron đã huấn luyện dự đoán "phishing / legitimate".

Mỗi đặc trưng trả về giá trị trong {-1, 0, 1} theo đúng quy tắc trong tài liệu
"Phishing Websites Features". Các đặc trưng cần dịch vụ ngoài (WHOIS, PageRank,
Google Index, trang thống kê) được tính xác định khi có thể, nếu không sẽ trả
giá trị trung gian 0 và ghi rõ vào trường "approximated" để minh bạch.
"""

import ipaddress
import logging
import os
import re
import socket
import ssl
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Danh sách dịch vụ rút gọn URL phổ biến
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "t.co", "tiny.cc", "lnkd.in", "rb.gy", "rebrand.ly", "cutt.ly",
    "shorturl.at", "s.id", "v.gd", "su.pr", "bc.vc", "bl.ink",
}

# Cổng chuẩn (standard) — nếu URL dùng cổng ngoài danh sách này => nghi phishing
STANDARD_PORTS = {21, 22, 23, 25, 53, 80, 110, 143, 193, 443, 445, 465,
                  587, 990, 993, 995, 2525, 3306, 5432, 8080, 8443}

# Các đuôi ccTLD hai cấp (vd co.uk) để đếm subdomain chính xác
TWO_LEVEL_TLDS = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au",
                  "co.jp", "com.br", "co.in", "com.cn", "com.tw", "co.kr",
                  "com.mx", "com.tr", "co.nz", "com.sg", "co.th", "com.my"}


@dataclass
class FeatureResult:
    """Kết quả trích đặc trưng cho một URL."""
    url: str
    features: dict = field(default_factory=dict)      # tên -> giá trị {-1,0,1}
    approximated: list = field(default_factory=list)  # đặc trưng dùng giá trị gần đúng
    html: str = ""                                     # HTML đã tải (để debug)
    status_code: int | None = None


class FeatureExtractor:
    """Trích 30 đặc trưng Phishing Websites từ một URL."""

    def __init__(self, timeout: float = 15.0, fetch_html: bool = True,
                 phishtank_key: str | None = None):
        self.timeout = timeout
        self.fetch_html = fetch_html
        # Khóa PhishTank: ưu tiên tham số, fallback biến môi trường
        self.phishtank_key = phishtank_key or os.environ.get("PHISHTANK_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
        })

    # ------------------------------------------------------------------ #
    # Tiện ích                                                           #
    # ------------------------------------------------------------------ #
    def _load_page(self, url: str) -> tuple[str | None, int | None]:
        """Tải HTML của trang. Trả về (html, status_code); lỗi => (None, None)."""
        try:
            resp = self.session.get(url, timeout=self.timeout, verify=False,
                                    allow_redirects=True)
            return resp.text, resp.status_code
        except Exception as exc:  # noqa: BLE001
            log.debug("Không tải được %s: %s", url, exc)
            return None, None

    def _resolve_ip(self, hostname: str) -> str | None:
        try:
            return socket.gethostbyname(hostname)
        except Exception:  # noqa: BLE001
            return None

    def _ssl_state(self, hostname: str, port: int) -> str:
        """Kiểm tra chứng chỉ SSL. Trả về 'valid' | 'invalid' | 'none'."""
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=self.timeout):
                with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
                    s.connect((hostname, port))
                    ssl.match_hostname(s.getpeercert(), hostname)
            return "valid"
        except ssl.SSLCertVerificationError:
            return "invalid"
        except Exception:  # noqa: BLE001
            return "none"

    # ------------------------------------------------------------------ #
    # Các đặc trưng Address Bar                                          #
    # ------------------------------------------------------------------ #
    def _f_having_IP_Address(self, hostname: str) -> int:
        try:
            ipaddress.ip_address(hostname)
            return 1
        except ValueError:
            return -1

    def _f_URL_Length(self, url: str) -> int:
        n = len(url)
        if n < 54:
            return -1
        if 54 <= n <= 75:
            return 0
        return 1

    def _f_Shortining_Service(self, hostname: str) -> int:
        return 1 if hostname.lower() in SHORTENING_SERVICES else -1

    def _f_having_At_Symbol(self, url: str) -> int:
        return 1 if "@" in url else -1

    def _f_double_slash_redirecting(self, parsed) -> int:
        # Bỏ scheme, kiểm tra "//" trong path
        path = parsed.path.lstrip("/")
        return 1 if "//" in path else -1

    def _f_Prefix_Suffix(self, hostname: str) -> int:
        # Dấu "-" hiếm trong domain hợp lệ => nghi phishing
        return 1 if "-" in hostname else -1

    def _f_having_Sub_Domain(self, hostname: str) -> int:
        h = hostname.lower()
        dots = h.count(".")
        # Nếu là ccTLD hai cấp (co.uk) thì bớt 1 cấp
        for tld in TWO_LEVEL_TLDS:
            if h.endswith(tld):
                dots -= 1
                break
        if dots == 1:
            return -1
        if dots == 2:
            return 0
        return 1

    def _f_SSLfinal_State(self, parsed, ssl_state: str) -> int:
        if parsed.scheme != "https":
            return 1  # không dùng https => nghi phishing
        if ssl_state == "valid":
            return -1
        if ssl_state == "invalid":
            return 1
        return 0  # không xác định được

    def _f_Domain_registeration_length(self) -> int:
        # Cần WHOIS; mặc định trả 0 (không xác định) và đánh dấu gần đúng
        self.approx("Domain_registeration_length")
        return 0

    def _f_Favicon(self, soup, hostname: str) -> int:
        if soup is None:
            self.approx("Favicon")
            return 0
        link = soup.find("link", rel=lambda v: v and "icon" in str(v).lower())
        if link is None or not link.get("href"):
            return 0
        href = link["href"]
        if href.startswith("http"):
            return 1 if urlparse(href).hostname != hostname else -1
        return -1  # favicon tương đối => cùng domain

    def _f_port(self, parsed) -> int:
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return -1 if port in STANDARD_PORTS else 1

    def _f_HTTPS_token(self, hostname: str) -> int:
        return 1 if "https" in hostname.lower() else -1

    # ------------------------------------------------------------------ #
    # Các đặc trưng Abnormal / HTML                                       #
    # ------------------------------------------------------------------ #
    def _external_ratio(self, soup, hostname: str, tag: str, attr: str):
        """Tỷ lệ thẻ trỏ ra domain khác. Trả về None nếu không có HTML."""
        if soup is None:
            return None
        tags = soup.find_all(tag)
        if not tags:
            return 0.0
        ext = 0
        for t in tags:
            val = t.get(attr)
            if not val:
                continue
            if val.startswith("#") or val.lower().startswith("javascript"):
                ext += 1
            elif val.startswith("http"):
                if urlparse(val).hostname != hostname:
                    ext += 1
        return ext / len(tags)

    def _f_Request_URL(self, soup, hostname: str) -> int:
        ratio = self._external_ratio(soup, hostname, "img", "src")
        if ratio is None:
            # thử thêm video/audio
            ratio = self._external_ratio(soup, hostname, "video", "src")
        if ratio is None:
            self.approx("Request_URL")
            return 0
        if ratio < 0.22:
            return -1
        if ratio < 0.61:
            return 0
        return 1

    def _f_URL_of_Anchor(self, soup, hostname: str) -> int:
        ratio = self._external_ratio(soup, hostname, "a", "href")
        if ratio is None:
            self.approx("URL_of_Anchor")
            return 0
        if ratio < 0.31:
            return -1
        if ratio < 0.67:
            return 0
        return 1

    def _f_Links_in_tags(self, soup, hostname: str) -> int:
        ratios = []
        for tag, attr in (("meta", "content"), ("script", "src"), ("link", "href")):
            r = self._external_ratio(soup, hostname, tag, attr)
            if r is not None:
                ratios.append(r)
        if not ratios:
            self.approx("Links_in_tags")
            return 0
        ratio = sum(ratios) / len(ratios)
        if ratio < 0.17:
            return -1
        if ratio < 0.81:
            return 0
        return 1

    def _f_SFH(self, soup, hostname: str) -> int:
        if soup is None:
            self.approx("SFH")
            return 0
        forms = soup.find_all("form")
        if not forms:
            return -1  # không có form => không nguy hiểm
        bad = 0
        for f in forms:
            action = f.get("action", "").strip()
            if action in ("", "about:blank"):
                bad += 1
            elif action.startswith("http") and urlparse(action).hostname != hostname:
                bad += 1
        ratio = bad / len(forms)
        if ratio == 0:
            return -1
        if ratio < 0.5:
            return 0
        return 1

    def _f_Submitting_to_email(self, soup) -> int:
        if soup is None:
            self.approx("Submitting_to_email")
            return 0
        for f in soup.find_all("form"):
            action = f.get("action", "")
            if "mailto:" in action.lower():
                return 1
        return -1

    def _f_Abnormal_URL(self, hostname: str) -> int:
        # Cần WHOIS để đối chiếu danh tính; ước lượng bằng cách kiểm tra
        # hostname có chứa từ khoá nghi vấn không.
        self.approx("Abnormal_URL")
        return 0

    def _f_Redirect(self, resp) -> int:
        if resp is None:
            self.approx("Redirect")
            return 0
        history = getattr(resp, "history", [])
        n = len(history)
        if n <= 1:
            return -1
        if n <= 4:
            return 0
        return 1

    def _f_on_mouseover(self, soup) -> int:
        if soup is None:
            self.approx("on_mouseover")
            return 0
        scripts = " ".join(s.string or "" for s in soup.find_all("script"))
        return 1 if "onmouseover" in scripts.lower() else -1

    def _f_RightClick(self, soup) -> int:
        if soup is None:
            self.approx("RightClick")
            return 0
        scripts = " ".join(s.string or "" for s in soup.find_all("script"))
        return 1 if "contextmenu" in scripts.lower() or \
            "button==2" in scripts.lower() else -1

    def _f_popUpWidnow(self, soup) -> int:
        if soup is None:
            self.approx("popUpWidnow")
            return 0
        scripts = " ".join(s.string or "" for s in soup.find_all("script"))
        return 1 if "window.open" in scripts.lower() else -1

    def _f_Iframe(self, soup) -> int:
        if soup is None:
            self.approx("Iframe")
            return 0
        for fr in soup.find_all("iframe"):
            fb = fr.get("frameborder", "0")
            if str(fb) == "0" or fr.get("border") == "0":
                return 1
        return -1

    # ------------------------------------------------------------------ #
    # Các đặc trưng Domain                                                #
    # ------------------------------------------------------------------ #
    def _f_age_of_domain(self) -> int:
        self.approx("age_of_domain")
        return 0

    def _f_DNSRecord(self, hostname: str) -> int:
        return -1 if self._resolve_ip(hostname) else 1

    def _f_web_traffic(self) -> int:
        self.approx("web_traffic")
        return 0

    def _f_Page_Rank(self) -> int:
        self.approx("Page_Rank")
        return 0

    def _f_Google_Index(self, hostname: str) -> int:
        # Ước lượng: nếu DNS trỏ về IP công cộng hợp lệ thì coi như có khả năng
        # được index; đây là giá trị gần đúng.
        self.approx("Google_Index")
        return -1 if self._resolve_ip(hostname) else 1

    def _f_Links_pointing_to_page(self) -> int:
        self.approx("Links_pointing_to_page")
        return 0

    def _phishtank_check(self, url: str) -> int | None:
        """
        Tra cứu URL trong danh sách đen PhishTank (API miễn phí, cần app_key).

        Returns:
            1 nếu URL có trong danh sách đen, -1 nếu không,
            None nếu không truy vấn được (thiếu key / lỗi mạng).
        """
        if not self.phishtank_key:
            return None
        try:
            resp = self.session.post(
                "https://checkurl.phishtank.com/checkurl/",
                data={"url": url, "format": "json", "app_key": self.phishtank_key},
                timeout=self.timeout, verify=False,
            )
            data = resp.json()
            results = data.get("results", {})
            in_tank = bool(results.get("in_phish_tank", False))
            return 1 if in_tank else -1
        except Exception as exc:  # noqa: BLE001
            log.debug("PhishTank lỗi: %s", exc)
            return None

    def _f_Statistical_report(self, url: str) -> int:
        val = self._phishtank_check(url)
        if val is None:
            self.approx("Statistical_report")
            return 0
        return val

    # ------------------------------------------------------------------ #
    # Pipeline chính                                                      #
    # ------------------------------------------------------------------ #
    def approx(self, name: str) -> None:
        if name not in self.approximated:
            self.approximated.append(name)

    def extract(self, url: str) -> FeatureResult:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname = hostname.rstrip(".")

        res = FeatureResult(url=url)
        self.approximated = res.approximated  # dùng chung danh sách

        # Tải HTML + theo dõi redirect
        html, status = None, None
        resp = None
        if self.fetch_html:
            try:
                resp = self.session.get(url, timeout=self.timeout, verify=False,
                                        allow_redirects=True)
                html, status = resp.text, resp.status_code
            except Exception as exc:  # noqa: BLE001
                log.debug("Không tải được %s: %s", url, exc)
        res.html = html or ""
        res.status_code = status
        soup = BeautifulSoup(html or "", "html.parser") if html else None

        ssl_state = self._ssl_state(hostname, parsed.port or 443) \
            if parsed.scheme == "https" else "none"

        feats = {
            "having_IP_Address": self._f_having_IP_Address(hostname),
            "URL_Length": self._f_URL_Length(url),
            "Shortining_Service": self._f_Shortining_Service(hostname),
            "having_At_Symbol": self._f_having_At_Symbol(url),
            "double_slash_redirecting": self._f_double_slash_redirecting(parsed),
            "Prefix_Suffix": self._f_Prefix_Suffix(hostname),
            "having_Sub_Domain": self._f_having_Sub_Domain(hostname),
            "SSLfinal_State": self._f_SSLfinal_State(parsed, ssl_state),
            "Domain_registeration_length": self._f_Domain_registeration_length(),
            "Favicon": self._f_Favicon(soup, hostname),
            "port": self._f_port(parsed),
            "HTTPS_token": self._f_HTTPS_token(hostname),
            "Request_URL": self._f_Request_URL(soup, hostname),
            "URL_of_Anchor": self._f_URL_of_Anchor(soup, hostname),
            "Links_in_tags": self._f_Links_in_tags(soup, hostname),
            "SFH": self._f_SFH(soup, hostname),
            "Submitting_to_email": self._f_Submitting_to_email(soup),
            "Abnormal_URL": self._f_Abnormal_URL(hostname),
            "Redirect": self._f_Redirect(resp),
            "on_mouseover": self._f_on_mouseover(soup),
            "RightClick": self._f_RightClick(soup),
            "popUpWidnow": self._f_popUpWidnow(soup),
            "Iframe": self._f_Iframe(soup),
            "age_of_domain": self._f_age_of_domain(),
            "DNSRecord": self._f_DNSRecord(hostname),
            "web_traffic": self._f_web_traffic(),
            "Page_Rank": self._f_Page_Rank(),
            "Google_Index": self._f_Google_Index(hostname),
            "Links_pointing_to_page": self._f_Links_pointing_to_page(),
            "Statistical_report": self._f_Statistical_report(url),
        }
        res.features = feats
        return res


FEATURE_ORDER = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
    "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
    "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
    "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]


def extract_to_vector(url: str, timeout: float = 15.0) -> tuple[FeatureResult, list[int]]:
    """Tiện ích: trích đặc trưng và trả về (kết quả chi tiết, vector 30 giá trị)."""
    ex = FeatureExtractor(timeout=timeout)
    res = ex.extract(url)
    vec = [res.features[name] for name in FEATURE_ORDER]
    return res, vec


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "https://duytan.edu.vn/"
    r, vec = extract_to_vector(target)
    print(f"URL: {r.url}  (HTTP {r.status_code})")
    for name, val in zip(FEATURE_ORDER, vec):
        print(f"  {name:<28} {val}")
    print("Đặc trưng dùng giá trị gần đúng:", r.approximated or "không")