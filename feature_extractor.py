import json
import os
import re
from urllib.parse import urlsplit, urljoin

import requests


FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatio",
    "NoOfEquals",
    "NoOfQMark",
    "NoOfAmpersand",
    "NoOfOtherSpecialChars",
    "SpacialCharRatio",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]


THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        THIS_DIR,
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model"
)

TLD_PROB_PATH = os.path.join(
    MODEL_DIR,
    "tld_prob.json"
)

CHAR_PROB_PATH = os.path.join(
    MODEL_DIR,
    "char_prob.json"
)


REQUEST_TIMEOUT = 8


session = requests.Session()

session.headers.update({
    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/120.0 Safari/537.36"
})


def _load_json(path):

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            dict
        ):
            return data

    except Exception:
        pass

    return {}


_TLD_PROB = _load_json(
    TLD_PROB_PATH
)

_CHAR_PROB = _load_json(
    CHAR_PROB_PATH
)


def canonicalize_url(url):

    url = str(
        url or ""
    ).strip().lower()

    if not url:
        return ""

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url
    ):
        url = (
            "https://"
            + url
        )

    try:

        parts = urlsplit(
            url
        )

        host = (
            parts.hostname
            or ""
        ).lower()

        if host.startswith(
            "www."
        ):
            host = host[4:]

        path = (
            parts.path
            or ""
        )

        query = (
            parts.query
            or ""
        )

        return (
            host
            + path
            + (
                "?"
                + query
                if query
                else ""
            )
        )

    except Exception:

        return url


def _is_ip(host):

    parts = str(
        host or ""
    ).split(".")

    if len(parts) != 4:
        return 0

    try:

        return int(
            all(
                0 <= int(part) <= 255
                for part in parts
            )
        )

    except Exception:

        return 0


def _get_tld(host):

    host = str(
        host or ""
    ).strip(
        "."
    ).lower()

    if "." not in host:
        return ""

    return host.rsplit(
        ".",
        1
    )[-1]


def _safe_fetch(url):

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )

        return (
            response.text or "",
            response.url,
            len(
                response.history
            ),
            True
        )

    except Exception:

        return (
            "",
            url,
            0,
            False
        )


def _title_match_score(
    title,
    text
):

    title_words = [
        word.strip().lower()
        for word in str(
            title or ""
        ).split()
        if word.strip()
    ]

    text = re.sub(
        r"[^a-zA-Z0-9]",
        "",
        str(
            text or ""
        ).lower()
    )

    if not title_words or not text:
        return 0.0

    score = 0.0

    base_score = (
        100.0
        / len(title_words)
    )

    for word in title_words:

        word_clean = re.sub(
            r"[^a-zA-Z0-9]",
            "",
            word
        )

        if not word_clean:
            continue

        if word_clean in text:

            score += (
                base_score
                * len(word_clean)
                / max(
                    1,
                    len(word)
                )
            )

            text = text.replace(
                word_clean,
                ""
            )

    return min(
        100.0,
        score
    )


def _url_char_probability(url):

    if not url:
        return 0.0

    values = []

    for char in str(
        url
    ).lower():

        if char in _CHAR_PROB:

            try:

                values.append(
                    float(
                        _CHAR_PROB[
                            char
                        ]
                    )
                )

            except Exception:
                pass

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def _continuation_rate(url):

    if not url:
        return 0.0

    url = str(
        url
    )

    if len(url) <= 1:
        return 1.0

    same_class = 0

    previous = None

    for char in url:

        if char.isalpha():
            current = "letter"

        elif char.isdigit():
            current = "digit"

        else:
            current = "special"

        if previous == current:
            same_class += 1

        previous = current

    return (
        same_class
        / max(
            1,
            len(url) - 1
        )
    )


def extract_features_from_url(
    url
):

    url = str(
        url or ""
    ).strip()

    if not url:
        return (
            [0.0] * 50,
            False,
            False
        )

    normalized = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        normalized
    ):
        normalized = (
            "https://"
            + normalized
        )

    try:

        parts = urlsplit(
            normalized
        )

    except Exception:

        return (
            [0.0] * 50,
            False,
            False
        )

    host = (
        parts.hostname
        or ""
    ).lower()

    scheme = (
        parts.scheme
        or ""
    ).lower()

    if not host:

        return (
            [0.0] * 50,
            False,
            False
        )

    path = (
        parts.path
        or ""
    )

    query = (
        parts.query
        or ""
    )

    tld = _get_tld(
        host
    )

    features = {
        name: 0.0
        for name in FEATURE_NAMES
    }

    # -------------------------------------------------
    # URL FEATURES
    # -------------------------------------------------

    features[
        "URLLength"
    ] = float(
        len(normalized)
    )

    features[
        "DomainLength"
    ] = float(
        len(host)
    )

    features[
        "IsDomainIP"
    ] = float(
        _is_ip(host)
    )

    features[
        "TLDLength"
    ] = float(
        len(tld)
    )

    features[
        "TLDLegitimateProb"
    ] = float(
        _TLD_PROB.get(
            tld,
            0.5
        )
    )

    features[
        "URLCharProb"
    ] = float(
        _url_char_probability(
            normalized
        )
    )

    features[
        "CharContinuationRate"
    ] = float(
        _continuation_rate(
            normalized
        )
    )

    features[
        "NoOfSubDomain"
    ] = float(
        max(
            0,
            len(
                host.split(".")
            ) - 2
        )
    )

    features[
        "IsHTTPS"
    ] = float(
        scheme == "https"
    )

    letters = sum(
        char.isalpha()
        for char in normalized
    )

    digits = sum(
        char.isdigit()
        for char in normalized
    )

    length = max(
        1,
        len(normalized)
    )

    features[
        "NoOfLettersInURL"
    ] = float(
        letters
    )

    features[
        "LetterRatioInURL"
    ] = (
        letters
        / length
    )

    features[
        "NoOfDegitsInURL"
    ] = float(
        digits
    )

    features[
        "DegitRatio"
    ] = (
        digits
        / length
    )

    features[
        "NoOfEquals"
    ] = float(
        normalized.count("=")
    )

    features[
        "NoOfQMark"
    ] = float(
        normalized.count("?")
    )

    features[
        "NoOfAmpersand"
    ] = float(
        normalized.count("&")
    )

    other_special = sum(
        1
        for char in normalized
        if (
            not char.isalnum()
            and char not in
            ":/.?=&-_"
        )
    )

    features[
        "NoOfOtherSpecialChars"
    ] = float(
        other_special
    )

    features[
        "SpacialCharRatio"
    ] = (
        other_special
        / length
    )

    obfuscated = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            normalized
        )
    )

    features[
        "NoOfObfuscatedChar"
    ] = float(
        obfuscated
    )

    features[
        "HasObfuscation"
    ] = float(
        obfuscated > 0
    )

    features[
        "ObfuscationRatio"
    ] = (
        obfuscated
        / length
    )

    # -------------------------------------------------
    # WEB PAGE
    # -------------------------------------------------

    html, final_url, redirects, fetched = (
        _safe_fetch(
            normalized
        )
    )

    if not fetched:

        return (
            [
                float(
                    features[name]
                )
                for name in FEATURE_NAMES
            ],
            False,
            False
        )

    try:

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        lines = html.splitlines()

        features[
            "LineOfCode"
        ] = float(
            len(lines)
        )

        features[
            "LargestLineLength"
        ] = float(
            max(
                (
                    len(line)
                    for line in lines
                ),
                default=0
            )
        )

        title_tag = soup.find(
            "title"
        )

        title = (
            title_tag.get_text(
                " ",
                strip=True
            )
            if title_tag
            else ""
        )

        features[
            "HasTitle"
        ] = float(
            bool(title)
        )

        # -------------------------------------------------
        # TITLE MATCH
        # -------------------------------------------------

        domain_text = host

        if domain_text.startswith(
            "www."
        ):
            domain_text = (
                domain_text[4:]
            )

        domain_text = (
            domain_text
            .replace(
                "." + tld,
                ""
            )
        )

        url_text = (
            normalized
            .replace(
                "https://",
                ""
            )
            .replace(
                "http://",
                ""
            )
            .replace(
                "www.",
                ""
            )
            .replace(
                "." + tld,
                ""
            )
            .replace(
                "/",
                ""
            )
        )

        features[
            "DomainTitleMatchScore"
        ] = _title_match_score(
            title,
            domain_text
        )

        features[
            "URLTitleMatchScore"
        ] = _title_match_score(
            title,
            url_text
        )

        # -------------------------------------------------
        # URL SIMILARITY
        # -------------------------------------------------

        features[
            "URLSimilarityIndex"
        ] = (
            100.0
            if (
                canonicalize_url(
                    normalized
                )
                == canonicalize_url(
                    final_url
                )
            )
            else 0.0
        )

        # -------------------------------------------------
        # FAVICON
        # -------------------------------------------------

        features[
            "HasFavicon"
        ] = float(
            bool(
                soup.find(
                    "link",
                    rel=lambda value:
                    value
                    and
                    "icon"
                    in str(
                        value
                    ).lower()
                )
            )
        )

        # -------------------------------------------------
        # ROBOTS
        # -------------------------------------------------

        try:

            robots_url = urljoin(
                normalized,
                "/robots.txt"
            )

            robots_response = (
                session.get(
                    robots_url,
                    timeout=4
                )
            )

            features[
                "Robots"
            ] = float(
                robots_response.status_code
                == 200
            )

        except Exception:

            features[
                "Robots"
            ] = 0.0

        # -------------------------------------------------
        # RESPONSIVE
        # -------------------------------------------------

        features[
            "IsResponsive"
        ] = float(
            bool(
                soup.find(
                    "meta",
                    attrs={
                        "name":
                            re.compile(
                                r"^viewport$",
                                re.I
                            )
                    }
                )
            )
        )

        # -------------------------------------------------
        # REDIRECTS
        # -------------------------------------------------

        features[
            "NoOfURLRedirect"
        ] = float(
            redirects
        )

        original_host = host

        final_host = (
            urlsplit(
                final_url
            ).hostname
            or ""
        ).lower()

        features[
            "NoOfSelfRedirect"
        ] = float(
            redirects
            if final_host
            == original_host
            else 0
        )

        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        features[
            "HasDescription"
        ] = float(
            bool(
                soup.find(
                    "meta",
                    attrs={
                        "name":
                            re.compile(
                                r"description",
                                re.I
                            )
                    }
                )
            )
        )

        # -------------------------------------------------
        # POPUPS
        # -------------------------------------------------

        features[
            "NoOfPopup"
        ] = float(
            len(
                re.findall(
                    r"window\.open\s*\(",
                    html,
                    re.I
                )
            )
        )

        # -------------------------------------------------
        # IFRAMES
        # -------------------------------------------------

        features[
            "NoOfiFrame"
        ] = float(
            len(
                soup.find_all(
                    "iframe"
                )
            )
        )

        # -------------------------------------------------
        # FORMS
        # -------------------------------------------------

        external_form = 0

        for form in soup.find_all(
            "form"
        ):

            action = str(
                form.get(
                    "action",
                    ""
                )
                or ""
            ).strip()

            if not action:
                continue

            action_url = urljoin(
                normalized,
                action
            )

            action_host = (
                urlsplit(
                    action_url
                ).hostname
                or ""
            ).lower()

            if (
                action_host
                and
                action_host
                != original_host
            ):

                external_form = 1
                break

        features[
            "HasExternalFormSubmit"
        ] = float(
            external_form
        )

        # -------------------------------------------------
        # SOCIAL NETWORK
        # -------------------------------------------------

        social_domains = [
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com",
            "tiktok.com",
            "pinterest.com",
            "x.com",
        ]

        social = 0

        for anchor in soup.find_all(
            "a",
            href=True
        ):

            href = str(
                anchor.get(
                    "href",
                    ""
                )
            ).lower()

            if any(
                site in href
                for site in social_domains
            ):

                social = 1
                break

        features[
            "HasSocialNet"
        ] = float(
            social
        )

        # -------------------------------------------------
        # SUBMIT / HIDDEN / PASSWORD
        # -------------------------------------------------

        features[
            "HasSubmitButton"
        ] = float(
            bool(
                soup.find(
                    "button",
                    attrs={
                        "type":
                            re.compile(
                                r"^submit$",
                                re.I
                            )
                    }
                )
                or
                soup.find(
                    "input",
                    attrs={
                        "type":
                            re.compile(
                                r"^submit$",
                                re.I
                            )
                    }
                )
            )
        )

        features[
            "HasHiddenFields"
        ] = float(
            bool(
                soup.find(
                    "input",
                    attrs={
                        "type":
                            re.compile(
                                r"^hidden$",
                                re.I
                            )
                    }
                )
            )
        )

        features[
            "HasPasswordField"
        ] = float(
            bool(
                soup.find(
                    "input",
                    attrs={
                        "type":
                            re.compile(
                                r"^password$",
                                re.I
                            )
                    }
                )
            )
        )

        # -------------------------------------------------
        # SECURITY KEYWORDS
        # -------------------------------------------------

        page_text = soup.get_text(
            " ",
            strip=True
        ).lower()

        features[
            "Bank"
        ] = float(
            "bank"
            in page_text
        )

        features[
            "Pay"
        ] = float(
            "pay"
            in page_text
        )

        features[
            "Crypto"
        ] = float(
            "crypto"
            in page_text
        )

        # -------------------------------------------------
        # COPYRIGHT
        # -------------------------------------------------

        features[
            "HasCopyrightInfo"
        ] = float(
            "copyright"
            in page_text
            or "©"
            in html
        )

        # -------------------------------------------------
        # RESOURCES
        # -------------------------------------------------

        features[
            "NoOfImage"
        ] = float(
            len(
                soup.find_all(
                    "img"
                )
            )
        )

        features[
            "NoOfCSS"
        ] = float(
            len(
                soup.find_all(
                    "link",
                    rel=lambda value:
                    value
                    and
                    "stylesheet"
                    in str(
                        value
                    ).lower()
                )
            )
        )

        features[
            "NoOfJS"
        ] = float(
            len(
                soup.find_all(
                    "script"
                )
            )
        )

        # -------------------------------------------------
        # LINKS
        # -------------------------------------------------

        self_ref = 0
        empty_ref = 0
        external_ref = 0

        for anchor in soup.find_all(
            "a",
            href=True
        ):

            href = str(
                anchor.get(
                    "href",
                    ""
                )
                or ""
            ).strip()

            if (
                not href
                or href == "#"
            ):

                empty_ref += 1
                continue

            if href.lower().startswith(
                (
                    "javascript:",
                    "mailto:",
                    "tel:"
                )
            ):

                empty_ref += 1
                continue

            absolute = urljoin(
                normalized,
                href
            )

            link_host = (
                urlsplit(
                    absolute
                ).hostname
                or ""
            ).lower()

            if not link_host:

                empty_ref += 1

            elif link_host == original_host:

                self_ref += 1

            else:

                external_ref += 1

        features[
            "NoOfSelfRef"
        ] = float(
            self_ref
        )

        features[
            "NoOfEmptyRef"
        ] = float(
            empty_ref
        )

        features[
            "NoOfExternalRef"
        ] = float(
            external_ref
        )

        return (
            [
                float(
                    features[name]
                )
                for name in FEATURE_NAMES
            ],
            True,
            True
        )

    except Exception:

        return (
            [
                float(
                    features[name]
                )
                for name in FEATURE_NAMES
            ],
            False,
            True
        )