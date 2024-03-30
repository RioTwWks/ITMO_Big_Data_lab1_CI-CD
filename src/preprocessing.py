import unicodedata

import re


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)

    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text
