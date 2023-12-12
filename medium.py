import requests
from bs4 import BeautifulSoup


def get_text(url):
    response = requests.get(url)
    bs = BeautifulSoup(response.text)
    ps = bs.find_all("p", {"class": "pw-post-body-paragraph"})
    text = ""
    for p in ps:
        ptext = p.get_text()
        if "_" in ptext and ptext.replace("_", "").strip() == "":
            break
        if all(not char.isalpha() for char in ptext):
            continue
        text += ptext.strip() + " "
    return text
