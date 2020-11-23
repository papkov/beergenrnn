import os
from typing import List
from urllib.parse import quote
from urllib.request import Request, urlopen, urlretrieve

from bs4 import BeautifulSoup
from tqdm.auto import tqdm, trange

root = "https://brewdogrecipes.com/recipes/"
page = root + "p"


def get_brewdog_recipe_links(n_pages: int = 14) -> List[str]:
    recipe_links = []
    for i in trange(1, n_pages + 1):
        html = urlopen(Request(f"{page}{i}"))
        soup = BeautifulSoup(html, "lxml")

        for link in soup.findAll("a"):
            link = link.get("href")
            if link.startswith(root) and not link.startswith(page):
                recipe_links.append(link.replace("/recipes/", "/beerxml/") + ".xml")

    return recipe_links


def download_recipes(recipe_links, target_dir: str = "../data/brewdog"):
    for link in tqdm(recipe_links):
        name = link.split("/")[-1]
        path = os.path.join(target_dir, name)
        try:
            urlretrieve(quote(link, encoding="utf-8", safe=":/"), path)
        except KeyboardInterrupt:
            break


#         except:
#             print(name)
