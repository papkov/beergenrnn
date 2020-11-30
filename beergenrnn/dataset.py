import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from xml.etree.ElementTree import Element

import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange


def tag_to_dict_list(
    root: Element, tag: str, convert: Optional[Dict[str, Callable]] = None
):
    tags = []
    for tag in root.findall(tag)[0]:
        elem_dict = {child.tag.lower(): child.text for child in tag}
        if convert is not None:
            elem_dict = {k: convert[k](v) for k, v in elem_dict.items() if k in convert}
        # if 'yield' in elem_dict:
        #     elem_dict['result'] = elem_dict.pop('yield')

        tags.append(elem_dict)
    return tags


def time_to_float_hours(time_str: str, max_time: float = 2.0):
    """
    Converts time from minutes to hours, clips dry hop time to max_time
    :param time_str:
    :param max_time:
    :return:
    """
    return min(float(time_str) / 60, max_time)


@dataclass
class RecipeDataset(Dataset):
    path: Union[Path, str] = "../data/brewdog"
    max_len: int = 16
    dicts: Optional[Dict[str, Dict[str, int]]] = None
    normalize_prop: str = "batch_size"

    def __post_init__(self):
        self.path = Path(self.path)
        self.recipe_paths = list(self.path.glob("*.xml"))
        if self.dicts is None:
            self.dicts = self._build_dicts()

    def __len__(self):
        return len(self.recipe_paths)

    def _get_dict(
        self, i: int
    ) -> Dict[str, Union[np.ndarray, List[Union[int, float, str]]]]:
        """
        Get not encoded BeerXML in flattened form
        :param i:
        :return:
        """
        # read BeerXML
        root = ET.parse(self.recipe_paths[i]).getroot()
        properties = {
            k: float(root[0].findall(k.upper())[0].text)
            for k in ["batch_size", "boil_size", "boil_time", "efficiency"]
        }

        item = dict()
        ingredients_to_read = {
            "hops": {
                "name": str,
                "amount": lambda x: float(x)
                / properties[self.normalize_prop]
                * 1000,  # to g/l
                "time": time_to_float_hours,
                "use": str,
            },
            "fermentables": {
                "name": str,
                "amount": lambda x: float(x) / properties[self.normalize_prop],
            },  # to kg/l
            "yeasts": {"name": str},
        }

        for ingredient, convert in ingredients_to_read.items():
            elements = tag_to_dict_list(root[0], ingredient.upper(), convert=convert)
            item[ingredient] = elements

        try:
            # Flatten ingredients (hop: name: [] into hop_name: [])
            item = {
                f"{kind}_{prop}": [ingredient[prop] for ingredient in ingredient_list]
                for kind, ingredient_list in item.items()
                for prop in ingredient_list[0].keys()
            }
        except IndexError:
            print(i, self.recipe_paths[i])
            print(item)
            raise

        # Add style info
        item["style_name"] = [root[0].findall("STYLE")[0][1].text]
        return item

    def __getitem__(self, i: int):
        item = self._get_dict(i)
        for k, value_list in item.items():
            if k in self.dicts:
                item[k] = [self.dicts[k][v] for v in value_list]

            # Pad hops and fermentables
            ingredient, prop = k.split("_")
            if ingredient in ("hops", "fermentables"):
                if prop not in ("name", "use"):
                    item[k] = [0] + item[k] + [0]
                else:
                    item[k] = (
                        [self.dicts[k]["<start>"]] + item[k] + [self.dicts[k]["<end>"]]
                    )
                item[k] = np.pad(
                    item[k][: self.max_len],
                    (0, max(0, self.max_len - len(item[k]))),
                    constant_values=0,
                )
            else:
                item[k] = np.array(item[k])

        return item

    def _build_dicts(
        self, ends: Tuple[str] = ("name", "use", "style")
    ) -> Dict[str, Dict[str, int]]:
        """
        Builds dicts for numerical encoding of all the categorical values in the data
        :param ends: ends of field names to encode
        :return: dict of dicts variable -> name -> encoding
        """
        items = [self._get_dict(i) for i in range(len(self))]
        dicts = {k: v for k, v in items[0].items() if k.split("_")[-1] in ends}
        for item in items[1:]:
            for key, value_list in item.items():
                if key in dicts:
                    dicts[key] += value_list

        for key, value_list in dicts.items():
            unique = sorted(set(value_list))
            dicts[key] = {v: i + 3 for i, v in enumerate(unique)}
            if key != "style_name":
                dicts[key].update({"<pad>": 0, "<start>": 1, "<end>": 2})

        return dicts
