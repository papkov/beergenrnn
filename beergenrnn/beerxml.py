from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple


@dataclass
class Fermentable:
    name: str = ""
    version: int = 0
    type: str = ""
    amount: float = 0.0
    result: float = 0.0
    color: int = 1

    def __post_init__(self):
        self.amount = float(self.amount)


@dataclass
class Hop:
    name: str
    version: int
    origin: str
    alpha: float
    amount: float
    use: str
    time: int
    type: str
    form: str
    beta: float

    def __post_init__(self):
        self.amount = float(self.amount)
        self.alpha = float(self.alpha)
        self.time = int(self.time)


@dataclass
class Yeast:
    name: str = ""
    version: int = 0
    type: str = ""
    form: str = ""
    amount: float = 0

    def __post_init__(self):
        self.amount = float(self.amount)


@dataclass
class MashStep:
    name: str
    version: int
    type: str
    step_temp: int
    step_time: int


@dataclass
class Mash:
    steps: List[MashStep]
    version: int = 0
    grain_temp: int = 20


@dataclass
class Style:
    name: str = ""
    category: str = ""


@dataclass
class Recipe:
    style: Style
    fermentables: List[Fermentable]
    hops: List[Hop]
    yeasts: List[Yeast]
    mash: Mash = None
    miscs: List[Any] = None
    name: str = ""
    version: int = 0
    og: float = 1
    fg: float = 1
    type: str = ""
    batch_size: int = 20
    boil_size: int = 25
    boil_time: int = 60
    efficiency: float = 75
    primary_temp: int = 25
