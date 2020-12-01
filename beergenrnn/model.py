import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


def each_value_to_list(d: Dict[str, Any]):
    out = d.copy()
    for k, v in d.items():
        out[k] = [v]
    return out


class Meter:
    def __init__(self, metrics: Optional[Dict[str, Union[float, List[float]]]] = None):
        self.metrics = metrics if metrics is None else each_value_to_list(metrics)

    def __add__(self, other):
        if self.metrics is None:
            self.metrics = each_value_to_list(other)
            return self

        if isinstance(other, Meter):
            other = other.metrics

        if isinstance(other, dict):
            for k, v in other.items():
                if k in self.metrics:
                    if isinstance(v, list):
                        self.metrics[k].extend(v)
                    else:
                        self.metrics[k].append(v)
                else:
                    self.metrics[k] = v if isinstance(v, list) else [v]
        else:
            raise TypeError

        return self

    def mean(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def last(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self.metrics.items()}


class Heads(nn.Module):
    def __init__(self, in_features: int, output_sizes: List[int], dropout: float = 0.5):
        super(Heads, self).__init__()
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, s))
                for s in output_sizes
            ]
        )

    def forward(self, x: T) -> List[T]:
        out = [head(x).transpose(1, 2) for head in self.fcs]
        return out


class Model(nn.Module):
    def __init__(
        self,
        output_sizes: List[int],
        hidden_dim: int = 16,
        n_layers: int = 1,
        embedding_dim: int = 16,
        num_embeddings: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.heads = Heads(hidden_dim, output_sizes)

    def forward(self, x, hidden: Optional[T] = None) -> Tuple[T, T, T, T]:
        # if hidden is None:
        #     hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # Get RNN output
        emb = self.emb(x)
        if hidden is None:
            out, (hn, cn) = self.rnn(emb)
        else:
            out, (hn, cn) = self.rnn(emb, hidden)
        # print(out.shape, hn.shape, cn.shape)
        out = self.heads(out)
        return out, hn, cn, emb


class RecipeModel(nn.Module):
    def __init__(
        self,
        n_fermentables: int,
        n_hops: int,
        n_yeasts: int,
        n_hops_uses: int,
        n_styles: int,
        hidden_dim: int = 128,
        embedding_dim: int = 32,
        n_layers: int = 1,
    ):

        super(RecipeModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.style_model = nn.Sequential(
            nn.Embedding(n_styles, embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
        )

        # Predict fermentable name and amount
        self.fermentables_model = Model(
            num_embeddings=n_fermentables,
            embedding_dim=embedding_dim,
            output_sizes=[n_fermentables, 1],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        # Predict hop name, hop use (boil, dry hop, etc.), time, and amount
        self.hops_model = Model(
            num_embeddings=n_hops,
            embedding_dim=embedding_dim,
            output_sizes=[n_hops, n_hops_uses, 1, 1],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        # Predict yeast name and amount
        self.yeast_model = Model(
            num_embeddings=n_yeasts,
            embedding_dim=embedding_dim,
            output_sizes=[n_yeasts],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        # self.yeast_model = Heads(
        #     in_features=hidden_dim,
        #     output_sizes=[n_yeasts],
        # )

    def forward(self, x: Dict[str, T]) -> Tuple[List[T], List[T], List[T], T]:

        # Get hidden state from style embeddings
        style = self.style_model(x["style_name"].long()).transpose(0, 1)

        # Collect outputs from all the models
        out_fermentables, hn, cn, emb_fermentables = self.fermentables_model(
            x["fermentables_name"].long(), (style, torch.zeros_like(style))
        )
        # print([o.shape for o in out_fermentables], hn.shape, cn.shape, style.shape)
        out_hops, hn, cn, emb_hops = self.hops_model(
            x["hops_name"].long(), (hn + style, cn)
        )
        out_yeasts, hn, cn, emb_yeasts = self.yeast_model(
            x["yeasts_name"].long(), (hn + style, cn)
        )

        return out_fermentables, out_hops, out_yeasts, hn

    def generate(
        self,
        style: Optional[int] = None,
        max_steps: int = 16,
        deterministic: bool = True,
        device="cpu",
    ):
        self.eval()
        if style is None:
            hn = torch.zeros(self.n_layers, 1, self.hidden_dim).to(device)
        else:
            hn = self.style_model(torch.tensor(style).reshape(1, 1).to(device))
        cn = torch.zeros_like(hn).to(device)

        # TODO take actual start and end token
        start = torch.tensor(1).reshape(1, 1).to(device)
        end = [torch.tensor(2).reshape(1, 1), torch.tensor(0).reshape(1, 1)]

        output = {
            "fermentables_name": [],
            "fermentables_amount": [],
            "hops_name": [],
            "hops_amount": [],
            "hops_time": [],
            "hops_use": [],
            "yeasts_name": [],
        }

        out = start.clone()
        step = 0
        while step < max_steps:
            out, hn, cn, emb = self.fermentables_model(out, (hn, cn))
            cat_prob = torch.softmax(out[0].squeeze(), dim=-1).detach().cpu()
            if deterministic:
                cat = torch.argmax(cat_prob)
            else:
                cat = torch.tensor(
                    np.random.choice(cat_prob.size()[0], p=cat_prob.numpy())
                )
            if cat.reshape(1, 1) in end:
                break
            output["fermentables_name"].append(cat.numpy())
            output["fermentables_amount"].append(
                torch.clamp(out[1], min=0).detach().cpu().numpy().squeeze()
            )
            step += 1
            out = cat.reshape(1, 1).to(device)

        out = start.clone()
        step = 0
        while step < max_steps:
            out, hn, cn, emb = self.hops_model(out, (hn, cn))
            cat_prob = torch.softmax(out[0].squeeze(), dim=-1).detach().cpu()
            if deterministic:
                cat = torch.argmax(cat_prob)
            else:
                cat = torch.tensor(
                    np.random.choice(cat_prob.size(0), p=cat_prob.numpy())
                )
            if cat.reshape(1, 1) in end:
                break
            output["hops_name"].append(cat.numpy())
            output["hops_use"].append(
                torch.argmax(out[1].squeeze()).detach().cpu().numpy()
            )
            output["hops_time"].append(
                torch.clamp(out[2], min=0).detach().cpu().numpy().squeeze()
            )
            output["hops_amount"].append(
                torch.clamp(out[3], min=0).detach().cpu().numpy().squeeze()
            )
            step += 1
            out = cat.reshape(1, 1).to(device)

        out = start.clone()
        step = 0
        while step < max_steps:
            out, hn, cn, emb = self.yeast_model(out, (hn, cn))
            cat = torch.argmax(out[0].squeeze()).detach().cpu()
            if cat.reshape(1, 1) in end:
                break
            output["yeasts_name"].append(cat.numpy())
            step += 1
            out = cat.reshape(1, 1).to(device)

        # out = self.yeast_model(hn.transpose(0, 1))
        # output["yeasts_name"].append(torch.argmax(out[0].squeeze()).detach().numpy())

        output = {k: np.array(v) for k, v in output.items()}
        return output


class Trainer(nn.Module):
    def __init__(self, model: RecipeModel, name: str = "model", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.name = name
        self.model = model.to(device)
        self.criterion_clf = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, x: Dict[str, T]) -> Tuple[List[T], List[T], List[T], T]:
        return self.model(x)

    def one_epoch(self, loader: DataLoader, epoch: int = 0):
        self.model.train()
        iterator = tqdm(loader, leave=False, position=0, desc=f"ep. {epoch:04d}")
        meter = Meter()
        for i, batch in enumerate(iterator):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            out_mash, out_hops, out_yeasts, hidden = self(batch)
            loss, loss_dict = self.loss(batch, out_mash, out_hops, out_yeasts)
            meter += loss_dict
            iterator.set_postfix(**meter.mean())
            loss.backward()

            self.optimizer.step()

        self.model.eval()
        return meter.mean()

    def loss(
        self,
        batch: Dict[str, T],
        out_fermentables: List[T],
        out_hops: List[T],
        out_yeasts: List[T],
    ) -> Tuple[T, Dict[str, T]]:

        # Classification losses
        loss_clf_mash = self.criterion_clf(
            out_fermentables[0][..., :-1], batch["fermentables_name"][:, 1:].long()
        )
        loss_clf_hops = self.criterion_clf(
            out_hops[0][..., :-1], batch["hops_name"][:, 1:].long()
        )
        loss_clf_hops_uses = self.criterion_clf(
            out_hops[1][..., :-1], batch["hops_use"][:, 1:].long()
        )
        loss_clf_yeasts = self.criterion_clf(out_yeasts[0], batch["yeasts_name"].long())

        # Regression losses
        loss_reg_mash_amt = self.criterion_reg(
            out_fermentables[-1][..., :-1].squeeze(),
            batch["fermentables_amount"][:, 1:].float(),
        )
        loss_reg_hops_amt = self.criterion_reg(
            out_hops[-1][..., :-1].squeeze(), batch["hops_amount"][:, 1:].float()
        )
        loss_reg_hops_time = self.criterion_reg(
            out_hops[-2][..., :-1].squeeze(), batch["hops_time"][:, 1:].float()
        )

        # Combine all
        loss_clf = loss_clf_mash + loss_clf_hops + loss_clf_hops_uses + loss_clf_yeasts
        loss_reg = (
            loss_reg_mash_amt * 10
            + loss_reg_hops_amt  # TODO tune coefficient for hops
            + loss_reg_hops_time
        )

        # TODO coefficient
        loss = loss_clf + loss_reg

        loss_dict = {
            "loss_clf_mash": loss_clf_mash.item(),
            "loss_clf_hops": loss_clf_hops.item(),
            "loss_clf_hops_uses": loss_clf_yeasts.item(),
            "loss_clf_yeasts": loss_clf_yeasts.item(),
            "loss_reg_mash_amt": loss_reg_mash_amt.item(),
            "loss_reg_hops_amt": loss_reg_hops_amt.item(),
            "loss_reg_hops_time": loss_reg_hops_time.item(),
            "loss": loss.item(),
        }

        return loss, loss_dict

    def fit(
        self,
        n_epochs: int,
        loader_train: DataLoader,
        loader_valid: Optional[DataLoader] = None,
    ):
        meter = Meter()
        iterator = trange(n_epochs)
        for epoch in iterator:
            meter += self.one_epoch(loader_train, epoch=epoch)
            torch.save(
                self.model.state_dict(),
                os.path.join("../checkpoints", self.name) + ".pth",
            )
            iterator.set_postfix(**meter.last())

        return meter.metrics
