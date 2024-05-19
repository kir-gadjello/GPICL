#!/usr/bin/env python3

import sys
import os

import json
import uuid
import itertools
import hashlib
import time
import math
import logging
import decimal
import random
import types

from tqdm import tqdm

import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Lambda, Resize, Compose

import torch.optim as optim

logger = logging.getLogger(__name__)

DATA_ROOT = "./basic_datasets"

_VER = "v0.2"

available_experiments = dict(
    replication_small=dict(
        name="GPICL_base_replication_small",
        dataset="smallFashionMNIST",
        unseen_datasets=["smallMNIST"],
        context_len=50,
        bs=128,
        augs=dict(
            perm_count=2**17,
            random_linear_proj=True,
            # 2**25 per support material, but 2**17 per figure 7 Will have to be generated on the fly
            single_perm_relaxation=0.9,
            single_perm_relaxation_multi=False,
            per_batch_relaxed_perm=False,
            relax_perm_only=True,
        ),
        seed=3162541,
        opt=dict(name="Adam", config=dict(lr=0.0002, eps=1e-10)),
        model_hparams=dict(
            model_dim=256,
            depth=4,
            heads=8,
            tweaks=dict(USE_SMALL_EMB=False, USE_POST_LN=False, ROTARY_POS_EMB=True),
        ),
        save_schedule=[
            500,
            *[i for i in range(1000, 10001, 1000)],
            *[i for i in range(10000, 50001, 5000)],
        ],
    ),
    test=dict(
        name="GPICL_test_no_permutation",
        dataset="smallFashionMNIST",
        unseen_datasets=["smallMNIST"],
        context_len=50,
        bs=128,
        augs=dict(
            perm_count=0,
            random_linear_proj=False,
            # 2**25 per support material, but 2**17 per figure 7 Will have to be generated on the fly
            single_perm_relaxation=None,
            single_perm_relaxation_multi=False,
            per_batch_relaxed_perm=False,
            relax_perm_only=False,
        ),
        seed=3162541,
        opt=dict(name="Adam", config=dict(lr=0.0002, eps=1e-10)),
        model_hparams=dict(
            model_dim=256,
            depth=4,
            heads=8,
            tweaks=dict(USE_SMALL_EMB=True, USE_POST_LN=False, ROTARY_POS_EMB=False),
        ),
        save_schedule=[1000],
    ),
)

_exp = "replication_small"
if len(sys.argv) >= 2:
    _exp = sys.argv[1]
    assert sys.argv[1] in available_experiments


spec = available_experiments[_exp]

DEVICE = os.environ.get("DL_DEVICE", "cpu")  # cuda, mps

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"

LOG_HEADER_PATH = "./log.json.txt"
log_header_fh = open(LOG_HEADER_PATH, "a")
log_stream_fh = None

ds_defs = dict(
    smallMNIST=dict(name="MNIST", resize_to=15, to_rgb=False),
    smallFashionMNIST=dict(name="FashionMNIST", resize_to=15, to_rgb=False),
)

_num_labels = dict(MNIST=10, FashionMNIST=10)

ds_norms = {
    "smallMNIST": {
        "train": {"mean": 0.13057130846946088, "std": 0.25209219224403256},
        "eval": {"mean": 0.1324341262398292, "std": 0.2551300750769682},
    },
    "smallFashionMNIST": {
        "train": {"mean": 0.2871933293840227, "std": 0.2841885384127496},
        "eval": {"mean": 0.2880041665494442, "std": 0.28373850803189005},
    },
}

SI = {
    24: "Y",
    21: "Z",
    18: "E",
    15: "P",
    12: "T",
    9: "G",
    6: "M",
    3: "K",
    0: "",
    -3: "m",
    -6: "µ",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
    -21: "z",
    -24: "y",
}


def e(exponent):
    value = str(abs(exponent))
    if exponent < 0:
        return "E-" + value
    elif exponent > 0:
        return "E+" + value
    else:
        return ""


def replace(string, mapping):
    for match, replacement in mapping.items():
        string = string.replace(match, replacement)
    return string


def feng(value, precision=3, prefix=True, prefixes=SI, sep=None):
    """Convert a number to engineering notation."""

    display = decimal.Context(prec=precision)
    value = decimal.Decimal(value).normalize(context=display)
    string = value.to_eng_string()

    if sep is not None:
        string = string.replace("E", sep + "E")

    if prefix:
        prefixes = {e(exponent): prefix for exponent, prefix in prefixes.items()}
        return replace(string, prefixes)
    else:
        return string


def announce_exp(params):
    try:
        d = json.dumps({**params, "start": time.time()}) + "\n"
        log_header_fh.write(d)
        log_header_fh.close()
    except Exception:
        print("error, aborting")
        sys.exit(-1)


def getopt(optname):
    return getattr(optim, optname)


class bcolors:
    GREEN = "\033[92m"  # GREEN
    WARNING = "\033[93m"  # YELLOW
    FAIL = "\033[91m"  # RED
    RED = "\033[91m"  # RED
    RESET = "\033[0m"  # RESET COLOR


def tprint(*args):
    return tqdm.write(" ".join(map(str, args)))


def precache_ds(ds):
    N = len(ds)
    st_samples = []
    st_labels = torch.zeros((N,), dtype=torch.long)

    for i in range(N):
        x, y = ds[i]
        st_samples.append(x.unsqueeze(0))
        st_labels[i] = y

    X = torch.vstack(st_samples)
    return TensorDataset(X, st_labels)


def build_datasets(
    ds_defs,
    data_root=DATA_ROOT,
    do_download=True,
):
    ret = {}
    for name, spec in ds_defs.items():
        assert hasattr(datasets, spec["name"])
        make_ds = getattr(datasets, spec["name"])
        dim = spec["resize_to"] * spec["resize_to"] * (3 if spec.get("to_rgb") else 1)
        ds_ret = []

        # TODO omniglot background
        for train in [True, False]:
            trf = [Resize((spec["resize_to"], spec["resize_to"])), ToTensor()]

            if spec.get("to_rgb"):
                trf.append(Lambda(lambda x: torch.stack([x, x, x], 0)))

            assert name in ds_norms
            norm_spec = ds_norms[name]

            if isinstance(norm_spec, dict):
                norm_key = "train" if train else "eval"
                norm_values = norm_spec[norm_key]
                trf.append(Normalize((norm_values["mean"],), (norm_values["std"],)))
            else:  # hardcoded, see comment
                trf.append(Normalize(*norm_spec))

            ds = make_ds(
                root=DATA_ROOT,
                train=train,
                download=do_download,
                transform=Compose(trf),
            )

            print("Loading DS:", name, "train" if train else "eval", spec)

            ds_ret.append(precache_ds(ds))

        ds_ret.append(dict(num_labels=_num_labels[spec["name"]], dim=dim))
        ret[name] = ds_ret
    return ret


########################################################################################################
# MHA: Multi-head Attention + Rotary Encoding
########################################################################################################


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len))
        )

        if self.config.ROTARY_POS_EMB:
            self.rotary_ndims = int(self.head_size * 0.5)
            self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        q = (
            self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)

        if self.config.ROTARY_POS_EMB:
            q, query_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
            k, key_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
            q = torch.cat((q, query_pass), dim=-1)
            k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # causal mask
        att = F.softmax(att, dim=-1)  # softmax

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = (
            x.transpose(1, 2).contiguous().view(B, T, -1)
        )  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class PenGPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if (
            (not self.config.USE_POST_LN)
            and (self.config.USE_SMALL_EMB)
            and (self.layer_id == 0)
        ):  # LN(SmallInit(Emb))
            self.lnPre = nn.LayerNorm(config.n_embd)

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.att = MHA(config)
        self.ffn = GeGLU(config)

    def forward(self, x):
        if self.config.USE_POST_LN:
            x = self.ln1(x)
            x = x + self.att(x)
            x = self.ln2(x)
            x = x + self.ffn(x)
        else:
            if self.config.USE_SMALL_EMB and self.layer_id == 0:  # LN(SmallInit(Emb))
                x = self.lnPre(x)

            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))

        return x


class PenGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ctx_len = config.ctx_len

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not self.config.ROTARY_POS_EMB:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.ctx_len, config.n_embd)
            )  # note: i initialize abs.pos.emb to zero

        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, (nn.Embedding)):
            if self.config.USE_SMALL_EMB:
                nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)  # SmallInit(Emb)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # config is mostly for weight decay, unused for now
    def prepare_optim_groups(self, train_config={}):
        decay = (
            set()
        )  # separate out all parameters to those that will and won't experience regularizing weight decay
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if (
                    pn.endswith("bias")
                    or ("time" in fpn)
                    or ("head" in fpn)
                    or ("scale" in fpn)
                    or ("pos_emb" in fpn)
                ):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {
            pn: p for pn, p in self.named_parameters()
        }  # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.get("weight_decay", 0),
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(
        self,
        idx_or_raw,
        targets=None,
        input_raw_embeds=False,
        return_embeddings=False,
        mask=None,
    ):
        isz = idx_or_raw.size()
        T = isz[1]
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx_or_raw) if not input_raw_embeds else idx_or_raw

        if not self.config.ROTARY_POS_EMB:
            x = x + self.pos_emb[:, :T, :]

        x = self.blocks(x)

        x = self.ln_out(x)

        if return_embeddings:
            return x

        x = self.head(x)

        # print(self.emb.weight.detach().cpu().numpy()) # <-------- Show embedding matrix changes

        # loss = None
        # if targets is not None:
        #    loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        # return x, loss
        return x


def hashmix(*args):
    key = ""

    for a in args:
        key += f"_{a}"

    h_bytes = hashlib.sha256(key.encode("utf-8")).digest()[2:10]

    return int.from_bytes(h_bytes, byteorder="little", signed=False)


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def make_permuted_metalearning_dataset_loader(
    dataset,
    bs=128,
    seq_len=100,
    augs={},
    max_label_count=None,
    pad_token=0,
    add_special_tokens=0,
    validation=False,
    seed=1337,
    test=False,
    pin_memory=True,
):
    _augs = dict(
        perm_count=2**17,
        random_linear_proj=True,
        single_perm_relaxation=0.0,
        per_batch_relaxed_perm=False,
        relax_perm_only=False,
        single_perm_relaxation_multi=False,
    )

    _augs.update(augs)

    # generate linear matrices
    # generate permutations
    # Input normalization Each dataset is z-normalized by its mean and standard deviation across all examples and pixels.

    # custom_token_placeholder = pad_token
    first_label = pad_token  # see paper
    n_special_tokens = 1 + add_special_tokens  # [pad]
    base_num_labels = dataset[2]["num_labels"]
    n_total_tokens = base_num_labels + n_special_tokens
    dim = dataset[2].get("dim")

    perm_gen = torch.Generator()
    av_gen = torch.Generator()

    batch_count = -1

    perm_count = _augs.get("perm_count", 2**17)
    random_linear_proj = _augs.get("random_linear_proj", True)
    single_perm_relaxation = _augs.get("single_perm_relaxation", 0)
    single_perm_relaxation_multi = _augs.get("single_perm_relaxation_multi", False)
    per_batch_relaxed_perm = _augs.get("per_batch_relaxed_perm", False)
    relax_perm_only = _augs.get("relax_perm_only", False)

    # data distribution for learning-to-learn. This enables a different kind of intervention: Biasing the data distribution. The approach is inspired by the observation that before leaving the loss plateau the model memorizes biases in the data. Instead of sampling label permutations uniformly at random, we bias towards
    # a specific permutation by using a fixed permutation for a fraction of each batch. This completely eliminates the loss plateau, enabling a smooth path from memorizing to learning (Fig-ure 8). Surprisingly, even when heavily bias-ing the distribution, memorization is followedby generalization. This biased data distribution can be viewed as a curriculum, solving an easierp roblem first that enables the subsequent harder learning-to-learn.

    print(
        f"perm_count: {perm_count}\nrandom_linear_proj: {random_linear_proj}\nsingle_perm_relaxation: {single_perm_relaxation}\nsingle_perm_relaxation_multi: {single_perm_relaxation_multi}\nper_batch_relaxed_perm: {per_batch_relaxed_perm}\nrelax_perm_only: {relax_perm_only}\n"
    )

    fixed_perm_bc = (
        int(bs * single_perm_relaxation) + 1 if single_perm_relaxation else 0
    )
    cached_relaxed_perm_ids = (
        [random.randint(0, _augs.get("perm_count", 1)) for _ in range(fixed_perm_bc)]
        if single_perm_relaxation_multi
        else [random.randint(0, _augs.get("perm_count", 1))] * fixed_perm_bc
    )

    def collate_fn(batch):
        nonlocal batch_count
        relaxed_perm_id = (
            (
                [
                    random.randint(0, _augs.get("perm_count", 1))
                    for _ in range(fixed_perm_bc)
                ]
                if single_perm_relaxation_multi
                else [random.randint(0, _augs.get("perm_count", 1))] * fixed_perm_bc
            )
            if perm_count > 0
            else None
        )

        batch_count += 1
        ret = []
        expected_output = []
        label = []
        bi = 0

        for bi, seq in enumerate(chunked_iterable(batch, seq_len)):
            _ret = []
            _gt_ret = []
            _labels = []
            prev_y = first_label

            Av = None
            perm = None
            perm_seed = None
            av_seed = None

            if perm_count:
                if single_perm_relaxation and (bi < single_perm_relaxation * bs):
                    perm_id = (
                        relaxed_perm_id
                        if per_batch_relaxed_perm
                        else cached_relaxed_perm_ids
                    )[bi]
                    perm_seed = hashmix(perm_id, seed, "_SALT")
                    av_seed = (
                        random.randint(0, perm_count)
                        if relax_perm_only
                        else hashmix(perm_seed, "_MATRIX_S")
                    )
                else:
                    perm_id = random.randint(0, perm_count - 1)
                    perm_seed = hashmix(perm_id, seed, "_SALT")
                    av_seed = (
                        random.randint(0, perm_count)
                        if relax_perm_only
                        else hashmix(perm_seed, "_MATRIX_S")
                    )

                if random_linear_proj:
                    av_gen.manual_seed(av_seed)
                    torch.randint(2**32, (8,), generator=av_gen)
                    Av = torch.randn((dim, dim), generator=av_gen) / float(dim)

                perm_gen.manual_seed(perm_seed)
                torch.randint(2**32, (8,), generator=perm_gen)
                perm = torch.randperm(base_num_labels, generator=perm_gen)

            for si, dp in enumerate(seq):
                x, y = dp

                x = torch.matmul(Av, x.flatten()) if Av is not None else x.flatten()
                y = int(perm[y]) if perm is not None else y

                _ret.append(x)
                _labels.append(
                    F.one_hot(
                        torch.LongTensor([prev_y + n_special_tokens])[0],
                        n_total_tokens,
                    ).float()
                )
                _gt_ret.append(y + n_special_tokens)

                prev_y = y

            ret.append(torch.vstack(_ret).unsqueeze(0))
            label.append(torch.vstack(_labels).unsqueeze(0))
            expected_output.append(torch.tensor(_gt_ret, dtype=torch.long))

        return dict(
            tokens=torch.vstack(ret),
            labels=torch.vstack(label),
            expected_output=pad_sequence(expected_output, True, pad_token).long(),
        )

    return DataLoader(
        dataset[1] if validation else dataset[0],
        batch_size=bs * seq_len,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )


class GPICLWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        dim=256,
        num_labels=10,
        pad_token=0,
        n_aux_tokens=1,
        custom_token_dim=None,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_labels
        self.total_num_tokens = num_labels + n_aux_tokens
        self.pad_token = pad_token
        self.model = base_model
        self.custom_embedder = (
            nn.Linear(custom_token_dim, dim - self.total_num_tokens, bias=False)
            if custom_token_dim is not None
            else nn.Identity()
        )
        self.loss = nn.CrossEntropyLoss()
        self.cached_bmask = {}
        self.name = "GPICL_Transformer"
        self._device = device

    def configure_optimizer(self, opt, opt_conf):
        optimizer = opt(self.parameters(), **opt_conf)
        return optimizer

    def forward(
        self,
        sample,
        mask=None,
        return_embeddings=False,
        return_GPICL_loss=False,
        **kwargs,
    ):

        input_tokens = sample.get("tokens")
        input_labels = sample.get("labels")
        expected_output = sample.get("expected_output")

        x_emb = self.custom_embedder(input_tokens)

        x = torch.concat((x_emb, input_labels), dim=2).to(self._device)

        if return_GPICL_loss:
            out_logits = self.model(
                x, input_raw_embeds=True, return_embeddings=False, mask=mask, **kwargs
            )

            loss = self.loss(
                out_logits.permute(0, 2, 1),
                expected_output.to(self._device),
            )

            # mem leak?
            del sample

            return loss

        if return_embeddings:
            return self.model(
                x, input_raw_embeds=True, return_embeddings=True, mask=mask, **kwargs
            )
        else:
            return self.model(
                x, input_raw_embeds=True, return_embeddings=False, mask=mask, **kwargs
            )


seed = spec["seed"]

print(f'GPICL {_VER} SEED={seed} EXPERIMENT="{spec["name"]}"')

print("Specification:\n", json.dumps(spec, indent=True))

if not os.path.isdir("./checkpoints"):
    os.mkdir("./checkpoints")

if not os.path.isdir("./logs"):
    os.mkdir("./logs")


_datasets = build_datasets(ds_defs)

torch.manual_seed(seed)
device = torch.device(DEVICE)

base_ds = _datasets[spec["dataset"]]
base_num_labels = base_ds[2]["num_labels"]
context_len = spec["context_len"]
n_special_tokens = 1  # for [PAD]
n_total_tokens = base_num_labels + n_special_tokens
base_img_dim = base_ds[2]["dim"]  # + n_total_tokens
batch_size = spec["bs"]

train_ds = make_permuted_metalearning_dataset_loader(
    base_ds,
    bs=batch_size,
    seq_len=context_len,
    augs=spec["augs"],
    seed=seed,
    pin_memory=True,
)

val_datasets = []

val_datasets.append(
    (
        spec["dataset"],
        make_permuted_metalearning_dataset_loader(
            base_ds,
            bs=12,
            seq_len=context_len,
            augs=spec["augs"],
            validation=True,
            seed=seed,
            pin_memory=True,
        ),
    )
)

for unseen_ds in spec["unseen_datasets"]:
    assert _datasets[unseen_ds][2]["num_labels"] <= base_num_labels
    assert _datasets[unseen_ds][2]["dim"] <= base_ds[2]["dim"]

    val_datasets.append(
        (
            f"unseen_{unseen_ds}",
            make_permuted_metalearning_dataset_loader(
                _datasets[unseen_ds],
                bs=16,
                seq_len=context_len,
                augs=dict(
                    perm_count=2**25,
                    random_linear_proj=True,
                    single_perm_relaxation=0,
                ),
                validation=True,
                seed=seed,
                pin_memory=True,
            ),
        )
    )

max_seq_len = context_len + 1

model_dim = spec["model_hparams"]["model_dim"]
depth = spec["model_hparams"]["depth"]
heads = spec["model_hparams"]["heads"]

config = PenGPTConfig(
    n_total_tokens,
    max_seq_len,
    n_layer=depth,
    n_head=heads,
    n_embd=model_dim,
    n_attn=model_dim,
    n_ffn=model_dim,
    **spec["model_hparams"]["tweaks"],
)
base_model = PenGPT(config).to(device)

model = GPICLWrapper(
    base_model,
    dim=model_dim,
    num_labels=base_num_labels,
    custom_token_dim=base_img_dim,
    device=device,
)

run_id = str(uuid.uuid4())
log_stream_path = os.path.join("logs", f"{run_id}.jsonl")
log_stream_fh = open(log_stream_path, "w")


def log(data):
    try:
        json.dump(data, log_stream_fh)
        log_stream_fh.write("\n")
    except Exception:
        print("error, skipping...")


announce_exp(
    {
        "run_id": run_id,
        "log_stream_path": log_stream_path,
        "v": _VER,
        "experiment": spec,
    }
)

modes = types.SimpleNamespace(GPICL=1)

train_dataloader = train_ds
val_dataloader = val_datasets
save_at = spec["save_schedule"]
quit_at = max(save_at) + 1
total_token_count = quit_at * batch_size * context_len
total_bs = batch_size
sequence_length = max_seq_len
grad_norm_clip = 2.0
mode = modes.GPICL
opt = spec["opt"]
print("Checkpoints enabled at:", save_at)
validate_every = 10
track_fn = log


def save_model(nt, epoch, with_opt=True, meta={}):
    tqdm.write("saving...")
    torch.save(
        {
            **meta,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict() if with_opt else None,
        },
        f"./checkpoints/{run_id}_{model.name}_{feng(nt)}T_of_{feng(total_token_count)}T.pt",
    )


t0 = time.time()
nt = 0
val_batch = None

if val_dataloader is not None:
    if isinstance(val_dataloader, list):
        val_dataloader = list(
            map(
                lambda pair: (pair[0], itertools.cycle(iter(pair[1]))),
                val_dataloader,
            )
        )
    else:
        val_dataloader = itertools.cycle(iter(val_dataloader))

val_loss = None


def compute_val(val_dataloader_iter, name="default"):
    val_batch = None
    try:
        val_batch = next(val_dataloader_iter)
    except Exception as e:
        tqdm.write("Val dataloader failure:", e)

    if val_batch is None:
        return

    loss = model(val_batch, return_GPICL_loss=True)

    if "loss_val" not in metrics:
        metrics["loss_val"] = {}

    metrics["loss_val"][name] = float(loss)
    val_loss = float(loss)

    if name:
        name = f"[{name}]_"

    tprint(
        f"B[{feng(bi)}] T[{feng(nt).ljust(5)}] {name}_Val_Loss: {bcolors.RED}{round(val_loss, 4)}{bcolors.RESET}"
    )


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


tqdm.write(f"[OPT] Using {opt['name']} @ {opt['config']}")

opt = model.configure_optimizer(getopt(opt["name"]), opt["config"])

n_params = get_n_params(model)

print("Model params:", n_params, "~=", feng(n_params))

with tqdm(total=total_token_count, unit="T") as pbar:
    train_iterable = iter(train_dataloader)

    bi = -1

    while True:
        try:
            # Samples the batch
            batch_or_minibatches = next(train_iterable)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_iterable = iter(train_dataloader)
            batch_or_minibatches = next(train_iterable)

        bi += 1

        nt = bi * total_bs * sequence_length

        metrics = {"bi": bi, "t": time.time(), "events": []}

        model.zero_grad()

        if isinstance(batch_or_minibatches, list):
            minibatches = batch_or_minibatches
        else:
            minibatches = [batch_or_minibatches]

        nmb = len(minibatches)

        b_t0 = time.time()

        batch_train_loss = 0.0
        total_minibatches = 0

        for mbi, batch in enumerate(minibatches):
            total_minibatches += 1

            loss = model(batch, return_GPICL_loss=True)

            batch_train_loss += float(loss)

            loss.backward()

        batch_train_loss /= float(total_minibatches)
        metrics["loss_train"] = batch_train_loss

        b_dt = time.time() - b_t0
        metrics["batch_dt"] = b_dt

        speed_sample = b_dt / float(total_bs)
        tprint(
            f"Batch({nmb} x mb) time: {round(b_dt,2)}s, Speed: {round(speed_sample, 3)}s per sample"
        )
        tprint(
            f"B[{feng(bi)}] T[{feng(nt).ljust(5)}] loss: {bcolors.GREEN}{round(float(loss), 4)}{bcolors.RESET}"
        )

        if grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

        metrics["lr"] = float(get_lr(opt))
        opt.step()

        just_saved = False

        if val_dataloader is not None and validate_every and bi % validate_every == 0:
            with torch.no_grad():
                model.eval()

                if mode == modes.GPICL:
                    if isinstance(val_dataloader, list):
                        for vd_name, vd_iter in val_dataloader:
                            compute_val(vd_iter, vd_name)
                    else:
                        compute_val(val_dataloader)

                model.train()

        if bi in save_at:
            save_model(nt, 0, meta=dict(bi=bi, loss_val=val_loss))

        pbar.update(total_bs * sequence_length)

        if track_fn is not None:
            track_fn(metrics)

        if quit_at is not None and bi >= quit_at:
            sys.exit(0)
