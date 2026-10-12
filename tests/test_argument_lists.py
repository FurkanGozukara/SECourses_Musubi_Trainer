"""Optimizer / LR scheduler / network arguments reach every trainer as clean key=value items however they were typed
(unified/arguments.py): multiple spaces, commas, new lines, ';', quotes, brackets, `key = value`, lowercase booleans."""

from __future__ import annotations

import ast
import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from musubi_tuner_gui.unified.arguments import argument_items, argument_problems, split_items  # noqa: E402

ADAFACTOR = ["scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01"]


@pytest.mark.parametrize(
    "typed",
    [
        "scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01",
        "scale_parameter=False   relative_step=False\twarmup_init=False    weight_decay=0.01  ",
        "scale_parameter=False,relative_step=False,warmup_init=False,weight_decay=0.01",
        "scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.01,",
        "scale_parameter=False\nrelative_step=False\r\nwarmup_init=False\n\nweight_decay=0.01\n",
        "scale_parameter=False; relative_step=False; warmup_init=False; weight_decay=0.01",
        "scale_parameter = False relative_step= False warmup_init =False weight_decay = 0.01",
        '"scale_parameter=False" \'relative_step=False\' "warmup_init=False" "weight_decay=0.01"',
        "scale_parameter=false relative_step=FALSE warmup_init=False weight_decay=0,01",
        "['scale_parameter=False', 'relative_step=False', 'warmup_init=False', 'weight_decay=0.01']",
        ["scale_parameter=False", " relative_step=False ", "warmup_init=False weight_decay=0.01"],
    ],
)
def test_every_way_of_typing_gives_the_same_items(typed):
    assert argument_items(typed, literal=True) == ADAFACTOR


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("betas=(0.9, 0.999) eps=1e-8", ["betas=(0.9, 0.999)", "eps=1e-8"]),
        ('"betas=(0.9, 0.999)" eps=1e-8', ["betas=(0.9, 0.999)", "eps=1e-8"]),
        ("betas=0.9,0.999 eps=1e-8", ["betas=0.9,0.999", "eps=1e-8"]),
        ("betas=0.9, 0.999 eps=1e-8", ["betas=0.9,0.999", "eps=1e-8"]),
        ("betas=0.9 0.999", ["betas=0.9,0.999"]),
        ("mode='triangular2' step_rules='1:10,0.1:20,0.01'", ["mode='triangular2'", "step_rules='1:10,0.1:20,0.01'"]),
        ("d_coef=2 use_bias_correction=true safeguard_warmup=TRUE", ["d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]),
        ("foo=none bar=null", ["foo=None", "bar=None"]),
    ],
)
def test_values_keep_together(typed, expected):
    items = argument_items(typed, literal=True)
    assert items == expected
    for item in items:
        ast.literal_eval(item.split("=", 1)[1])  # what musubi / sd-scripts do with every value


def test_network_arguments_keep_their_text():
    typed = "exclude_patterns=['.*(norm|bias).*', '.*embed.*']  loraplus_lr_ratio = 4\nverbose=true"
    assert argument_items(typed, literal=False) == ["exclude_patterns=['.*(norm|bias).*', '.*embed.*']", "loraplus_lr_ratio=4", "verbose=true"]
    # sd-scripts' own syntax: a value that holds '=' and a comma stays one item
    reg = r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3"
    assert split_items(reg) == [reg]


def test_problems_name_the_broken_items():
    assert argument_problems("Optimizer Arguments", ADAFACTOR, literal=True) == []
    problems = argument_problems("Optimizer Arguments", "weight_decay=0.01 oops betas=(0.9", literal=True)
    assert len(problems) == 2 and "oops" in problems[0] and "betas=(0.9" in problems[1]
    assert argument_problems("Optimizer Arguments", "mode=triangular2", literal=True) == []  # a bare word is passed as text
    assert argument_problems("Network Arguments", "conv_dim=4 =3", literal=False)


def test_empty_values():
    assert argument_items("", literal=True) == [] and argument_items(None, literal=True) == [] and argument_items([], literal=True) == []
    assert argument_items("  \n ; ,  ", literal=True) == []
