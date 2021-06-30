import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import typing
from typing import Any, Tuple

import preprocess_sequences
import utils


class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any


class DecoderOutput(typing.NamedTuple):
  logits: Any
  attention_weights: Any
