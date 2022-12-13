"""
Asymmetric Numeral Systems

x: compressed message, represented by current state of the encoder/decoder.

precision: the natural numbers are divided into ranges of size 2^precision.

start & freq: start indicates the beginning of the range in [0, 2^precision-1]
that the current symbol is represented by. freq is the length of the range for
the given symbol.

The probability distribution is quantized to 2^precision, where
P(symbol) ~= freq(symbol) / 2^precision

Compressed state is represented as a stack (head, tail)
"""

import numpy as np
import torch

RANS_L = 1 << 31  # the lower bound of the normalisation interval

def empty_message(shape):
    return (np.full(shape, RANS_L, "uint64"), ())