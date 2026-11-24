"""
SCoPE
Compression Functions
Jesus Alan Hernandez Galvan
"""

import bz2
import gzip
import lzma
import zlib

from .base import BaseCompressor


class LzmaCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_name="lzma",
        )

    def compress(self, sequence: bytes) -> bytes:
        return lzma.compress(
            sequence,
        )


class Bz2Compressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return bz2.compress(sequence, compresslevel=self._compression_level)


class ZlibCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return zlib.compress(sequence, level=self._compression_level)


class GZipCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return gzip.compress(sequence, compresslevel=self._compression_level)

