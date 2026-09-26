"""How an octet sequence is read as UTF-8."""

__all__ = ["decode"]

# Each row gives a lead byte of a UTF-8 sequence by its first and last
# value, the number of tail bytes after it and the range of the byte
# that follows it; every later tail byte is 0x80 to 0xBF (RFC 3629, 4.
# Syntax of UTF-8 Byte Sequences).
_SEQUENCES = (
    (0x00, 0x7F, 0, 0x80, 0xBF),
    (0xC2, 0xDF, 1, 0x80, 0xBF),
    (0xE0, 0xE0, 2, 0xA0, 0xBF),
    (0xE1, 0xEC, 2, 0x80, 0xBF),
    (0xED, 0xED, 2, 0x80, 0x9F),
    (0xEE, 0xEF, 2, 0x80, 0xBF),
    (0xF0, 0xF0, 3, 0x90, 0xBF),
    (0xF1, 0xF3, 3, 0x80, 0xBF),
    (0xF4, 0xF4, 3, 0x80, 0x8F),
)
_TAIL = range(0x80, 0xC0)


def decode(data: bytes) -> str:
    # An octet sequence is UTF-8 only if every sequence in it matches
    # the syntax of UTF-8 (RFC 3629, 4. Syntax of UTF-8 Byte Sequences).
    chars = []
    at = 0
    while at < len(data):
        lead = data[at]
        for first, last, tails, low, high in _SEQUENCES:
            if first <= lead <= last:
                break
        else:
            raise ValueError(f"not UTF-8 at byte {at}")
        sequence = data[at + 1 : at + 1 + tails]
        if len(sequence) < tails or not all(byte in _TAIL for byte in sequence):
            raise ValueError(f"not UTF-8 at byte {at}")
        if tails and not low <= sequence[0] <= high:
            raise ValueError(f"not UTF-8 at byte {at}")
        # The lead byte holds the code point's first bits, below its
        # length marker; each tail byte holds six more.
        code = lead & (0xFF >> (tails + 2)) if tails else lead
        for byte in sequence:
            code = code << 6 | byte & 0x3F
        chars.append(chr(code))
        at += 1 + tails
    return "".join(chars)
