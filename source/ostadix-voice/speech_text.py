"""Lossless text chunking for the Android speech transports."""


def speech_chunks(text, limit=3000):
    """Yield all text in chunks bounded by UTF-16 code units.

    The installed helper accepts at most 3900 units per utterance. The default
    3000 units also fit comfortably in its 32768-byte JSON command, including
    JSON's worst-case six-byte escapes. Prefer whitespace boundaries, retaining
    every character so joining the chunks recovers the complete response.
    """
    if limit < 2:
        raise ValueError('Speech chunks require room for a Unicode character')
    start = 0
    while start < len(text):
        end, units, boundary = start, 0, None
        while end < len(text):
            width = 2 if ord(text[end]) > 0xffff else 1
            if units + width > limit:
                break
            units += width
            end += 1
            if text[end - 1].isspace() and units >= limit // 2:
                boundary = end
        if end < len(text) and boundary is not None:
            end = boundary
        yield text[start:end]
        start = end
