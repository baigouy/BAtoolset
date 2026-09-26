"""Conservative typography tidy-up for figure text.

Only the *objective, unambiguous* fixes are applied — the kind that are tedious
to do by hand but safe to automate:

  - number × number      2*3        -> 2×3            (multiplication sign)
  - temperatures         37 degrees C -> 37 °C
  - angles               45 degrees -> 45°            (skips 'degrees of freedom')
  - number ↔ unit gap    10µm       -> 10 µm          (curated unit whitelist only)
  - numeric ranges       10-20      -> 10–20          (en dash; ISO dates protected)
  - negative sign        -5         -> −5             (proper minus sign)
  - runaway whitespace   'a   b'    -> 'a b'          (+ trim trailing spaces)

Deliberately NOT done here (too context-dependent — leave to a human / LLM):
italicising variables, superscripting unit divisions, %-placement, capitalisation.
"""
import re

# Curated, unambiguous units only. No bare single letters (m/s/g/K/M...) because
# they collide with things like "10K" (=10000), "2D", "H2O", gene names, etc.
_UNITS = [
    'µm', 'um', 'nm', 'mm', 'cm', 'dm', 'km',
    'µl', 'ul', 'ml', 'mL', 'µL',
    'µg', 'ug', 'mg', 'ng', 'pg', 'kg',
    'ms', 'µs', 'us', 'ns', 'min',
    'kHz', 'MHz', 'GHz', 'Hz',
    'bp', 'kb', 'Mb', 'kDa', 'Da',
    'mV', 'kV', 'mA', 'µA', 'kW', 'mW', 'kPa', 'Pa',
    'mM', 'µM', 'nM', 'pM', 'rpm', '°C', '°F',
]
# longest first so 'ml' wins over a would-be 'm', 'MHz' over 'Hz', etc.
_UNIT_ALT = '|'.join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))

_MULT = re.compile(r'(\d\s*)\*(\s*\d)')
_DEG_TEMP = re.compile(r'(\d)\s*degrees?\s*([CF])\b', re.IGNORECASE)
_DEG_ANGLE = re.compile(r'(\d)\s*degrees?\b(?!\s*of\s+freedom)', re.IGNORECASE)
_UNIT_GAP = re.compile(r'(?<![A-Za-z\d])(\d+(?:\.\d+)?)(' + _UNIT_ALT + r')\b')
_ISO_DATE = re.compile(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{2,4}')
_NEG = re.compile(r'(?<![\w–—−])-(\d)')
_RANGE = re.compile(r'(\d)\s*-\s*(\d)')
_MULTISPACE = re.compile(r'[ \t]{2,}')
_TRAILING = re.compile(r'[ \t]+(\n|$)')
_TAG = re.compile(r'<[^>]*>')
# whole blocks whose *content* is not visible label text and must never be tidied
# (Qt's toHtml() wraps captions in a full document with a <head>/<style> block)
_SKIP_BLOCK = re.compile(r'<(style|script|head)\b[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)


def tidy_typography(text, multiplication=True, degrees=True, unit_spacing=True,
                    ranges=True, collapse_spaces=True):
    """Return *text* with the safe typography fixes applied. Each rule can be
    toggled off. Idempotent (running it twice changes nothing more)."""
    if not text or not isinstance(text, str):
        return text
    s = text

    if multiplication:
        s = _MULT.sub(r'\1×\2', s)

    if degrees:
        s = _DEG_TEMP.sub(r'\1 °\2', s)
        s = _DEG_ANGLE.sub(r'\1°', s)

    if unit_spacing:
        s = _UNIT_GAP.sub(r'\1 \2', s)

    if ranges:
        # shield ISO-ish dates from the range/negative rules
        stash = []

        def _keep(m):
            stash.append(m.group(0))
            return '\x00%d\x00' % (len(stash) - 1)

        s = _ISO_DATE.sub(_keep, s)
        s = _NEG.sub(r'−\1', s)          # leading hyphen before a number -> minus sign
        s = _RANGE.sub(r'\1–\2', s)      # number-number -> en dash
        s = re.sub(r'\x00(\d+)\x00', lambda m: stash[int(m.group(1))], s)

    if collapse_spaces:
        s = _MULTISPACE.sub(' ', s)
        s = _TRAILING.sub(r'\1', s)

    return s


def tidy_html(html, **flags):
    """Apply tidy_typography to the *visible* text of an HTML string, leaving all
    tags, tag attributes and <style>/<script>/<head> blocks untouched (so CSS like
    'margin:-1px' is never mangled). Whitespace collapsing is off by default here
    because HTML manages its own spacing."""
    if not html or not isinstance(html, str):
        return html
    flags.setdefault('collapse_spaces', False)

    # 1. shield non-text blocks (their CSS/JS must not be tidied)
    blocks = []

    def _shield(m):
        blocks.append(m.group(0))
        return '\x01%d\x01' % (len(blocks) - 1)

    tmp = _SKIP_BLOCK.sub(_shield, html)

    # 2. tidy only the text that sits between the remaining tags
    out, last = [], 0
    for m in _TAG.finditer(tmp):
        out.append(tidy_typography(tmp[last:m.start()], **flags))
        out.append(m.group(0))  # tag verbatim
        last = m.end()
    out.append(tidy_typography(tmp[last:], **flags))
    result = ''.join(out)

    # 3. restore the shielded blocks
    if blocks:
        result = re.sub('\x01(\\d+)\x01', lambda m: blocks[int(m.group(1))], result)
    return result
