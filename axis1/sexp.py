"""Minimal PDDL s-expression reader/writer used by the obfuscator.

Comments are dropped while reading (they leak names such as source URLs), and all
symbols are lower-cased, as PDDL is case-insensitive and Pyperplan lower-cases too.
"""

import re

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def strip_comments(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def tokenize(text):
    return _TOKEN.findall(strip_comments(text).lower())


def parse(text):
    """Parse one top-level s-expression into nested Python lists of strings."""
    tokens = tokenize(text)
    pos = 0

    def read():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while tokens[pos] != ")":
                out.append(read())
            pos += 1
            return out
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    tree = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens after the top-level expression")
    return tree


def to_text(expr, indent=0):
    """Pretty-print: flat lists on one line, nested lists one child per line."""
    pad = "  " * indent
    if isinstance(expr, str):
        return pad + expr
    if all(isinstance(e, str) for e in expr):
        return pad + "(" + " ".join(expr) + ")"
    head = []
    rest = list(expr)
    while rest and isinstance(rest[0], str) and not (rest[0].startswith(":") and head):
        head.append(rest.pop(0))
    lines = [pad + "(" + " ".join(head)]
    i = 0
    while i < len(rest):
        e = rest[i]
        # keep ":keyword value" pairs together on one line when the value is flat
        if isinstance(e, str) and e.startswith(":") and i + 1 < len(rest):
            nxt = rest[i + 1]
            if isinstance(nxt, str) or all(isinstance(x, str) for x in nxt):
                lines.append(pad + "  " + e + " " + to_text(nxt).strip())
                i += 2
                continue
            lines.append(pad + "  " + e)
            lines.append(to_text(nxt, indent + 2))
            i += 2
            continue
        lines.append(to_text(e, indent + 1))
        i += 1
    lines[-1] += ")"
    return "\n".join(lines)


def atoms(expr):
    """Yield every atom (string) in an expression."""
    if isinstance(expr, str):
        yield expr
    else:
        for e in expr:
            yield from atoms(e)


def map_atoms(expr, fn):
    if isinstance(expr, str):
        return fn(expr)
    return [map_atoms(e, fn) for e in expr]


# ---------- typed lists: "a b - t c - u d" ----------

def parse_typed_list(items):
    """Return [(name, type_or_None)] in order. Types may be atoms or (either ...) lists."""
    out, pending = [], []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-":
            typ = items[i + 1]
            out += [(n, typ) for n in pending]
            pending = []
            i += 2
        else:
            pending.append(tok)
            i += 1
    out += [(n, None) for n in pending]
    return out


def unparse_typed_list(pairs):
    """Write one 'name - type' per entry; untyped names go last so they stay untyped."""
    typed = [p for p in pairs if p[1] is not None]
    untyped = [p for p in pairs if p[1] is None]
    out = []
    for name, typ in typed:
        out += [name, "-", typ]
    out += [name for name, _ in untyped]
    return out


def section(tree, key):
    """Return the first sub-list of a define-block whose head is `key`, or None."""
    for e in tree:
        if isinstance(e, list) and e and e[0] == key:
            return e
    return None
