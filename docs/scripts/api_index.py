"""Print the markdown tables of docs/site/api/index.md (run in a subprocess by markdown-exec)."""

import html
import re

import griffe

pkg = griffe.load("bulanik", search_paths=["src"], docstring_parser=None)

# (page, module whose names are listed)
SECTIONS = [
    ("numbers", "bulanik.numbers"),
    ("aggregate", "bulanik.aggregate"),
    ("sets", "bulanik.sets"),
    ("scales", "bulanik.scales"),
    ("inference", "bulanik.inference"),
    ("type2", "bulanik.type2"),
    ("anfis", "bulanik.anfis"),
    ("mcdm", "bulanik.mcdm"),
    ("mcdm", "bulanik.mcdm.bwm"),
    ("lumix", "bulanik.lumix"),
    ("sezgi", "bulanik.sezgi"),
    ("plot", "bulanik.plot"),
]


def names_of(mod):
    exported = mod.exports
    if exported:
        return [str(n) for n in exported]
    return [n for n, m in mod.members.items() if not n.startswith("_") and not m.is_alias and not m.is_module]


ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "etc.", "vs.", "cf.", "resp.")


def strip_refs(text):
    """Cross-reference markup to its text: [`Term`][id] -> `Term`, [text](url) -> text."""
    text = re.sub(r"\[(`[^`\]]+`)\]\[[\w.]*\]", r"\1", text)
    text = re.sub(r"\[([^\]`]+)\]\[([\w]+\.[\w.]*)\]", r"\1", text)
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)


def first_sentence(text):
    """Up to the first full stop outside parentheses, backticks and math, skipping abbreviations and '...'."""
    depth, code, math = 0, False, False
    for i, ch in enumerate(text):
        if ch == "`":
            code = not code
        elif ch == "$" and not code:
            math = not math
        elif code or math:
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0 and (i + 1 == len(text) or text[i + 1] == " "):
            if text[max(0, i - 1) : i] == "." or text[: i + 1].endswith(ABBREVIATIONS):
                continue
            return text[: i + 1]
    return text


def to_html(text):
    """Escape text; render `code` and $math$ spans (math through the arithmatex class)."""
    parts = []
    for piece in re.split(r"(``.+?``|`[^`]*`|\$[^$]*\$)", text):
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 1:
            parts.append(f"<code>{html.escape(piece.strip('`').strip())}</code>")
        elif piece.startswith("$") and piece.endswith("$") and len(piece) > 1:
            parts.append(f'<span class="arithmatex">\\({html.escape(piece[1:-1])}\\)</span>')
        else:
            parts.append(html.escape(piece))
    return "".join(parts)


def summary(obj):
    target = obj.final_target if obj.is_alias else obj
    doc = target.docstring.value if target.docstring else ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip() if doc else ""
    return to_html(first_sentence(strip_refs(first)))


def kind(obj):
    target = obj.final_target if obj.is_alias else obj
    return "class" if target.is_class else "function" if target.is_function else "constant"


out = []
for page, path in SECTIONS:
    mod = pkg
    for part in path.split(".")[1:]:
        mod = mod.members[part]
    rows = []
    if path == "bulanik.inference":
        # public names re-exported by the package, then the two rules-only helpers
        names = names_of(mod)
        extra = [n for n in names_of(mod.members["rules"]) if n not in names]
        owners = {n: mod for n in names}
        owners.update({n: mod.members["rules"] for n in extra})
        names = sorted(owners, key=str.lower)
        for n in names:
            owner = owners[n]
            target = owner.members[n]
            anchor = target.final_target.path if target.is_alias else target.path
            rows.append((n, kind(target), summary(target), anchor))
    else:
        for n in sorted(names_of(mod), key=lambda s: (s.startswith("__"), s.lower())):
            if n.startswith("__"):
                continue
            target = mod.members[n]
            anchor = f"{path}.{n}"
            rows.append((n, kind(target), summary(target), anchor))
    out.append(f'<h2><code>{path}</code></h2>')
    out.append(f'<p>Full entries: <autoref identifier="{path}">{path}</autoref></p>')
    out.append("<table><thead><tr><th>Name</th><th>Kind</th><th>Summary</th></tr></thead><tbody>")
    for n, k, s, anchor in rows:
        out.append(
            f'<tr><td><autoref identifier="{anchor}"><code>{n}</code></autoref></td>'
            f"<td>{k}</td><td>{s}</td></tr>"
        )
    out.append("</tbody></table>")
print("\n".join(out))
