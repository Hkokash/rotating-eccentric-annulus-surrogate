import re
import os

BASE = r"C:\Users\ADMIN\research-projects\proj-2026-ecc-annulus-01\05-manuscript"
os.chdir(BASE)

sections = ["main.tex"] + [f"sections/{f}" for f in os.listdir("sections") if f.endswith(".tex")]
all_text = ""
for s in sections:
    if os.path.exists(s):
        all_text += open(s, encoding="utf-8").read()

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text)
print("=== Figure files ===")
for f in figs:
    candidates = [f, f + ".png", f + ".pdf", f + ".jpg"]
    found = any(os.path.exists(c) for c in candidates)
    print(("OK     " if found else "MISSING"), f)

cites = set()
for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", all_text):
    cites.update(x.strip() for x in m.split(","))
bib = open("references.bib", encoding="utf-8").read()
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))

print()
print("=== Citations not in bib ===")
missing = cites - bibkeys
print(missing if missing else "none")

print()
print("=== Bib entries never cited ===")
unused = bibkeys - cites
print(unused if unused else "none")

print()
print("=== \\ref labels used vs defined ===")
labels_defined = set(re.findall(r"\\label\{([^}]+)\}", all_text))
labels_used = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", all_text))
print("Used but not defined:", labels_used - labels_defined if labels_used - labels_defined else "none")
