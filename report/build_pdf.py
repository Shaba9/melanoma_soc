"""Build report/paper.pdf from report/paper.md.

Converts the Markdown source to IEEE-styled two-column HTML (figures embedded,
equations rendered with MathJax) and prints it to PDF with headless Chromium.
Produces report/paper.html (intermediate) and report/paper.pdf (deliverable).
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
PAPER_DIR = REPORT / "paper"
MD = PAPER_DIR / "paper.md"
HTML = PAPER_DIR / "paper.html"
PDF = PAPER_DIR / "paper.pdf"

text = MD.read_text(encoding="utf-8")

# Protect math ($$...$$ and $...$) from markdown processing.
math_blocks: list[str] = []


def _stash(m: re.Match) -> str:
    math_blocks.append(m.group(0))
    return f"@@MATH{len(math_blocks) - 1}@@"


text = re.sub(r"\$\$.*?\$\$", _stash, text, flags=re.DOTALL)
text = re.sub(r"(?<!\$)\$[^\$\n]+\$", _stash, text)

# Resolve figure paths to absolute file URIs so Chromium can load them.
def _img(m: re.Match) -> str:
    alt, rel = m.group(1), m.group(2)
    p = (REPORT / rel).resolve()
    return f"![{alt}]({p.as_uri()})"


text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img, text)

body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
)

# Restore math placeholders.
for i, blk in enumerate(math_blocks):
    body = body.replace(f"@@MATH{i}@@", blk)

# Wrap each <img> and its caption (markdown emits both in one <p>) into a
# <figure> so the caption stays with the image, and cap image height so the tall
# Grad-CAM figure cannot overflow its column and overlap text.
body = re.sub(
    r"<p>(<img[^>]*?>)\s*(.*?)</p>",
    lambda m: f'<figure class="fig">{m.group(1)}'
              f'<figcaption>{m.group(2).strip()}</figcaption></figure>',
    body,
    flags=re.DOTALL,
)

# Keep each table title with its table so the title cannot be orphaned at the
# bottom of a column.
body = re.sub(
    r"(<p><strong>TABLE[^<]*</strong></p>)\s*(<table>.*?</table>)",
    r'<div class="tablewrap">\1\2</div>',
    body,
    flags=re.DOTALL,
)

CSS = """
@page { size: Letter; margin: 0.75in 0.6in; }
body { font-family: 'Times New Roman', Times, serif; font-size: 9.5pt;
       line-height: 1.28; color: #000; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 2pt; column-span: all;
     line-height: 1.2; }
.byline { text-align: center; font-size: 10pt; margin: 2pt 0 10pt;
          column-span: all; }
h2 { font-size: 10pt; text-transform: uppercase; text-align: center;
     margin: 12pt 0 4pt; }
h3 { font-size: 10pt; font-style: italic; margin: 8pt 0 3pt; font-weight: normal; }
.content { column-count: 2; column-gap: 0.3in; text-align: justify; }
p { margin: 0 0 5pt; }
img { max-width: 100%; height: auto; }
figure { break-inside: avoid; margin: 7pt 0 4pt; text-align: center; }
figure img { display: block; margin: 0 auto; max-width: 100%; width: auto;
             height: auto; max-height: 8.4in; }
figcaption { font-size: 8.5pt; text-align: left; margin-top: 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0 6pt; }
th, td { border: 1px solid #000; padding: 2px 4px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
hr { border: none; border-top: 1px solid #bbb; margin: 8pt 0; }
strong { font-weight: bold; }
table { break-inside: avoid; }
.tablewrap { break-inside: avoid; }
h2, h3 { break-after: avoid; }
"""

MATHJAX = """
<script>
window.MathJax = { tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
                   svg: { fontCache: 'global' } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>
"""

# Split the first <h1> and the author/affiliation lines out of the two-column flow.
html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>{CSS}</style>{MATHJAX}</head>
<body>
<div class="content">
{body}
</div>
</body></html>"""

HTML.write_text(html_doc, encoding="utf-8")
print(f"Wrote {HTML}")

from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page()
    page.goto(HTML.as_uri())
    # Wait for MathJax to finish typesetting.
    page.wait_for_timeout(2500)
    page.pdf(path=str(PDF), format="Letter",
             margin={"top": "0.75in", "bottom": "0.75in",
                     "left": "0.6in", "right": "0.6in"},
             print_background=True)
    browser.close()

print(f"Wrote {PDF}")
