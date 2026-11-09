"""Build report/report.pdf from report/report.html.

LibreOffice does not load relative <img> paths reliably, so the figure is
embedded as base64 in a temp copy of the HTML before conversion.
Usage (from repo root): python report/build_report_pdf.py
"""
import base64, re, shutil, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
html = (HERE / "report.html").read_text()


def embed(m):
    p = (HERE / m.group(1)).resolve()
    return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'


html = re.sub(r'src="(?!data:)([^"]+\.png)"', embed, html)
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d) / "report.html"
    tmp.write_text(html)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", d, str(tmp)], check=True)
    shutil.copy(Path(d) / "report.pdf", HERE / "report.pdf")
print("wrote", HERE / "report.pdf")
