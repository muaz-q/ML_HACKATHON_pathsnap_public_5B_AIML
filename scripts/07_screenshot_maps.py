"""Screenshot Folium HTML maps to PNG (for the PDF report) with headless Chrome."""
import glob
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def shot(html, png, w=1400, h=900):
    url = "file:///" + os.path.abspath(html).replace("\\", "/")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={w},{h}", "--virtual-time-budget=15000",
                    f"--screenshot={os.path.abspath(png)}", url], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "figures")
    os.makedirs(out, exist_ok=True)
    for html in glob.glob(os.path.join(ROOT, "results", "maps", "*.html")) + \
            glob.glob(os.path.join(ROOT, "results", "eda", "*.html")):
        png = os.path.join(out, os.path.basename(html).replace(".html", ".png"))
        shot(html, png)
        print(png)
