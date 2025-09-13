<h1 align="center">🔢 Digits Image Dataset</h1>

<p align="center">
  <img alt="Type" src="https://img.shields.io/badge/Type-Grayscale%20Images-8B5CF6">
  <img alt="Shape" src="https://img.shields.io/badge/Shape-32×32-10B981">
  <img alt="Count" src="https://img.shields.io/badge/Images-12,000-22C55E">
  <img alt="Format" src="https://img.shields.io/badge/Format-PNG%20(%2Ejpg%20names)-F59E0B">
  <img alt="Status" src="https://img.shields.io/badge/Status-Private%20(Local)-6366F1">
  <a href="#license-terms">
    <img alt="License" src="https://img.shields.io/badge/License-owner%20to%20specify-0EA5E9">
  </a>
</p>

<!-- Decorative gradient divider (inline SVG works on GitHub) -->
<p align="center">
  <svg width="100%" height="6" viewBox="0 0 100 6" preserveAspectRatio="none">
    <defs>
      <linearGradient id="g1" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="#FB923C"/>
        <stop offset="50%" stop-color="#38BDF8"/>
        <stop offset="100%" stop-color="#A78BFA"/>
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="100" height="6" fill="url(#g1)"/>
  </svg>
</p>

<blockquote>
  A compact <b>digits image</b> collection for classical CV/ML tasks (e.g., classification, clustering, autoencoders).
  Images are <b>32×32</b>, <b>grayscale (L)</b>, and stored as <b>PNG</b> though current filenames use <code>.jpg</code>.
</blockquote>

<hr>

<h2 id="dataset-summary">📌 Dataset Summary</h2>

<p>
  This folder contains <b>12,000</b> digit images under <code>digits/</code>, named sequentially:
  <code>00000.jpg</code> … <code>11999.jpg</code>. Each file decodes to a <b>PNG</b> (not JPEG) and is <b>grayscale 8-bit</b>, <b>32×32</b> pixels.
  <br>
  <b>Labels:</b> Not included in the archive. If you plan supervised tasks, add a <code>labels.csv</code> (<code>filename,label</code>) or organize into class folders.
</p>

<h3>Quick Facts</h3>
<table>
  <tr><td>🧾 <b>Total images</b></td><td>12,000</td></tr>
  <tr><td>🖼️ <b>Size</b></td><td>32×32 pixels</td></tr>
  <tr><td>🎚️ <b>Channels</b></td><td>Grayscale (mode <code>L</code>)</td></tr>
  <tr><td>📦 <b>Encoded format</b></td><td>PNG (filenames currently end with <code>.jpg</code>)</td></tr>
  <tr><td>🏷️ <b>Labels</b></td><td>Not provided (optional <code>labels.csv</code> supported)</td></tr>
</table>

<hr>

<h2>🔗 Table of Contents</h2>
<ul>
  <li><a href="#files">📂 Files in This Folder</a></li>
  <li><a href="#canonical-structure">🗃️ Recommended Canonical Structure</a></li>
  <li><a href="#fix-extensions">🛠️ Fix Wrong File Extensions</a></li>
  <li><a href="#splits">✂️ Create Train/Val/Test Splits</a></li>
  <li><a href="#quick-start">🔎 Quick Start (Python)</a></li>
  <li><a href="#citation">📑 Citation</a></li>
  <li><a href="#acknowledgements">🙏 Acknowledgements</a></li>
  <li><a href="#license-terms">🛡️ License &amp; Terms</a></li>
</ul>

<hr>

<h2 id="files">📂 Files in This Folder</h2>

<pre><code>Dataset/
├─ digits/
│  ├─ 00000.jpg
│  ├─ 00001.jpg
│  ├─ 00002.jpg
│  ├─ 00003.jpg
│  └─ ... (total 12,000 files; each actually PNG, 32×32, grayscale)
└─ README.md   # this file
</code></pre>

<p><b>Note:</b> Filenames have <code>.jpg</code> but decode as <b>PNG</b>. See <a href="#fix-extensions">fix steps</a>.</p>

<hr>

<h2 id="canonical-structure">🗃️ Recommended Canonical Structure</h2>

<p>For supervised tasks (with labels), consider organizing like this:</p>

<pre><code>Dataset/
├─ images/
│  └─ *.png                      # all images with correct .png extension
├─ labels.csv                    # optional: filename,label
├─ splits/
│  ├─ train.txt                  # one filename per line
│  ├─ val.txt
│  └─ test.txt
└─ README.md
</code></pre>

<p>Alternatively, you can use class folders if labels are 0–9:</p>

<pre><code>Dataset/
└─ images_by_class/
   ├─ 0/  1/  2/  ...  9/        # place images per class
</code></pre>

<hr>

<h2 id="fix-extensions">🛠️ Fix Wrong File Extensions</h2>

<details>
  <summary><b>Python script: detect PNG signature &amp; rename to .png</b></summary>

```python
from pathlib import Path
import shutil

root = Path("Dataset/digits")
out  = Path("Dataset/images")
out.mkdir(parents=True, exist_ok=True)

def is_png(p: Path) -> bool:
    with p.open("rb") as f:
        return f.read(8) == b"\\x89PNG\\r\\n\\x1a\\n"

for p in sorted(root.glob("*.jpg")):
    target = out / (p.stem + ".png")
    if is_png(p):
        shutil.copy2(p, target)   # or p.rename(target) if you want to move
    else:
        # fallback: copy as-is with original suffix
        shutil.copy2(p, out / p.name)
print("Done. Converted/copied -> Dataset/images/")
```
</details>

<hr>

<h2 id="splits">✂️ Create Train/Val/Test Splits</h2>

<details>
  <summary><b>80/10/10 split &amp; optional labels.csv scaffold</b></summary>

```python
import random
from pathlib import Path

imgs = sorted(Path("Dataset/images").glob("*.png"))
random.seed(42)
random.shuffle(imgs)

n = len(imgs)
train, val = int(0.8*n), int(0.9*n)

splits = {
    "train.txt": imgs[:train],
    "val.txt":   imgs[train:val],
    "test.txt":  imgs[val:]
}

Path("Dataset/splits").mkdir(parents=True, exist_ok=True)
for name, items in splits.items():
    with open(Path("Dataset/splits")/name, "w") as f:
        for p in items:
            f.write(p.name + "\\n")

# OPTIONAL: labels.csv template (fill labels later)
with open("Dataset/labels.csv", "w") as f:
    f.write("filename,label\\n")
    for p in imgs:
        f.write(f"{p.name},\\n")   # fill labels manually or via a script
print("Wrote splits and labels.csv template.")
```
</details>

<hr>

<h2 id="quick-start">🔎 Quick Start (Python)</h2>

<details>
  <summary><b>PyTorch</b></summary>

```python
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class DigitsDataset(Dataset):
    def __init__(self, img_dir="Dataset/images", split_file="Dataset/splits/train.txt", labels_csv=None, transform=None):
        self.root = Path(img_dir)
        self.transform = transform
        self.files = [line.strip() for line in open(split_file)]
        self.labels = None
        if labels_csv:
            import csv
            m = {}
            with open(labels_csv) as f:
                for row in csv.DictReader(f):
                    m[row["filename"]] = row["label"]
            self.labels = [int(m.get(fn)) if m.get(fn) not in (None, "") else None for fn in self.files]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fp = self.root / self.files[idx]
        img = Image.open(fp).convert("L")  # grayscale
        if self.transform: img = self.transform(img)
        if self.labels is None or self.labels[idx] is None:
            return img
        return img, self.labels[idx]

ds = DigitsDataset()
loader = DataLoader(ds, batch_size=64, shuffle=True)
```
</details>

<details>
  <summary><b>TensorFlow / tf.data</b></summary>

```python
import tensorflow as tf
IMG_DIR = "Dataset/images"

def decode_png(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)      # grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [32, 32])        # safety

paths = tf.data.Dataset.list_files(IMG_DIR + "/*.png", shuffle=True)
ds = paths.map(decode_png).batch(64).prefetch(tf.data.AUTOTUNE)
```
</details>

<hr>

<h2 id="citation">📑 Citation</h2>

<pre><code>Sudipta Kumar Das. "Digits Image Dataset." Private collection, 2025.
URL (if published later): &lt;to be added&gt;
</code></pre>

<hr>

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

<ul>
  <li><b>Sudipta Kumar Das</b> — dataset preparation and curation.</li>
  <li>Any scripts, models, or tools used to generate or verify images—acknowledge here if applicable.</li>
</ul>

<blockquote>
  Any analyses or errors are solely those of the authors.
</blockquote>

<hr>

<h2 id="license-terms">🛡️ License &amp; Terms</h2>

<p>
  <b>Choose a license</b> before sharing (e.g., <i>CC BY-NC 4.0</i> for non-commercial, or <i>CC BY 4.0</i> for permissive reuse).
  Until specified, this dataset is for <b>personal research/education only</b>.
</p>

<p align="center">
  <img alt="License notice" src="https://img.shields.io/badge/Usage-Specify%20a%20clear%20license-EAB308">
</p>

<hr>

<p align="center">
  <sub>Like this README? ⭐ Star the repo and share with your team.</sub>
</p>
