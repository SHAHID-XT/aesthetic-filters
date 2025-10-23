# 🎨 Aesthetic Image Filter

**Python-based image filter system using OpenCV and NumPy**

This project provides a fully modular and extendable Python class for applying *cinematic, vintage, aesthetic, and artistic filters* to images — directly from code.

---

## 🚀 Features

* 10+ built-in high-quality filters:

  * **warm** – cozy sunset tones
  * **cool** – bluish cold vibe
  * **cinematic** – teal-orange movie look with vignette
  * **faded** – matte vintage aesthetic
  * **vintage / vintage2** – old-film color mood
  * **low_key** – dramatic dark composition
  * **punch / punchy_cinematic** – saturated, crisp tones
  * **soft_blue** – dreamy haze
  * **emboss** – 3D texture effect

* Modular architecture – easily extend with your own filters

* Adjustable **intensity** per filter

* Works with any image type (JPG, PNG, etc.)

* Outputs smooth, balanced tones instead of overexposed garbage

* Zero dependencies beyond `opencv-python` and `numpy`

---

## 📦 Installation

```bash
pip install opencv-python numpy
```

Clone this repository:

```bash
git clone https://github.com/SHAHID-XT/aesthetic-filters.git
cd aesthetic-filters
```

---

## 🧠 Usage

Basic example:

```python
from filters import ImageFilter

f = ImageFilter()
f.apply_filter("cinematic", "input.jpg", outpath="output.jpg", intensity=0.8)
```

That’s it — your image is now blessed with teal-orange energy.

---

## ⚙️ Available Filters

| Filter Name        | Description                                    |
| ------------------ | ---------------------------------------------- |
| `warm`             | Adds warmth, soft contrast, gentle glow        |
| `cool`             | Slight blue tone with smooth highlights        |
| `cinematic`        | Teal-orange LUT-style tone with vignette       |
| `faded`            | Desaturated, flat matte look                   |
| `vintage`          | Sepia and subtle grain for film feel           |
| `low_key`          | Deep shadows and moody lighting                |
| `punch`            | Vibrant contrast with rich saturation          |
| `soft_blue`        | Dreamy, smooth pastel tint                     |
| `punchy_cinematic` | Sharper and more dramatic version of cinematic |
| `emboss`           | Stylized artistic texture effect               |
| `vintage2`         | Stronger blend of vintage + fade               |

---

## 🔧 Parameters

| Parameter     | Type    | Description                                          |
| ------------- | ------- | ---------------------------------------------------- |
| `filter_name` | `str`   | The name of the filter (from list above)             |
| `image_path`  | `str`   | Path to the input image                              |
| `outpath`     | `str`   | Path where the output will be saved                  |
| `intensity`   | `float` | Value between 0–1 controlling strength of the effect |

---

## 🧩 Add Your Own Filters

You can define your own filter like this inside the `ImageFilter` class:

```python
def filter_my_custom(self, img, intensity=0.7):
    s = self.bilateral_smooth(img)
    t = self.adjust_contrast_brightness(s, contrast=1.2, bright=0.05)
    out = self.color_shift(t, shift=(0.02, 0.0, -0.03))
    return self.blend(img, out, intensity)
```

Then register it in the constructor:

```python
self.FILTERS["my_custom"] = self.filter_my_custom
```

---

## 🖼️ Example Output

| Filter    | Preview                              |
| --------- | ------------------------------------ |
| Cinematic | ![cinematic](examples/cinematic.jpg) |
| emboss     | ![emboss](examples/faded.jpg)       |
| Warm      | ![warm](examples/warm.jpg)           |

---

## 🧰 Internals

Each filter uses combinations of:

* Bilateral smoothing for edge-preserving blur
* Gaussian glow blending
* Per-channel RGB curve adjustments
* Sepia, vignette, and S-curve contrast corrections
* Custom blending masks for vignette and matte effects


---

## 🌈 Credits

Made with Python, OpenCV, and unhealthy amounts of caffeine.
Inspired by the film LUT aesthetics and classic digital color grading styles.

---

