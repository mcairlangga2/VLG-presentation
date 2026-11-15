# Knowledge Enrichment in VLMs: Training & Editing

Slides built with [manim-slides](https://github.com/jeertmans/manim-slides). Equations are drawn with matplotlib's
mathtext (`mathobj.py`), so **no LaTeX install is needed**.

## Setup

Requires Python 3.12 and [FFmpeg](https://ffmpeg.org/) (manim also needs Cairo/Pango; see the
[manim install guide](https://docs.manim.community/en/stable/installation.html)).

```bash
conda create -n slides python=3.12 -y
conda activate slides
pip install -r requirements.txt
```

## Render

```bash
manim-slides render presentation.py Presentation        # renders all slides (takes a while)
```

## Present / export

```bash
manim-slides present Presentation                              # present in a window
manim-slides convert Presentation presentation.html            # HTML (videos in a folder next to it)
manim-slides convert Presentation presentation.html --one-file # single self-contained HTML (large)
manim-slides convert Presentation presentation.pptx            # PowerPoint (one video per slide)
```

## Layout

- `presentation.py` – all slides (one method per slide, called in order from `Presentation.construct`)
- `mathobj.py` – `MathParts`: LaTeX-style equations via matplotlib, split into named, colourable parts
- `assets/` – images (paper figure crops, photos) and icons
- `fonts/InstrumentSans.ttf` – slide font (SIL Open Font License)

## Credits

- Icons: [Font Awesome Free](https://fontawesome.com) (CC BY 4.0)
- Figure crops are from the cited papers and belong to their authors.
