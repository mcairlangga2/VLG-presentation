"""LaTeX-style equations for manim without a TeX install.

Uses matplotlib's mathtext engine to lay out the formula, then converts the glyph
outlines into manim VMobjects. The formula is given as a list of named parts; each
part becomes its own VMobject (so it can be coloured, boxed or animated separately),
while spacing is computed on the *whole* formula so it looks exactly like one equation.

    eq = MathParts([("max", r"\\mathrm{max}\\;"), ("E", r"\\mathbb{E}_{x}"), ...], size=0.5)
    eq["E"].set_color(BLUE)
"""
import numpy as np
from manim import VGroup, VMobject, WHITE
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath, TextToPath

FONTSET = "stix"  # "cm" (Computer Modern) or "stix" (Times-like, proper blackboard bold)
rcParams["mathtext.fontset"] = FONTSET

_PROP = FontProperties(size=1, math_fontfamily=FONTSET)
_T2P = TextToPath()


def _layout_width(tex):
    if not tex:
        return 0.0
    # a prefix may open \left[ without its \right]: close it with an invisible delimiter
    tex = tex + r"\right." * (tex.count(r"\left") - tex.count(r"\right"))
    w, _, _ = _T2P.get_text_width_height_descent("$" + tex + "$", _PROP, ismath=True)
    return w


def _contours(tex):
    """Split the formula's outline into closed contours (lists of (code, points))."""
    path = TextPath((0, 0), "$" + tex + "$", prop=_PROP)
    contours, cur = [], None
    verts, codes = path.vertices, path.codes
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == Path.MOVETO:
            cur = [("M", [verts[i]])]
            contours.append(cur)
            i += 1
        elif c == Path.LINETO:
            cur.append(("L", [verts[i]]))
            i += 1
        elif c == Path.CURVE3:
            cur.append(("Q", [verts[i], verts[i + 1]]))
            i += 2
        elif c == Path.CURVE4:
            cur.append(("C", [verts[i], verts[i + 1], verts[i + 2]]))
            i += 3
        elif c == Path.CLOSEPOLY:
            cur.append(("Z", []))
            i += 1
        else:  # STOP
            i += 1
    return contours


def _to_vmobject(contours, scale):
    def p(v):
        return np.array([v[0] * scale, v[1] * scale, 0.0])

    vm = VMobject()
    for contour in contours:
        start = None
        for kind, pts in contour:
            if kind == "M":
                start = p(pts[0])
                vm.start_new_path(start)
            elif kind == "L":
                vm.add_line_to(p(pts[0]))
            elif kind == "Q":
                vm.add_quadratic_bezier_curve_to(p(pts[0]), p(pts[1]))
            elif kind == "C":
                vm.add_cubic_bezier_curve_to(p(pts[0]), p(pts[1]), p(pts[2]))
            elif kind == "Z" and start is not None:
                if not np.allclose(vm.get_last_point(), start):
                    vm.add_line_to(start)
    vm.set_fill(WHITE, 1).set_stroke(width=0)
    return vm


class MathParts(VGroup):
    """A formula made of named parts. `size` = height of 1 em in manim units."""

    def __init__(self, parts, size=0.5, color=WHITE, **kw):
        super().__init__(**kw)
        names = [n for n, _ in parts]
        full = "".join(t for _, t in parts)
        # boundaries between parts, measured on the full layout
        bounds, acc = [], ""
        for _, t in parts[:-1]:
            acc += t
            bounds.append(_layout_width(acc))
        buckets = [[] for _ in parts]
        for c in _contours(full):
            xs = [pt[0] for _, pts in c for pt in pts]
            if not xs:
                continue
            cx = (min(xs) + max(xs)) / 2
            idx = int(np.searchsorted(bounds, cx))
            buckets[idx].append(c)
        self.names = names
        for n, b in zip(names, buckets):
            m = _to_vmobject(b, size) if b else VMobject()
            m.set_fill(color, 1)
            self.add(m)
        self._index = {n: i for i, n in enumerate(names)}
        self.center()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.submobjects[self._index[key]]
        return super().__getitem__(key)
