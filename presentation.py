from pathlib import Path

import sys

from manim import *
from manim_slides import Slide

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from mathobj import MathParts  # noqa: E402  (LaTeX-style equations without TeX)

ROOT = Path(__file__).parent
FONT_FILE = ROOT / "fonts" / "InstrumentSans.ttf"
FONT = "Instrument Sans"
ASSETS = ROOT / "assets"

# Left margin used by all slides (frame is 14.22 x 8 units)
LEFT_X = -config.frame_width / 2 + 1.7

# Palette (readable on manim's default black background)
C_PRETRAIN = "#5DADE2"  # blue  - knowledge from pretraining
C_NEW = "#F1948A"  # coral - knowledge the model never had
C_ACCENT = "#F7DC6F"  # yellow - "current" highlight
C_MUTED = GREY_B
C_PANEL = "#1C2833"  # dark fill for bubbles
C_GOAL = "#82E0AA"  # green - target model / downstream tasks
C_LINK = "#85C1E9"  # citation links
C_ICE = "#AED6F1"  # frozen modules
C_FIRE = "#F5B041"  # trainable modules / gradients
C_FIRE_FILL = "#3A2412"
C_S2 = "#EB984E"  # stage 2
C_S2_FILL = "#3A2616"


def txt(s, size=24, bold=False, color=WHITE):
    return Text(s, font=FONT, font_size=size, weight=BOLD if bold else NORMAL, color=color)


def lines(strings, align=LEFT, buff=0.08, **kw):
    """Multi-line text with controllable alignment (LEFT / RIGHT / ORIGIN=center)."""
    g = VGroup(*[txt(s, **kw) for s in strings])
    if np.array_equal(align, ORIGIN):
        return g.arrange(DOWN, buff=buff)
    return g.arrange(DOWN, aligned_edge=align, buff=buff)


def speech_bubble(content, tail_dir=UP, tail_at=None, pad=0.22, color=C_MUTED):
    """Rounded box around `content` with a small triangular tail pointing in `tail_dir`."""
    box = RoundedRectangle(
        corner_radius=0.18,
        width=content.width + 2 * pad,
        height=content.height + 2 * pad,
        fill_color=C_PANEL,
        fill_opacity=1,
        stroke_color=color,
        stroke_width=2,
    ).move_to(content)
    tail = Triangle(fill_color=C_PANEL, fill_opacity=1, stroke_color=color, stroke_width=2)
    tail.scale(0.12).rotate(angle_of_vector(tail_dir) - PI / 2)
    edge = box.get_edge_center(tail_dir) if tail_at is None else tail_at
    tail.next_to(edge, tail_dir, buff=-0.02)
    # cover the tail's base so it looks attached to the box
    cover = Line(
        tail.get_center() - rotate_vector(tail_dir, PI / 2) * 0.1,
        tail.get_center() + rotate_vector(tail_dir, PI / 2) * 0.1,
        stroke_color=C_PANEL,
        stroke_width=6,
    ).shift(-tail_dir * 0.07)
    return VGroup(tail, box, cover, content)


def icon(name, color, h=0.4):
    m = SVGMobject(str(ASSETS / f"{name}.svg")).set_height(h)
    return m.set_fill(color, 1).set_stroke(width=0)


def block(label_lines, w, h, trainable=False, size=20):
    """Model component box: ice-blue = frozen, orange = trainable."""
    color = C_FIRE if trainable else C_ICE
    box = RoundedRectangle(
        corner_radius=0.15, width=w, height=h,
        fill_color=C_FIRE_FILL if trainable else C_PANEL, fill_opacity=1,
        stroke_color=color, stroke_width=2,
    )
    label = lines(label_lines, align=ORIGIN, size=size, bold=True, color=color).move_to(box)
    return VGroup(box, label)


def chip(content, color, pad=0.15, fill=C_PANEL):
    box = RoundedRectangle(
        corner_radius=0.12, width=content.width + 2 * pad, height=content.height + 2 * pad,
        fill_color=fill, fill_opacity=1, stroke_color=color, stroke_width=1.5,
    ).move_to(content)
    return VGroup(box, content)


def id_flag(h=0.8):
    w = h * 1.5
    top = Rectangle(width=w, height=h / 2, stroke_width=0, fill_color="#CE1126", fill_opacity=1)
    bot = Rectangle(width=w, height=h / 2, stroke_width=0, fill_color=WHITE, fill_opacity=1)
    bot.next_to(top, DOWN, buff=0)
    frame = Rectangle(width=w, height=h, stroke_color=GREY_C, stroke_width=1.5).move_to(VGroup(top, bot))
    return VGroup(top, bot, frame)


def make_cite(s):
    c = txt(s, size=17, color=C_LINK).move_to([0, -3.5, 0])
    if c.width > 12.6:
        c.scale_to_fit_width(12.6)
    return VGroup(c, Underline(c, buff=0.05, color=C_LINK, stroke_width=1.5))


def up_arrow(start, end):
    return Arrow(start, end, buff=0.06, color=GREY_B, stroke_width=3,
                 max_tip_length_to_length_ratio=0.35)


def make_arch(col_x=-4.4):
    """X-ray -> vision encoder -> projection -> LLM, with frozen/trainable icons."""
    a = {}
    a["xray"] = ImageMobject(str(ASSETS / "xray.png")).set_height(1.15).move_to([col_x, -2.4, 0])
    a["enc"] = block(["Vision", "Encoder"], 1.7, 0.9).move_to([col_x, -0.95, 0])
    a["proj"] = block(["Projection"], 1.7, 0.48, trainable=True).move_to([col_x, 0.12, 0])
    a["llm"] = block(["Large Language Model"], 5.0, 0.9, size=22)
    a["llm"].move_to([0, 1.3, 0]).align_to(a["enc"], LEFT)
    a["ice_llm"] = icon("snowflake", C_ICE).next_to(a["llm"], LEFT, buff=0.2)
    a["fire"] = icon("fire", C_FIRE).next_to(a["proj"], LEFT, buff=0.2)
    a["ice_enc"] = icon("snowflake", C_ICE).next_to(a["enc"], LEFT, buff=0.2)
    a["a1"] = up_arrow(a["xray"].get_top(), a["enc"].get_bottom())
    a["a2"] = up_arrow(a["enc"].get_top(), a["proj"].get_bottom())
    a["a3"] = up_arrow(a["proj"].get_top(), [col_x, a["llm"].get_bottom()[1], 0])
    return a


def stage_panel(n, task, hours, data, color, fill):
    title = txt(f"Stage {n}", 20, True, color)
    title = VGroup(title, Underline(title, buff=0.05, color=color, stroke_width=1.5))
    arrow = Arrow(LEFT * 1.0, RIGHT * 1.0, buff=0, color=WHITE, stroke_width=3,
                  max_tip_length_to_length_ratio=0.12)
    body = VGroup(title, txt(task, 17), arrow, txt(f"{hours} Hours", 30, True),
                  txt(data, 15, color=GREY_A)).arrange(DOWN, buff=0.17)
    panel = RoundedRectangle(corner_radius=0.3, width=body.width + 0.45, height=2.55,
                             fill_color=fill, fill_opacity=1, stroke_color=color, stroke_width=2)
    body.move_to(panel)
    return VGroup(panel, *body)  # [panel, title, task, arrow, hours, data]


def downstream_panel(head=19, row=17, bullet=15):
    def checkbox_row(label):
        box = Square(side_length=0.18, stroke_color=WHITE, stroke_width=2)
        return VGroup(box, txt(label, size=row)).arrange(RIGHT, buff=0.13)

    def bullet_row(label):
        dot = Circle(radius=0.04, stroke_color=GREY_A, stroke_width=1.5)
        return VGroup(dot, txt(label, size=bullet, color=GREY_A)).arrange(RIGHT, buff=0.13)

    h = txt("Downstream", size=head, bold=True, color=C_GOAL)
    h = VGroup(h, Underline(h, buff=0.05, color=C_GOAL, stroke_width=1.5))
    rows = VGroup(checkbox_row("Medical Visual Chat"), checkbox_row("Medical VQA"),
                  bullet_row("VQA-Radiology"), bullet_row("SLAKE"), bullet_row("Pathology-VQA"),
                  ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    for r in rows[2:]:
        r.shift(RIGHT * 0.35)
    body = VGroup(h, rows).arrange(DOWN, buff=0.22)
    panel = RoundedRectangle(corner_radius=0.3, width=body.width + 0.45, height=2.55,
                             fill_color="#15261C", fill_opacity=1, stroke_color=C_GOAL, stroke_width=2)
    body.move_to(panel)
    return VGroup(panel, body)


def build_overview():
    """All objects of slide 2 (Augment vs. Knowledge Update), in their pre-animation state."""
    o = {}
    LABEL_Y = -2.85
    venn_y = 0.55
    o["blue"] = Ellipse(width=4.2, height=2.7, color=C_PRETRAIN, fill_opacity=0.25, stroke_width=0).move_to([-4.55, venn_y, 0])
    o["red"] = Ellipse(width=4.2, height=2.7, color=C_NEW, fill_opacity=0.25, stroke_width=0).move_to([-2.35, venn_y, 0])
    bt = lines(["Knowledge", "obtained during", "pretraining"], size=21, bold=True, color=C_PRETRAIN)
    o["blue_txt"] = bt.move_to(o["blue"]).align_to(o["blue"], LEFT).shift(RIGHT * 0.45).set_z_index(2)
    rt = lines(["Knowledge the", "model never had"], align=RIGHT, size=21, bold=True, color=C_NEW)
    o["red_txt"] = rt.move_to(o["red"]).align_to(o["red"], RIGHT).shift(LEFT * 0.4).set_z_index(2)
    o["augment_lbl"] = txt("Augment", size=32, bold=True).move_to([-3.45, LABEL_Y, 0])

    tl_y = 0.85
    tick_xs = [1.55, 3.75, 5.95]
    o["tl_y"], o["tick_xs"] = tl_y, tick_xs
    o["timeline"] = Arrow([0.75, tl_y, 0], [6.85, tl_y, 0], buff=0, color=C_PRETRAIN,
                          stroke_width=5, max_tip_length_to_length_ratio=0.04)
    o["ticks"] = VGroup(*[Line([x, tl_y - 0.22, 0], [x, tl_y + 0.22, 0], color=WHITE, stroke_width=5)
                          for x in tick_xs])
    o["year_lbls"] = VGroup(*[txt(y, size=28, bold=True).move_to([x, tl_y - 0.55, 0])
                              for x, y in zip(tick_xs, ["2023", "2024", "2025"])])
    o["photo"] = ImageMobject(str(ASSETS / "person_crop.png")).set_height(1.75).move_to([1.55, 2.55, 0])
    o["photo_frame"] = SurroundingRectangle(o["photo"], buff=0, color=GREY_C, stroke_width=2)
    q_text = lines(["What position does", "this person hold?"], align=ORIGIN, size=22, bold=True).move_to([4.85, 2.55, 0])
    o["question"] = speech_bubble(q_text, tail_dir=LEFT, pad=0.3, color=C_PRETRAIN)
    o["update_lbl"] = txt("Knowledge Update", size=32, bold=True).move_to([3.85, LABEL_Y, 0])

    o["marker"] = Dot([tick_xs[0], tl_y, 0], radius=0.13, color=C_ACCENT)
    o["robots"], o["answers"] = [], []
    for x, ans in zip(tick_xs, [["Former", "U.S. President"], ["President-elect"], ["U.S. President"]]):
        robot = SVGMobject(str(ASSETS / "robot.svg")).set_height(0.65)
        robot.set_fill(GREY_A, 1).set_stroke(width=0).move_to([x, -0.4, 0])
        t = lines(ans, align=ORIGIN, size=17, bold=True, buff=0.06).next_to(robot, DOWN, buff=0.45)
        o["robots"].append(robot)
        o["answers"].append(speech_bubble(t, tail_dir=UP, pad=0.16, color=C_ACCENT))
    return o


def overview_final_state(o):
    """Apply the look slide 2 has after its last click (2025 current, earlier years greyed)."""
    for i in (0, 1):
        o["robots"][i].set_fill(GREY_D)
        o["answers"][i][0].set_stroke(GREY_D)
        o["answers"][i][1].set_stroke(GREY_D)
        o["answers"][i][3].set_color(GREY_C)
        o["year_lbls"][i].set_color(GREY_C)
        o["ticks"][i].set_color(GREY_C)
    o["year_lbls"][2].set_color(C_ACCENT)
    o["ticks"][2].set_color(C_ACCENT)
    o["marker"].move_to([o["tick_xs"][2], o["tl_y"], 0])
    return o


class Presentation(Slide):
    def construct(self):
        with register_font(str(FONT_FILE)):
            self.title_slide()
            self.knowledge_slide()
            self.augment_transition()
            self.llava_med_slide()
            self.llava_med_goal()
            self.llava_med_stage1()
            self.stage1_detail()
            self.stages_overview()
            self.stage2_detail()
            self.sota_results()
            self.yollava_problem()
            self.yollava_training()
            self.yollava_results()
            self.update_transition()
            self.evolving_knowledge()
            self.adaptation_retention()
            self.injection_methods()
            self.benchmark_construction()
            self.metrics_and_results()
            self.degradation()
            self.catastrophic_forgetting()
            self.replay()
            self.kore_augmentation()
            self.kore_constraint()
            self.kore_results()
            self.editing_transition()
            self.rome_locate()
            self.rome_mlp()
            self.rome_memory()
            self.rome_lsq()
            self.rome_rank1()
            self.rome_key()
            self.rome_value()
            self.rome_pipeline()
            self.mm_locate()
            self.mm_multedit()
            self.mm_math()
            self.mm_eval()
            self.m2_intro()
            self.m2_metrics()
            self.m2_trace()
            self.m2_locate()
            self.m2_edit()
            self.recap()
            self.conclusion()
            # self.next_one()  # <- add your next slides as methods like this

    # ---------- shared helpers ----------
    def retitle(self, kicker_text, title, cite=None):
        """Fade everything except kicker/heading/citation; morph heading into `title`.
        The "LLaVA-Med" prefix (9 glyphs) morphs, the rest fades in/out."""
        lm = self.lm
        new_kicker = txt(kicker_text, size=18, bold=True, color=C_ACCENT)
        new_kicker.move_to(lm["kicker"]).align_to(lm["kicker"], LEFT)
        new_head = txt(title, size=56 if len(title) <= 9 else 40, bold=True)
        new_head.next_to(new_kicker, DOWN, buff=0.15, aligned_edge=LEFT)

        old = lm["head"]
        keep = {lm["kicker"], lm["cite"], *old}
        others = [m for m in self.mobjects if m not in keep]
        prefix, rest = new_head[:9], new_head[9:]
        anims = [
            *[FadeOut(m) for m in others],
            Transform(lm["kicker"], new_kicker),
            ReplacementTransform(old[0], prefix),
            *[FadeOut(m) for m in old[1:]],
        ]
        if cite is not None:
            anims.append(Transform(lm["cite"], make_cite(cite)))
        parts = [prefix]
        if len(rest):
            anims.append(FadeIn(rest, shift=LEFT * 0.3))
            parts.append(rest)
        self.play(*anims, run_time=1.2)
        lm["head"] = parts

    def build_arch(self, a):
        self.play(FadeIn(a["xray"], shift=UP * 0.2))
        self.play(GrowArrow(a["a1"]), FadeIn(a["enc"], shift=UP * 0.2), run_time=0.7)
        self.play(GrowArrow(a["a2"]), FadeIn(a["proj"], shift=UP * 0.2), run_time=0.7)
        self.play(GrowArrow(a["a3"]), FadeIn(a["llm"], shift=UP * 0.2), run_time=0.7)
        self.play(LaggedStart(FadeIn(a["ice_llm"], scale=1.6), FadeIn(a["ice_enc"], scale=1.6), lag_ratio=0.3))
        self.play(FadeIn(a["fire"], scale=1.6))
        self.play(Wiggle(a["fire"]))

    # ---------- Slide 1: Title ----------
    def title_slide(self):
        line1 = Text("Knowledge Enrichment in VLMs:", font=FONT, weight=BOLD, font_size=38.5)
        line2 = Text("Training & Editing", font=FONT, weight=BOLD, font_size=38.5)
        author = Text("Muhammad Cendekia Airlangga", font=FONT, font_size=24.5)

        title = VGroup(line1, line2).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        block = VGroup(title, author).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        block.move_to([0, 0.35, 0]).align_to([LEFT_X, 0, 0], LEFT)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(author, shift=UP * 0.2))
        self.next_slide()  # <- wait for a click

        self.play(FadeOut(block))

    # ---------- Slide 2: Augment vs. Knowledge Update ----------
    def knowledge_slide(self):
        o = build_overview()
        blue, red, blue_txt, red_txt = o["blue"], o["red"], o["blue_txt"], o["red_txt"]
        augment_lbl, update_lbl = o["augment_lbl"], o["update_lbl"]
        timeline, ticks, year_lbls = o["timeline"], o["ticks"], o["year_lbls"]
        photo, photo_frame, question = o["photo"], o["photo_frame"], o["question"]
        tick_xs, tl_y = o["tick_xs"], o["tl_y"]

        # Click 1: Augment label
        self.play(FadeIn(augment_lbl, shift=UP * 0.2))
        self.next_slide()

        # Click 2: pretraining knowledge
        self.play(FadeIn(blue, scale=0.9), run_time=0.8)
        self.play(FadeIn(blue_txt, shift=RIGHT * 0.2))
        self.next_slide()

        # Click 3: knowledge the model never had
        self.play(FadeIn(red, scale=0.9), run_time=0.8)
        self.play(FadeIn(red_txt, shift=LEFT * 0.2))
        self.next_slide()

        # Click 4: timeline + photo + question
        self.play(FadeIn(update_lbl, shift=UP * 0.2))
        self.play(GrowArrow(timeline), LaggedStart(*[Create(t) for t in ticks], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeIn(y, shift=UP * 0.15) for y in year_lbls], lag_ratio=0.2))
        self.play(FadeIn(photo), Create(photo_frame))
        self.play(FadeIn(question, shift=LEFT * 0.2))
        self.next_slide()

        # Clicks 5-7: each year's model answers with that year's fact
        marker = o["marker"]
        prev = None
        for i, x in enumerate(tick_xs):
            robot, answer = o["robots"][i], o["answers"][i]
            anims = [year_lbls[i].animate.set_color(C_ACCENT), ticks[i].animate.set_color(C_ACCENT)]
            if i == 0:
                anims.append(GrowFromCenter(marker))
            else:
                anims.append(marker.animate.move_to([x, tl_y, 0]))
                p_robot, p_answer, p_year, p_tick = prev
                anims += [
                    p_robot.animate.set_fill(GREY_D),
                    p_answer[1].animate.set_stroke(GREY_D),
                    p_answer[0].animate.set_stroke(GREY_D),
                    p_answer[3].animate.set_color(GREY_C),
                    p_year.animate.set_color(GREY_C),
                    p_tick.animate.set_color(GREY_C),
                ]
            self.play(*anims, run_time=0.8)
            self.play(FadeIn(robot, shift=UP * 0.25))
            self.play(GrowFromPoint(answer, robot.get_bottom()), run_time=0.6)
            self.play(Indicate(robot, color=C_ACCENT, scale_factor=1.1))

            prev = (robot, answer, year_lbls[i], ticks[i])
            self.next_slide()

        # keep handles for the transition that follows
        self.augment_venn = VGroup(blue, red, blue_txt, red_txt)
        self.augment_lbl = augment_lbl

    # ---------- Transition: zoom into "Augment" ----------
    def augment_transition(self):
        venn, lbl = self.augment_venn, self.augment_lbl
        keep = {*venn, lbl}
        others = [m for m in self.mobjects if m not in keep]

        # Knowledge Update half leaves, Augment half takes the stage
        self.play(
            *[FadeOut(m, shift=RIGHT * 0.6) for m in others],
            venn.animate.scale(1.2).move_to(UP * 0.6),
            lbl.animate.scale(1.2).move_to(DOWN * 1.75),
            run_time=1.2,
        )
        # Pop!
        self.play(lbl.animate.set_color(C_ACCENT), run_time=0.3)
        self.play(Indicate(lbl, scale_factor=1.35, color=C_ACCENT), run_time=0.8)
        self.next_slide()

    # ---------- Slide 3: LLaVA-Med ----------
    def llava_med_slide(self):
        venn, lbl = self.augment_venn, self.augment_lbl
        HEAD_X = -6.1

        kicker = txt("AUGMENT", size=18, bold=True, color=C_ACCENT)
        kicker.move_to([0, 3.4, 0]).align_to([HEAD_X, 0, 0], LEFT)
        heading = txt("LLaVA-Med", size=56, bold=True)
        heading.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)

        llava = ImageMobject(str(ASSETS / "llava_logo.png")).set_height(1.7)
        llava.move_to([-5.65, -0.25, 0])
        llava_lbl = txt("LLaVA", size=22, bold=True).next_to(llava, UP, buff=0.2)

        cite = txt(
            "LLaVA-Med: Training a Large Language-and-Vision Assistant "
            "for Biomedicine in One Day (Li et al., 2023)",
            size=17, color=C_LINK,
        ).move_to([0, -3.5, 0])
        cite = VGroup(cite, Underline(cite, buff=0.05, color=C_LINK, stroke_width=1.5))

        bubble_txt = lines(
            ["Never been trained in", "biomedical data!"],
            align=ORIGIN, size=25, bold=True, color=C_NEW,
        )
        bubble_txt.next_to(llava, RIGHT, buff=0.6)
        bubble = speech_bubble(bubble_txt, tail_dir=LEFT, pad=0.3, color=C_NEW)

        # Click: "Augment" label becomes the section tag, LLaVA-Med comes in
        self.play(
            FadeOut(venn, scale=0.6),
            ReplacementTransform(lbl, kicker),
            run_time=1,
        )
        self.play(Write(heading), FadeIn(cite, shift=UP * 0.2))
        self.play(FadeIn(llava, shift=UP * 0.2), FadeIn(llava_lbl, shift=UP * 0.2))
        self.next_slide()

        # Click: LLaVA "speaks"
        self.play(GrowFromPoint(bubble, llava.get_right()), run_time=0.6)
        self.play(Indicate(bubble_txt, color=C_NEW, scale_factor=1.06))
        self.next_slide()

        self.lm = dict(llava=llava, llava_lbl=llava_lbl, bubble=bubble,
                       kicker=kicker, heading=heading, cite=cite, head=[heading])

    # ---------- Slide 3 (cont.): the goal - LLaVA-Med + downstream tasks ----------
    def llava_med_goal(self):
        lm = self.lm

        # Downstream panel (right edge of the frame)
        def checkbox_row(label):
            box = Square(side_length=0.2, stroke_color=WHITE, stroke_width=2)
            return VGroup(box, txt(label, size=19)).arrange(RIGHT, buff=0.15)

        def bullet_row(label):
            dot = Circle(radius=0.045, stroke_color=GREY_A, stroke_width=1.5)
            return VGroup(dot, txt(label, size=17, color=GREY_A)).arrange(RIGHT, buff=0.15)

        head = txt("Downstream", size=22, bold=True, color=C_GOAL)
        head = VGroup(head, Underline(head, buff=0.05, color=C_GOAL, stroke_width=1.5))
        rows = VGroup(
            checkbox_row("Medical Visual Chat"),
            checkbox_row("Medical VQA"),
            bullet_row("VQA-Radiology"),
            bullet_row("SLAKE"),
            bullet_row("Pathology-VQA"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        for r in rows[2:]:
            r.shift(RIGHT * 0.4)
        body = VGroup(head, rows).arrange(DOWN, buff=0.25)
        panel = RoundedRectangle(
            corner_radius=0.3, width=body.width + 0.55, height=body.height + 0.5,
            fill_color="#15261C", fill_opacity=1, stroke_color=C_GOAL, stroke_width=2,
        )
        panel.move_to([0, -0.15, 0]).align_to([6.85, 0, 0], RIGHT)
        body.move_to(panel)

        llava_med = ImageMobject(str(ASSETS / "llava_med_logo.png")).set_height(1.7)
        llava_med.next_to(panel, LEFT, buff=0.3).match_y(lm["llava"])
        llava_med_lbl = txt("LLaVA-Med", size=22, bold=True).next_to(llava_med, UP, buff=0.2)

        # "?" bridge between the bubble and LLaVA-Med
        def bridge(x0, x1):
            y = lm["llava"].get_y()
            arrow = DashedLine([x0, y, 0], [x1, y, 0], dash_length=0.18, color=C_PRETRAIN, stroke_width=5)
            arrow.add_tip(tip_length=0.22, tip_width=0.22)
            q = txt("?", size=48, bold=True).next_to(arrow, UP, buff=0.15)
            return VGroup(arrow, q)

        gap_right = llava_med.get_left()[0] - 0.25
        q_bridge = bridge(lm["bubble"].get_right()[0] + 0.25, gap_right)

        # Click: the target model and what it should be able to do
        self.play(FadeIn(llava_med, shift=UP * 0.2), FadeIn(llava_med_lbl, shift=UP * 0.2))
        self.play(FadeIn(panel, scale=0.95), FadeIn(head))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.15))
        self.next_slide()

        # Click: how do we get from LLaVA to LLaVA-Med?
        self.play(Create(q_bridge[0]), run_time=0.8)
        self.play(FadeIn(q_bridge[1], scale=1.8), run_time=0.4)
        self.play(Wiggle(q_bridge[1]))
        self.next_slide()

        lm.update(llava_med=llava_med, panel=panel, rows=rows, q_bridge=q_bridge,
                  bridge=bridge, gap_right=gap_right)

    # ---------- Slide 3 (cont.): Stage 1 - medical concept alignment ----------
    def llava_med_stage1(self):
        lm = self.lm

        title = txt("Stage 1", size=22, bold=True, color=C_PRETRAIN)
        title = VGroup(title, Underline(title, buff=0.05, color=C_PRETRAIN, stroke_width=1.5))
        task = txt("Medical Concept Alignment", size=19)
        arrow = Arrow(LEFT * 1.1, RIGHT * 1.1, buff=0, color=WHITE, stroke_width=3,
                      max_tip_length_to_length_ratio=0.12)
        hours_slot = txt("7 Hours", size=34, bold=True)  # layout placeholder
        data = txt("1 epoch on 600K samples", size=17, color=GREY_A)
        body = VGroup(title, task, arrow, hours_slot, data).arrange(DOWN, buff=0.2)

        panel = RoundedRectangle(
            corner_radius=0.3, width=body.width + 0.55, height=lm["panel"].height,
            fill_color="#13253A", fill_opacity=1, stroke_color=C_PRETRAIN, stroke_width=2,
        )
        panel.next_to(lm["llava"], RIGHT, buff=0.3).match_y(lm["panel"])
        body.move_to(panel)

        # the "?" bridge shrinks into the gap that is still left (-> Stage 2)
        new_bridge = lm["bridge"](panel.get_right()[0] + 0.3, lm["gap_right"])

        # Click: Stage 1 replaces the complaint
        self.play(
            FadeOut(lm["bubble"], scale=0.8),
            GrowFromEdge(panel, LEFT),
            Transform(lm["q_bridge"], new_bridge),
            run_time=0.9,
        )
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(task, shift=DOWN * 0.1))

        # training "runs": arrow grows while the clock counts up to 7 hours
        hours = ValueTracker(0)
        clock = always_redraw(
            lambda: txt(f"{int(round(hours.get_value()))} Hours", size=34, bold=True).move_to(hours_slot)
        )
        self.add(clock)
        self.play(GrowArrow(arrow), hours.animate.set_value(7), run_time=1.8, rate_func=smooth)
        clock.clear_updaters()
        self.play(FadeIn(data, shift=UP * 0.1))
        self.play(Indicate(clock, color=C_PRETRAIN, scale_factor=1.1))
        self.next_slide()

        lm.update(stage1=VGroup(panel, title, task, arrow, clock, data))

    # ---------- Slide 4: Stage 1 in detail - Medical Concept Alignment ----------
    def stage1_detail(self):
        lm = self.lm
        col_x = -4.4  # vision column
        R = 3.6  # centre of the right half

        # ===== Transition: heading grows into the stage title =====
        self.retitle("AUGMENT · STAGE 1", "LLaVA-Med: Medical Concept Alignment")

        # ===== Click 1: architecture (bottom-up) =====
        arch = make_arch(col_x)
        self.build_arch(arch)
        xray, enc, proj, llm = arch["xray"], arch["enc"], arch["proj"], arch["llm"]
        ice_llm, fire, ice_enc = arch["ice_llm"], arch["fire"], arch["ice_enc"]
        self.next_slide()

        # ===== Click 2: where the training data comes from =====
        def info_chip(title, sub, color):
            g = VGroup(txt(title, 20, True, color), txt(sub, 14, color=GREY_A)).arrange(DOWN, buff=0.06)
            return chip(g, color, pad=0.18)

        pmc = info_chip("PMC-15M", "15M image–caption pairs", C_PRETRAIN).move_to([1.95, 1.55, 0])
        subset = info_chip("600K pairs", "1 epoch · 7 h on 8× A100", C_GOAL).move_to([5.3, 1.55, 0])
        filt = Arrow(pmc.get_right(), subset.get_left(), buff=0.1, color=GREY_B, stroke_width=3)
        filt_lbl = txt("filter", 14, color=GREY_B).next_to(filt, UP, buff=0.05)

        rule_head = txt("Instruction sampled by caption length", 15, True, GREY_B)
        rule_head.move_to([0, 0.65, 0]).align_to([0.6, 0, 0], LEFT)
        inst_short = chip(txt("\u201cDescribe the image concisely.\u201d", 15, True), C_ACCENT)
        inst_long = chip(txt("\u201cDescribe the following image in detail\u201d", 15, True, GREY_A), GREY_C)
        rows = VGroup()
        for y, cond, inst in [(0.1, "< 30 words", inst_short), (-0.5, "≥ 30 words", inst_long)]:
            c = txt(cond, 15, color=GREY_A).move_to([0, y, 0]).align_to([0.6, 0, 0], LEFT)
            inst.move_to([0, y, 0]).align_to([2.55, 0, 0], LEFT)
            a = Arrow([c.get_right()[0], y, 0], inst.get_left(), buff=0.12, color=GREY_B,
                      stroke_width=2.5, max_tip_length_to_length_ratio=0.25)
            rows.add(VGroup(c, a, inst))

        fmt = VGroup(
            Text("Human: <instruction> <image>", font="Menlo", font_size=15,
                 t2c={"<instruction>": C_ACCENT, "<image>": C_ICE}),
            Text("Assistant: <caption>", font="Menlo", font_size=15, t2c={"<caption>": C_NEW}),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        fmt_box = chip(fmt, GREY_C, pad=0.22).move_to([R, -1.85, 0])
        fmt_title = txt("one training example", 14, color=GREY_B).next_to(fmt_box, UP, buff=0.1, aligned_edge=LEFT)

        self.play(FadeIn(pmc, shift=LEFT * 0.2))
        self.play(GrowArrow(filt), FadeIn(filt_lbl), FadeIn(subset, shift=LEFT * 0.2))
        self.play(FadeIn(rule_head), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.3))
        self.play(FadeIn(fmt_title), FadeIn(fmt_box, shift=UP * 0.2))

        # our example caption is short -> the concise instruction goes into the model
        inst_pos = [-1.75, proj.get_y(), 0]
        sent = inst_short.copy()
        self.play(Indicate(inst_short, color=C_ACCENT, scale_factor=1.08))
        self.play(sent.animate.scale(0.9).move_to(inst_pos), run_time=1.0, path_arc=-0.5)
        a_inst = up_arrow(sent.get_top(), [sent.get_x(), llm.get_bottom()[1], 0])
        self.play(GrowArrow(a_inst), run_time=0.5)
        self.next_slide()

        # ===== Click 3: training step - only the projection learns =====
        right_data = VGroup(pmc, subset, filt, filt_lbl, rule_head, rows, fmt_box, fmt_title)
        self.play(FadeOut(right_data, shift=RIGHT * 0.3))

        target_lines = lines(
            ["Chest X-ray (PA view): clear", "lungs, no focal consolidation,", "effusion, or cardiomegaly."],
            size=17,
        )
        target_title = txt("target: the original caption", 14, True, C_NEW)
        target = VGroup(target_title, target_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        target_box = chip(target, C_NEW, pad=0.22)
        target_box.move_to([0, llm.get_y(), 0]).align_to([1.4, 0, 0], LEFT)
        pred_arrow = Arrow(llm.get_right(), target_box.get_left(), buff=0.1, color=GREY_B, stroke_width=3)
        pred_lbl = txt("predict", 14, color=GREY_B).next_to(pred_arrow, UP, buff=0.05)

        # forward pass
        fwd = VMobject().set_points_as_corners(
            [xray.get_top(), [col_x, llm.get_y(), 0], [llm.get_right()[0], llm.get_y(), 0]]
        )
        fwd_inst = Line(sent.get_top(), [sent.get_x(), llm.get_y(), 0])
        self.play(
            ShowPassingFlash(fwd.copy().set_stroke(WHITE, 7), time_width=0.4),
            ShowPassingFlash(fwd_inst.copy().set_stroke(C_ACCENT, 7), time_width=0.6),
            run_time=1.3,
        )
        self.play(GrowArrow(pred_arrow), FadeIn(pred_lbl), FadeIn(target_box[0]), FadeIn(target_title))
        self.play(Write(target_lines), run_time=1.5)

        # backward pass: gradient flows through the frozen LLM and stops at the projection
        loss_lbl = txt("loss vs. original caption", 15, True, C_FIRE)
        loss_lbl.next_to(target_box, DOWN, buff=0.2, aligned_edge=LEFT)
        bwd = VMobject().set_points_as_corners(
            [target_box.get_left() + LEFT * 0.1, [col_x, llm.get_y(), 0], proj.get_top()]
        )
        self.play(FadeIn(loss_lbl, shift=UP * 0.1))
        self.play(ShowPassingFlash(bwd.copy().set_stroke(C_FIRE, 8), time_width=0.5), run_time=1.5)
        self.play(
            proj[0].animate.set_fill(C_FIRE, 0.5),
            fire.animate.scale(1.5),
            Indicate(ice_llm, color=WHITE),
            Indicate(ice_enc, color=WHITE),
            run_time=0.6,
        )
        self.play(proj[0].animate.set_fill(C_FIRE_FILL, 1), fire.animate.scale(1 / 1.5), run_time=0.4)

        notes = VGroup(
            VGroup(icon("fire", C_FIRE, 0.3), txt("Only the projection is updated", 18, True, C_FIRE)).arrange(RIGHT, buff=0.15),
            VGroup(icon("snowflake", C_ICE, 0.3), txt("Vision encoder & LLM stay frozen", 18, True, C_ICE)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        notes.next_to(loss_lbl, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.15) for n in notes], lag_ratio=0.3))
        self.next_slide()

        # ===== Click 4: what is being learned - the embedding space =====
        self.play(FadeOut(VGroup(target_box, pred_arrow, pred_lbl, loss_lbl, notes)))

        space = RoundedRectangle(
            corner_radius=0.25, width=6.3, height=4.6,
            fill_color="#0E1116", fill_opacity=1, stroke_color=GREY_D, stroke_width=1.5,
        ).move_to([3.65, -0.55, 0])
        space_title = txt("LLM token-embedding space", 16, True, GREY_B)
        space_title.move_to(space.get_corner(UL) + [0.25, -0.3, 0], aligned_edge=UL)
        c = space.get_center()

        def P(x, y):
            return c + np.array([x, y, 0])

        general = {"dog": (-2.2, 0.9), "car": (-1.0, 1.4), "tree": (-1.1, 0.2), "pizza": (-2.3, -0.4)}
        medical = {"lung": (0.9, 1.1), "effusion": (2.1, 0.55), "cardiomegaly": (0.7, -0.15), "nodule": (2.0, -0.7)}
        feat_off = np.array([-0.25, -0.2, 0])

        words = VGroup()
        for w, (x, y) in {**general, **medical}.items():
            d = Dot(P(x, y), radius=0.06, color=WHITE)
            words.add(VGroup(d, txt(w, 15, color=GREY_A).next_to(d, RIGHT, buff=0.08)))

        gen_feats = VGroup(*[
            Square(0.17, fill_color=C_PRETRAIN, fill_opacity=1, stroke_width=0).move_to(P(x, y) + feat_off)
            for x, y in general.values()
        ])
        gen_lbl = lines(["general image features", "(already aligned by LLaVA)"], size=13, color=C_PRETRAIN, buff=0.04)
        gen_lbl.move_to(P(-1.75, -1.15))

        cluster = [(-0.2, -1.45), (0.25, -1.85), (0.55, -1.4), (0.0, -1.9)]
        med_feats = VGroup(*[
            Square(0.17, fill_color=C_NEW, fill_opacity=1, stroke_width=0).move_to(proj)
            for _ in cluster
        ])
        med_lbl = lines(["biomedical image features", "(not aligned yet)"], size=13, color=C_NEW, buff=0.04)
        med_lbl.next_to(P(0.8, -1.65), RIGHT, buff=0.1)

        self.play(FadeIn(space), FadeIn(space_title))
        self.play(LaggedStart(*[FadeIn(w, scale=0.8) for w in words], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromCenter(f) for f in gen_feats], lag_ratio=0.1), FadeIn(gen_lbl))
        # biomedical features come out of the (untrained) projection and land nowhere useful
        self.play(
            LaggedStart(*[f.animate.move_to(P(x, y)) for f, (x, y) in zip(med_feats, cluster)], lag_ratio=0.15),
            run_time=1.4,
        )
        self.play(FadeIn(med_lbl))
        self.next_slide()

        # ===== Click 5: Stage 1 aligns them =====
        self.play(proj[0].animate.set_fill(C_FIRE, 0.5), fire.animate.scale(1.5), run_time=0.4)
        self.play(
            *[f.animate.move_to(P(x, y) + feat_off) for f, (x, y) in zip(med_feats, medical.values())],
            FadeOut(med_lbl),
            proj[0].animate.set_fill(C_FIRE_FILL, 1),
            fire.animate.scale(1 / 1.5),
            run_time=1.8,
            path_arc=0.6,
        )
        self.play(LaggedStart(*[Indicate(w, color=C_NEW, scale_factor=1.15) for w in words[len(general):]], lag_ratio=0.15))

        tagline = lines(
            ["Stage 1 expands the aligned image–text", "vocabulary to the biomedical domain"],
            align=ORIGIN, size=17, bold=True, color=C_GOAL, buff=0.06,
        ).move_to(P(0.75, -1.9))
        self.play(FadeIn(tagline, shift=UP * 0.15))
        self.next_slide()

    # ---------- Slide 5: back to the overview - Stage 2 fills the gap ----------
    def stages_overview(self):
        lm = self.lm
        self.retitle("AUGMENT", "LLaVA-Med")

        Y = -0.15
        llava = ImageMobject(str(ASSETS / "llava_logo.png")).set_height(1.5)
        s1 = stage_panel(1, "Medical Concept Alignment", 7, "1 epoch on 600K samples", C_PRETRAIN, "#13253A")
        s2 = stage_panel(2, "Medical Instruction Tuning", 8, "3 epochs on 60K samples", C_S2, C_S2_FILL)
        llava_med = ImageMobject(str(ASSETS / "llava_med_logo.png")).set_height(1.5)
        down = downstream_panel()
        Group(llava, s1, s2, llava_med, down).arrange(RIGHT, buff=0.22).move_to([0, Y, 0])
        llava.set_y(Y - 0.1)
        llava_med.set_y(Y - 0.1)
        llava_lbl = txt("LLaVA", 20, True).next_to(llava, UP, buff=0.18)
        llava_med_lbl = txt("LLaVA-Med", 20, True).next_to(llava_med, UP, buff=0.18)

        y = llava.get_y()
        q_line = DashedLine([s1.get_right()[0] + 0.25, y, 0], [llava_med.get_left()[0] - 0.25, y, 0],
                            dash_length=0.18, color=C_PRETRAIN, stroke_width=5)
        q_line.add_tip(tip_length=0.22, tip_width=0.22)
        q = VGroup(q_line, txt("?", 48, True).next_to(q_line, UP, buff=0.15))

        # (same click as the retitle) restore the picture, now with room for Stage 2
        self.play(
            LaggedStart(
                FadeIn(Group(llava, llava_lbl), shift=UP * 0.2),
                FadeIn(s1, shift=UP * 0.2),
                FadeIn(q),
                FadeIn(Group(llava_med, llava_med_lbl), shift=UP * 0.2),
                FadeIn(down, shift=UP * 0.2),
                lag_ratio=0.2,
            ),
            run_time=1.8,
        )
        self.next_slide()

        # Click: Stage 2 replaces the "?"
        panel, title, task, arrow, hours_slot, data = s2
        self.play(FadeOut(q, scale=0.8), GrowFromEdge(panel, LEFT), run_time=0.9)
        self.play(FadeIn(title, shift=DOWN * 0.1), FadeIn(task, shift=DOWN * 0.1))
        hours = ValueTracker(0)
        clock = always_redraw(
            lambda: txt(f"{int(round(hours.get_value()))} Hours", 30, True).move_to(hours_slot)
        )
        self.add(clock)
        self.play(GrowArrow(arrow), hours.animate.set_value(8), run_time=1.8, rate_func=smooth)
        clock.clear_updaters()
        self.play(FadeIn(data, shift=UP * 0.1))
        self.play(Indicate(clock, color=C_S2, scale_factor=1.1))
        self.next_slide()

        # Click: the whole curriculum fits in a day
        brace = Brace(VGroup(s1, s2[0]), DOWN, buff=0.15, color=GREY_B)
        brace_txt = VGroup(
            txt("< 15 hours in total on 8× A100  →  \u201cin One Day\u201d", 20, True, C_ACCENT),
            txt("curriculum: biomedical concepts first, then open-ended conversation", 15, color=GREY_A),
        ).arrange(DOWN, buff=0.1).next_to(brace, DOWN, buff=0.15)
        self.play(GrowFromCenter(brace), FadeIn(brace_txt, shift=UP * 0.15))
        self.next_slide()

    # ---------- Slide 6: Stage 2 in detail - Medical Instruction Tuning ----------
    def stage2_detail(self):
        col_x = -4.4
        R = 3.6
        self.retitle("AUGMENT · STAGE 2", "LLaVA-Med: Medical Instruction Tuning")

        # ===== Click 1: same model ... but the LLM thaws =====
        arch = make_arch(col_x)
        self.build_arch(arch)
        xray, enc, proj, llm = arch["xray"], arch["enc"], arch["proj"], arch["llm"]
        llm_icon, fire, ice_enc = arch["ice_llm"], arch["fire"], arch["ice_enc"]

        hot_llm = block(["Large Language Model"], 5.0, 0.9, trainable=True, size=22).move_to(llm)
        llm_fire = icon("fire", C_FIRE).move_to(llm_icon)
        thaw = txt("unfrozen in Stage 2", 15, True, C_FIRE).next_to(llm, UP, buff=0.12, aligned_edge=RIGHT)
        self.play(Transform(llm, hot_llm), Transform(llm_icon, llm_fire), FadeIn(thaw, shift=DOWN * 0.1), run_time=1)
        self.play(Wiggle(llm_icon))
        self.next_slide()

        # ===== Click 2: how the instruction data is made (GPT-4, text only) =====
        stat = chip(VGroup(
            txt("PMC-15M  →  60K image–text pairs", 17, True, C_S2),
            txt("CXR · CT · MRI · histopathology · gross pathology", 14, color=GREY_A),
        ).arrange(DOWN, buff=0.06), C_S2, pad=0.15).move_to([R, 1.72, 0])

        def card(title, body_lines, color):
            t = txt(title, 14, True, color)
            b = lines(body_lines, size=13, color=GREY_A, buff=0.05)
            return chip(VGroup(t, b).arrange(DOWN, aligned_edge=LEFT, buff=0.1), color, pad=0.14)

        cap = card("Caption", [
            "Contrast-enhanced CT scan of the",
            "chest for patient #1. A large",
            "cavitating lesion was identified in",
            "the right upper lobe with pleural…",
        ], C_PRETRAIN)
        mentions = card("In-line mentions (paper text)", [
            "\u201c…demonstrated a large cavitating",
            "lesion in the posterior aspect of the",
            "right upper lobe with probable",
            "pleural involvement…\u201d",
        ], C_PRETRAIN)
        cards = VGroup(cap, mentions).arrange(RIGHT, buff=0.2, aligned_edge=UP).move_to([R, 0.5, 0])

        gpt = chip(VGroup(icon("eye-slash", GREY_B, 0.24), txt("GPT-4 (text only)", 17, True)).arrange(RIGHT, buff=0.15),
                   WHITE, pad=0.14).move_to([R - 0.7, -0.78, 0])
        gpt_note = txt("never sees the image", 14, color=GREY_B).next_to(gpt, RIGHT, buff=0.2)
        in_arrows = VGroup(*[
            Arrow(c.get_bottom(), gpt.get_top() + RIGHT * (0.5 if i else -0.5), buff=0.06, color=GREY_B,
                  stroke_width=2.5, tip_length=0.14)
            for i, c in enumerate(cards)
        ])

        conv_src = [
            ("User: Where is the lesion located?", 0),
            ("Assistant: The lesion is located in the posterior aspect of the right upper", 0),
            ("lobe of the lung.", 1),
            ("User: Is it extending into the chest wall?", 0),
            ("Assistant: Yes, the CT scan suggests that the large cavitating lesion in the", 0),
            ("right upper lobe is likely extending into the chest wall.", 1),
        ]
        conv_lines = VGroup(*[
            Text(t, font=FONT, font_size=13, color=GREY_A, t2c={"User:": C_ACCENT, "Assistant:": C_GOAL})
            for t, _ in conv_src
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        for l, (_, cont) in zip(conv_lines, conv_src):
            if cont:
                l.shift(RIGHT * 0.72)
        conv_title = txt("Generated multi-turn conversation", 14, True, C_GOAL)
        conv = chip(VGroup(conv_title, conv_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.12), C_GOAL, pad=0.15)
        conv.move_to([R, -2.1, 0])
        out_arrow = Arrow(gpt.get_bottom(), [gpt.get_x(), conv.get_top()[1], 0], buff=0.04, color=GREY_B,
                          stroke_width=2.5, tip_length=0.14)

        self.play(FadeIn(stat, shift=DOWN * 0.15))
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in cards], lag_ratio=0.3))
        self.play(LaggedStart(*[GrowArrow(a) for a in in_arrows], lag_ratio=0.1), FadeIn(gpt, scale=0.9))
        self.play(FadeIn(gpt_note, shift=LEFT * 0.1), Wiggle(gpt[1][0]))
        self.play(GrowArrow(out_arrow), FadeIn(conv[0]), FadeIn(conv_title))
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in conv_lines], lag_ratio=0.25), run_time=2)
        self.next_slide()

        # ===== Click 3: train on the whole conversation - LLM + projection learn =====
        seq_lines = VGroup(*[
            Text(t, font=FONT, font_size=14, weight=BOLD, color=WHITE,
                 t2c={"Q1:": C_ACCENT, "Q2:": C_ACCENT, "A1:": C_GOAL, "A2:": C_GOAL})
            for t in ["Q1: What type of image is this?", "A1: A frontal chest X-ray.", "Q2: …", "A2: …"]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        seq_title = txt("GPT-4 conversation, in one sequence", 14, True, C_GOAL)
        seq = chip(VGroup(seq_title, seq_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.12), C_GOAL, pad=0.15)
        seq.align_to([proj.get_right()[0] + 0.2, 0, 0], LEFT).align_to([0, 0.4, 0], UP)
        a_seq = up_arrow(seq.get_top(), [seq.get_x(), llm.get_bottom()[1], 0])

        self.play(FadeOut(VGroup(stat, cards, in_arrows, gpt, gpt_note, out_arrow, thaw)))
        self.play(ReplacementTransform(conv, seq), run_time=1.2)
        self.play(GrowArrow(a_seq), run_time=0.5)

        target_lines = lines(["A1: A frontal chest X-ray.", "A2: …"], size=17)
        target_title = txt("target: the assistant's answers", 14, True, C_NEW)
        target = VGroup(target_title, target_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        target_box = chip(target, C_NEW, pad=0.22)
        target_box.move_to([0, llm.get_y(), 0]).align_to([1.4, 0, 0], LEFT)
        pred_arrow = Arrow(llm.get_right(), target_box.get_left(), buff=0.1, color=GREY_B, stroke_width=3)
        pred_lbl = txt("predict", 14, color=GREY_B).next_to(pred_arrow, UP, buff=0.05)

        fwd = VMobject().set_points_as_corners(
            [xray.get_top(), [col_x, llm.get_y(), 0], [llm.get_right()[0], llm.get_y(), 0]]
        )
        fwd_seq = Line(seq.get_top(), [seq.get_x(), llm.get_y(), 0])
        self.play(
            ShowPassingFlash(fwd.copy().set_stroke(WHITE, 7), time_width=0.4),
            ShowPassingFlash(fwd_seq.copy().set_stroke(C_GOAL, 7), time_width=0.6),
            run_time=1.3,
        )
        self.play(GrowArrow(pred_arrow), FadeIn(pred_lbl), FadeIn(target_box, shift=LEFT * 0.1))

        loss_lbl = txt("loss vs. the generated answers", 15, True, C_FIRE)
        loss_lbl.next_to(target_box, DOWN, buff=0.2, aligned_edge=LEFT)
        bwd = VMobject().set_points_as_corners(
            [target_box.get_left() + LEFT * 0.1, [col_x, llm.get_y(), 0], proj.get_top()]
        )
        self.play(FadeIn(loss_lbl, shift=UP * 0.1))
        self.play(ShowPassingFlash(bwd.copy().set_stroke(C_FIRE, 8), time_width=0.5), run_time=1.5)
        self.play(
            llm[0].animate.set_fill(C_FIRE, 0.5), proj[0].animate.set_fill(C_FIRE, 0.5),
            llm_icon.animate.scale(1.5), fire.animate.scale(1.5),
            Indicate(ice_enc, color=WHITE), run_time=0.6,
        )
        self.play(
            llm[0].animate.set_fill(C_FIRE_FILL, 1), proj[0].animate.set_fill(C_FIRE_FILL, 1),
            llm_icon.animate.scale(1 / 1.5), fire.animate.scale(1 / 1.5), run_time=0.4,
        )
        notes = VGroup(
            VGroup(icon("fire", C_FIRE, 0.3), txt("Projection + LLM are updated", 18, True, C_FIRE)).arrange(RIGHT, buff=0.15),
            VGroup(icon("snowflake", C_ICE, 0.3), txt("Only the vision encoder stays frozen", 18, True, C_ICE)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        notes.next_to(loss_lbl, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.15) for n in notes], lag_ratio=0.3))
        self.next_slide()

        # ===== Click 4: does Stage 2 matter? (paper, Table 1) =====
        self.play(FadeOut(VGroup(target_box, pred_arrow, pred_lbl, loss_lbl, notes)))

        chart_title = txt("Does Stage 2 matter?", 22, True).move_to([0, 1.75, 0]).align_to([0.6, 0, 0], LEFT)
        chart_sub = txt("GPT-4-judged chat quality, relative score (Table 1)", 14, color=GREY_B)
        chart_sub.next_to(chart_title, DOWN, buff=0.1, aligned_edge=LEFT)

        x0, unit = 2.75, 3.6 / 60  # bar origin and units per score point
        rows = [("LLaVA", 36.1, GREY_C, 0.65), ("Stage 1 only", 23.3, GREY_C, -0.25), ("Stage 1 + 2", 50.2, C_S2, -1.15)]
        axis = Line([x0, 1.0, 0], [x0, -1.5, 0], color=GREY_D, stroke_width=2)
        bars, vals, names = [], [], []
        for name, v, col, y in rows:
            names.append(txt(name, 17, True, WHITE if name == "Stage 1 + 2" else GREY_A)
                         .move_to([0, y, 0]).align_to([x0 - 0.15, 0, 0], RIGHT))
            bars.append(RoundedRectangle(corner_radius=0.06, width=v * unit, height=0.42,
                                         fill_color=col, fill_opacity=1, stroke_width=0)
                        .move_to([x0 + v * unit / 2, y, 0]))
            vals.append(txt(f"{v:.1f}", 17, True).next_to(bars[-1], RIGHT, buff=0.12))
        drop_note = txt("captioning only → loses instruction following", 13, color=GREY_B)
        drop_note.next_to(bars[1], DOWN, buff=0.08, aligned_edge=LEFT).shift(RIGHT * 0.12)

        self.play(FadeIn(chart_title, shift=DOWN * 0.1), FadeIn(chart_sub), Create(axis))
        for i in (0, 1):
            self.play(FadeIn(names[i]), GrowFromEdge(bars[i], LEFT), run_time=0.8)
            self.play(FadeIn(vals[i], shift=LEFT * 0.1), run_time=0.3)
        self.play(FadeIn(drop_note, shift=UP * 0.1))
        self.next_slide()

        # ===== Click 5: ... with Stage 2 it becomes an assistant =====
        self.play(FadeIn(names[2]), GrowFromEdge(bars[2], LEFT), run_time=1.0)
        self.play(FadeIn(vals[2], shift=LEFT * 0.1), Indicate(bars[2], color=C_S2, scale_factor=1.05))
        quote = lines(
            ["\u201c…the model\u2019s behaviors transit from a layperson", "to a professional assistant\u201d"],
            align=ORIGIN, size=17, bold=True, color=C_GOAL, buff=0.06,
        ).move_to([R, -2.35, 0])
        self.play(FadeIn(quote, shift=UP * 0.15))
        self.next_slide()

    # ---------- Slide 7: comparison with SoTA (paper, Table 4a) ----------
    def sota_results(self):
        self.retitle("AUGMENT · RESULTS", "LLaVA-Med: Comparison with SoTA")

        # Table 4(a). Prior methods' open-set numbers come from the paper's "Ref" column
        # (classification over training answers) -> shown in the Open column with a dagger.
        ours = [
            ("LLaVA", [50.00, 65.07, 78.18, 63.22, 7.74, 63.20]),
            ("LLaVA-Med (From LLaVA)", [61.52, 84.19, 83.08, 85.34, 37.95, 91.21]),
            ("LLaVA-Med (From Vicuna)", [64.39, 81.98, 84.71, 83.17, 38.87, 91.65]),
            ("LLaVA-Med (BioMed CLIP)", [64.75, 83.09, 87.11, 86.78, 39.60, 91.09]),
        ]
        prior = [
            ("VL Encoder–Decoder", [71.49, 82.47, None, None, 71.49, 85.61]),
            ("Q2ATransformer", [79.19, 81.20, None, None, 54.85, 88.85]),
            ("Prefix T. Medical LM", [None, None, 84.30, 82.01, 40.00, 87.00]),
            ("PubMedCLIP", [60.10, 80.00, 78.40, 82.50, None, None]),
            ("BiomedCLIP", [67.60, 79.80, 82.05, 89.70, None, None]),
            ("M2I2", [66.50, 83.50, 74.70, 91.10, 36.30, 88.00]),
        ]
        NAME_X = -6.35
        col_x = [-2.55 + 0.98 * i for i in range(6)]
        STEP = 0.34
        SIZE = 15

        # header
        groups = VGroup(*[
            txt(g, SIZE + 1, True).move_to([(col_x[2 * i] + col_x[2 * i + 1]) / 2, 2.05, 0])
            for i, g in enumerate(["VQA-RAD", "SLAKE", "PathVQA"])
        ])
        subs = VGroup(*[
            txt("Open" if i % 2 == 0 else "Closed", SIZE - 1, color=GREY_B).move_to([x, 1.72, 0])
            for i, x in enumerate(col_x)
        ])
        method_hdr = txt("Method", SIZE + 1, True).move_to([0, 1.72, 0]).align_to([NAME_X, 0, 0], LEFT)
        top_rule = Line([NAME_X, 1.52, 0], [col_x[-1] + 0.55, 1.52, 0], color=GREY_C, stroke_width=1.5)
        seps = VGroup(*[
            Line([x, 2.2, 0], [x, -2.6, 0], color=GREY_D, stroke_width=1)
            for x in [(col_x[1] + col_x[2]) / 2, (col_x[3] + col_x[4]) / 2]
        ])

        def section(label, y):
            return txt(label, SIZE - 2, True, GREY_B).move_to([0, y, 0]).align_to([NAME_X, 0, 0], LEFT)

        def make_rows(data, y0, highlight, dagger):
            name_color = C_GOAL if highlight else GREY_A
            rows = []
            for r, (name, vals) in enumerate(data):
                y = y0 - r * STEP
                nm = txt(name, SIZE, highlight, name_color).move_to([0, y, 0]).align_to([NAME_X, 0, 0], LEFT)
                cells = []
                for c, v in enumerate(vals):
                    if v is None:
                        cell = txt("–", SIZE, color=GREY_D)
                    else:
                        label = f"{v:.2f}" + ("†" if dagger and c % 2 == 0 else "")
                        cell = txt(label, SIZE, color=WHITE if highlight else GREY_A)
                    cells.append(cell.move_to([col_x[c], y, 0]))
                rows.append((nm, cells))
            return rows

        sec1 = section("Supervised fine-tuning (paper's own runs)", 1.3)
        our_rows = make_rows(ours[:1], 0.98, False, False) + make_rows(ours[1:], 0.98 - STEP, True, False)
        mid_rule = Line([NAME_X, -0.2, 0], [col_x[-1] + 0.55, -0.2, 0], color=GREY_D, stroke_width=1)
        sec2 = section("Representative & SoTA methods (reported in literature)", -0.4)
        prior_rows = make_rows(prior, -0.74, False, True)
        foot = txt("† prior methods score open questions as classification over training-set answers "
                   "(may overestimate); LLaVA-Med reports recall of free-form answers",
                   11, color=GREY_B).move_to([0, -2.9, 0]).align_to([NAME_X, 0, 0], LEFT)

        def row_group(rows):
            return VGroup(*[VGroup(nm, *cells) for nm, cells in rows])

        # ===== (same click as retitle) build the table =====
        self.play(FadeIn(groups), FadeIn(subs), FadeIn(method_hdr), Create(top_rule), Create(seps))
        self.play(FadeIn(sec1))
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.15) for g in row_group(our_rows)], lag_ratio=0.15))
        self.play(Create(mid_rule), FadeIn(sec2))
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.15) for g in row_group(prior_rows)], lag_ratio=0.12))
        self.play(FadeIn(foot))
        self.next_slide()

        CALL_X = 3.45

        def callout(title, body, color, y_top):
            t = txt(title, 18, True, color)
            b = lines(body, size=14, color=GREY_A, buff=0.05)
            g = VGroup(t, b).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            return g.move_to([0, 0, 0]).align_to([CALL_X, y_top, 0], UL)

        def box(m, color):
            return SurroundingRectangle(m, buff=0.06, corner_radius=0.06, color=color, stroke_width=2.5)

        # ===== Click: LLaVA-Med >> LLaVA =====
        llava_box = box(row_group(our_rows[:1]), GREY_B)
        med_box = box(row_group(our_rows[1:]), C_GOAL)
        c1 = callout("LLaVA-Med ≫ LLaVA", ["every variant beats LLaVA", "e.g. PathVQA open: 7.74 → 39.60"], C_GOAL, 2.1)
        self.play(Create(llava_box), Create(med_box))
        self.play(Indicate(our_rows[0][1][4], color=GREY_B), Indicate(our_rows[3][1][4], color=C_GOAL))
        self.play(FadeIn(c1, shift=LEFT * 0.15))
        self.next_slide()

        all_rows = our_rows + prior_rows

        def column(c):
            return VGroup(*[cells[c] for _, cells in all_rows])

        def winner(c):
            vals = [(v[c], i) for i, (_, v) in enumerate(ours[1:] + prior) if v[c] is not None]
            best = max(vals)[1]
            rows = our_rows[1:] + prior_rows
            return rows[best][1][c], best < 3  # (cell, is LLaVA-Med)

        def focus(kind):
            """kind 1 = closed columns, 0 = open columns."""
            dim = [column(c) for c in range(6) if c % 2 != kind]
            lit = [column(c) for c in range(6) if c % 2 == kind]
            boxes = VGroup()
            for c in range(kind, 6, 2):
                cell, ours_wins = winner(c)
                boxes.add(box(cell, C_ACCENT if ours_wins else GREY_B))
            return dim, lit, boxes

        # ===== Click: closed-set questions =====
        dim, lit, closed_boxes = focus(1)
        c2 = callout("Closed-set (yes / no)", ["new SoTA on VQA-RAD & PathVQA:", "with clear answer options it",
                                                "follows instructions well"], C_ACCENT, 0.95)
        self.play(FadeOut(llava_box), FadeOut(med_box),
                  *[m.animate.set_opacity(0.25) for m in dim], *[m.animate.set_opacity(1) for m in lit])
        self.play(LaggedStart(*[Create(b) for b in closed_boxes], lag_ratio=0.3))
        self.play(FadeIn(c2, shift=LEFT * 0.15))
        self.next_slide()

        # ===== Click: open-set questions =====
        dim, lit, open_boxes = focus(0)
        c3 = callout("Open-set", ["SoTA on SLAKE only;", "behind on VQA-RAD & PathVQA:",
                                   "open biomedical questions are", "ambiguous without answer options"],
                     C_NEW, -0.35)
        self.play(FadeOut(closed_boxes),
                  *[m.animate.set_opacity(0.25) for m in dim], *[m.animate.set_opacity(1) for m in lit])
        self.play(LaggedStart(*[Create(b) for b in open_boxes], lag_ratio=0.3))
        self.play(FadeIn(c3, shift=LEFT * 0.15))
        self.next_slide()

    # ---------- Slide 8: Yo'LLaVA - the personalization problem ----------
    def yollava_problem(self):
        self.retitle("AUGMENT", "Yo\u2019LLaVA",
                     cite="Yo\u2019LLaVA: Your Personalized Language and Vision Assistant (Nguyen et al., 2024)")
        C_TOK = C_S2  # personalized concept token <bo>
        CX = -4.4  # left (context) column

        def chat_lines(strs, size=18, t2c=None, color=WHITE):
            return VGroup(*[
                Text(t, font=FONT, font_size=size, color=color, t2c=t2c or {}) for t in strs
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        # --- the personal concept: <bo> = my dog
        tok = Text("<bo>", font="Menlo", font_size=30, weight=BOLD, color=C_TOK)
        tok_desc = txt("= my dog", 22, True)
        tok_row = VGroup(tok, tok_desc).arrange(RIGHT, buff=0.2)
        bo_pics = Group(*[
            ImageMobject(str(ASSETS / f"bo_{i}.png")).set_height(1.05) for i in (1, 2)
        ]).arrange(RIGHT, buff=0.15)
        concept = Group(tok_row, bo_pics).arrange(DOWN, buff=0.22).move_to([CX, 1.0, 0])
        concept_lbl = txt("personalized concept", 14, color=GREY_B).next_to(concept, UP, buff=0.12)

        # --- the user
        user = icon("circle-user", GREY_A, 0.6).move_to([-1.15, 1.75, 0])
        user_lbl = txt("me", 14, color=GREY_B).next_to(user, DOWN, buff=0.08)
        intro = txt("Suppose we want a personalized LLaVA\u2026", 22, True, GREY_A)
        intro.next_to(user, RIGHT, buff=0.35)

        # (same click as retitle)
        self.play(FadeIn(concept_lbl), FadeIn(tok_row, shift=DOWN * 0.1),
                  LaggedStart(*[FadeIn(p, scale=0.9) for p in bo_pics], lag_ratio=0.3))
        self.play(FadeIn(user, scale=0.8), FadeIn(user_lbl))
        self.play(Write(intro), run_time=1)
        self.next_slide()

        # ===== Click: the question (with a photo) =====
        photo = ImageMobject(str(ASSETS / "yollava_input.png")).set_height(2.2).move_to([CX, -1.55, 0])
        photo_frame = SurroundingRectangle(photo, buff=0, color=GREY_C, stroke_width=2)
        photo_lbl = txt("input image", 14, color=GREY_B).next_to(photo, DOWN, buff=0.1)
        q_text = Text("Hey, can you see if <bo> is in this photo?", font=FONT, font_size=19,
                      t2c={"<bo>": C_TOK})
        q_text.next_to(user, RIGHT, buff=0.45)
        question = speech_bubble(q_text, tail_dir=LEFT, pad=0.22, color=GREY_B)

        self.play(FadeOut(intro, shift=UP * 0.1))
        self.play(FadeIn(photo, shift=RIGHT * 0.2), Create(photo_frame), FadeIn(photo_lbl))
        self.play(FadeIn(question[:3]), run_time=0.4)
        self.play(AddTextLetterByLetter(q_text, time_per_char=0.03))
        self.next_slide()

        # ===== Click: plain LLaVA does not know <bo> =====
        def assistant(row_y, strs, t2c, color, name, personalized):
            bot = icon("robot", color if personalized else GREY_B, 0.6).move_to([6.45, row_y, 0])
            parts = [bot]
            if personalized:
                badge = icon("heart", C_NEW, 0.24).move_to(bot.get_corner(DR) + [0.02, -0.02, 0])
                parts.append(badge)
            lbl = txt(name, 15, True, color).next_to(bot, UP, buff=0.1)
            body = chat_lines(strs, size=17, t2c=t2c)
            body.move_to([0, row_y, 0]).align_to([bot.get_left()[0] - 0.55, 0, 0], RIGHT)
            bub = speech_bubble(body, tail_dir=RIGHT, pad=0.2, color=color)
            return VGroup(*parts), lbl, bub

        llava_bot, llava_lbl, llava_bub = assistant(
            0.45,
            ["I\u2019m sorry, but I\u2019m not sure what you\u2019re referring to.",
             "Could you please provide more context or clarify your question?"],
            {"I\u2019m not sure what you\u2019re referring to": C_NEW},
            C_PRETRAIN, "LLaVA", False,
        )
        self.play(FadeIn(llava_bot, shift=LEFT * 0.2), FadeIn(llava_lbl))
        self.play(GrowFromPoint(llava_bub, llava_bot.get_left()), run_time=0.6)
        self.play(Indicate(llava_bub[3][0], color=C_NEW, scale_factor=1.04))
        self.play(Wiggle(llava_bot))
        self.next_slide()

        # ===== Click: what we want - a personalized LLaVA =====
        yo_bot, yo_lbl, yo_bub = assistant(
            -1.1,
            ["Yes, <bo> is in this photo. He is being held by",
             "a woman who is wearing a Mickey Mouse headband."],
            {"<bo>": C_TOK},
            C_GOAL, "Yo\u2019LLaVA", True,
        )
        want = txt("what we want", 14, True, C_GOAL).next_to(yo_bub, UP, buff=0.08, aligned_edge=LEFT)

        # locate <bo> in the photo (dog occupies roughly the left half)
        w, h = photo.width, photo.height
        dog_box = Rectangle(width=0.47 * w, height=0.7 * h, color=C_TOK, stroke_width=3)
        dog_box.move_to(photo.get_corner(UL) + np.array([0.31 * w, -0.61 * h, 0]))
        dog_tag = Text("<bo>", font="Menlo", font_size=16, weight=BOLD, color=C_TOK)
        dog_tag.move_to(dog_box.get_corner(UL), aligned_edge=UL).shift([0.06, -0.06, 0])
        dog_tag = VGroup(BackgroundRectangle(dog_tag, color=BLACK, fill_opacity=0.75, buff=0.04), dog_tag)

        self.play(FadeIn(yo_bot, shift=LEFT * 0.2), FadeIn(yo_lbl))
        self.play(GrowFromPoint(yo_bub, yo_bot.get_left()), FadeIn(want), run_time=0.6)
        self.play(Create(dog_box), FadeIn(dog_tag), Indicate(yo_bub[3][0], color=C_GOAL, scale_factor=1.04))

        takeaway = txt("LLaVA has never seen <bo>: the personal concept must be added to the model",
                       17, True, C_ACCENT)
        takeaway.move_to([2.6, -2.45, 0])
        self.play(FadeIn(takeaway, shift=UP * 0.1))
        self.next_slide()

    # ---------- Slide 9: Yo'LLaVA - training (paper Sec. 3-4) ----------
    def yollava_training(self):
        self.retitle("AUGMENT", "Yo\u2019LLaVA: Training")
        C_TOK = C_S2  # identifier token <bo>
        C_SOFT = "#BB8FCE"  # learnable soft tokens
        C_DOG = "#D4A373"
        T2C = {"<bo>": C_TOK, "<token1>": C_SOFT, "<token2>": C_SOFT, "<token3>": C_SOFT,
               "<token16>": C_SOFT, "<image>": C_ICE}
        RX = 0.45  # left edge of the right-hand panel

        def mono(s_, size=14, color=WHITE, bold=False):
            return Text(s_, font="Menlo", font_size=size, color=color,
                        weight=BOLD if bold else NORMAL, t2c=T2C)

        def sans(s_, size=15, color=WHITE, bold=False):
            return Text(s_, font=FONT, font_size=size, color=color,
                        weight=BOLD if bold else NORMAL, t2c=T2C)

        def panel_title(s_):
            return txt(s_, 17, True).move_to([0, 1.05, 0]).align_to([RX, 0, 0], LEFT)

        def tile(name, color, stroke=GREY_C, size=0.5):
            sq = RoundedRectangle(corner_radius=0.08, width=size, height=size, fill_color=C_PANEL,
                                  fill_opacity=1, stroke_color=stroke, stroke_width=1.5)
            return VGroup(sq, icon(name, color, size * 0.56).move_to(sq))

        # ===== (retitle click) the learnable soft prompt + frozen LLaVA =====
        def tchip(s_, color):
            return chip(Text(s_, font="Menlo", font_size=17, weight=BOLD, color=color), color, pad=0.08)

        strip_chips = [tchip("<bo>", C_TOK), tchip("<token1>", C_SOFT), tchip("<token2>", C_SOFT),
                       tchip("<token3>", C_SOFT), tchip("<token16>", C_SOFT)]
        strip = VGroup(strip_chips[0], txt("is", 18), *strip_chips[1:4], txt("…", 18),
                       strip_chips[4], txt(".", 18)).arrange(RIGHT, buff=0.12)
        strip.move_to([0, 1.62, 0]).align_to([-6.3, 0, 0], LEFT)
        strip_lbl = txt("learned per concept: a personalized soft prompt", 14, True, GREY_B)
        strip_lbl.next_to(strip, UP, buff=0.1, aligned_edge=LEFT)
        strip_fire = icon("fire", C_FIRE, 0.3).next_to(strip, RIGHT, buff=0.25)
        strip_note = txt("identifier + k = 16 learnable tokens", 14, color=GREY_A).next_to(strip_fire, RIGHT, buff=0.12)

        llm = block(["Large Language Model"], 5.6, 0.62, size=20).move_to([-3.5, 0.2, 0])
        ice_llm = icon("snowflake", C_ICE, 0.32).next_to(llm, LEFT, buff=0.12)
        enc = block(["Vision Encoder", "& Projector"], 2.5, 0.8, size=16).move_to([0, -0.95, 0]).align_to(llm, LEFT)
        ice_enc = icon("snowflake", C_ICE, 0.32).next_to(enc, LEFT, buff=0.12)
        prompt_lines = VGroup(
            mono("[system prompt]", 12, GREY_B),
            mono("<bo> is <token1>…<token16>.", 12),
            mono("Can you recognize <bo>", 12),
            mono("in this photo?", 12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        prompt = chip(prompt_lines, GREY_C, pad=0.12).move_to([0, -0.95, 0]).align_to(llm, RIGHT)
        photo = ImageMobject(str(ASSETS / "bo_1.png")).set_height(1.0).next_to(enc, DOWN, buff=0.45)
        out = mono("Yes, <bo> is indeed in the photo.", 14).next_to(llm, UP, buff=0.35)
        a_img = up_arrow(photo.get_top(), enc.get_bottom())
        a_enc = up_arrow(enc.get_top(), [enc.get_x(), llm.get_bottom()[1], 0])
        a_txt = up_arrow(prompt.get_top(), [prompt.get_x(), llm.get_bottom()[1], 0])
        a_out = up_arrow(llm.get_top(), out.get_bottom())

        self.play(FadeIn(strip_lbl), LaggedStart(*[GrowFromCenter(m) for m in strip], lag_ratio=0.08))
        self.play(FadeIn(strip_fire, scale=1.5), FadeIn(strip_note, shift=LEFT * 0.1))
        self.play(FadeIn(photo, shift=UP * 0.2), GrowArrow(a_img), FadeIn(enc, shift=UP * 0.1),
                  FadeIn(prompt, shift=UP * 0.1), run_time=0.8)
        self.play(GrowArrow(a_enc), GrowArrow(a_txt), FadeIn(llm, shift=UP * 0.1), run_time=0.7)
        self.play(GrowArrow(a_out), FadeIn(out, shift=UP * 0.1), FadeIn(ice_llm, scale=1.5),
                  FadeIn(ice_enc, scale=1.5), run_time=0.7)
        self.next_slide()

        # ===== Click: what exactly is trainable =====
        def vrow(label, y, new=False, color=None):
            l = mono(label, 14, color or GREY_A, bold=new).move_to([0, y, 0]).align_to([RX, 0, 0], LEFT)
            bar = Rectangle(width=1.7, height=0.2, stroke_width=1.5, stroke_color=color or GREY_C,
                            fill_color=color or "#2C3E50", fill_opacity=0.85 if new else 0.6).move_to([2.95, y, 0])
            ic = icon("fire" if new else "snowflake", C_FIRE if new else C_ICE, 0.22).next_to(bar, RIGHT, buff=0.1)
            return VGroup(l, bar, ic)

        vocab_title = panel_title("Vocabulary (input embeddings)")
        old_rows = VGroup(vrow("the", 0.55), vrow("dog", 0.22), vrow("photo", -0.11),
                          mono("⋮", 16, GREY_B).move_to([2.95, -0.42, 0]))
        new_rows = VGroup(vrow("<bo>", -0.78, True, C_TOK), vrow("<token1>", -1.11, True, C_SOFT),
                          mono("⋮", 16, C_SOFT).move_to([2.95, -1.44, 0]),
                          vrow("<token16>", -1.77, True, C_SOFT))

        W_title = txt("Output head W", 17, True).move_to([0, 1.05, 0]).align_to([4.55, 0, 0], LEFT)
        cols = VGroup(*[
            Rectangle(width=0.13, height=1.9, stroke_width=1, stroke_color=GREY_C,
                      fill_color="#2C3E50", fill_opacity=0.6) for _ in range(8)
        ]).arrange(RIGHT, buff=0.05).move_to([0, -0.4, 0]).align_to([4.6, 0, 0], LEFT)
        new_col = Rectangle(width=0.13, height=1.9, stroke_width=1, stroke_color=C_TOK,
                            fill_color=C_TOK, fill_opacity=0.85).next_to(cols, RIGHT, buff=0.12)
        ice_W = icon("snowflake", C_ICE, 0.22).next_to(cols, DOWN, buff=0.1)
        fire_W = icon("fire", C_FIRE, 0.22).next_to(new_col, DOWN, buff=0.1)
        col_lbl = txt("N words", 13, color=GREY_B).next_to(ice_W, DOWN, buff=0.06)
        new_lbl = mono("<bo>", 13, C_TOK, True).next_to(fire_W, DOWN, buff=0.06)
        W_dims = txt("C × N  →  C × (N+1)", 14, color=GREY_A).next_to(cols, UP, buff=0.12, aligned_edge=LEFT)

        train_caption = VGroup(
            sans("Trainable: <bo> + 16 soft tokens + 1 output column", 16, C_FIRE, True),
            sans("vision encoder, projector and LLM stay frozen", 14, GREY_A),
        ).arrange(DOWN, buff=0.08).move_to([3.6, -2.55, 0])

        self.play(FadeIn(vocab_title), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in old_rows], lag_ratio=0.15),
                  FadeIn(W_title), FadeIn(cols), FadeIn(ice_W), FadeIn(col_lbl), FadeIn(W_dims))
        self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in new_rows], lag_ratio=0.2), run_time=1.2)
        self.play(GrowFromEdge(new_col, DOWN), FadeIn(fire_W, scale=1.5), FadeIn(new_lbl))
        self.play(FadeIn(train_caption, shift=UP * 0.1),
                  LaggedStart(*[Indicate(c, color=C_FIRE, scale_factor=1.08) for c in strip_chips], lag_ratio=0.1))
        self.next_slide()

        # ===== Clicks: which images to train on (hard negative mining) =====
        vocab_panel = VGroup(vocab_title, old_rows, new_rows, W_title, cols, new_col, ice_W, fire_W,
                             col_lbl, new_lbl, W_dims, train_caption)
        P0 = np.array([3.45, -0.4, 0])
        data_title = panel_title("Training images (arranged by CLIP similarity)")
        pos = Group(*[ImageMobject(str(ASSETS / f"bo_{i}.png")).set_height(0.62) for i in (1, 2)]).arrange(RIGHT, buff=0.1)
        pos.move_to(P0)
        pos_lbl = sans("5 photos of <bo>", 13, C_TOK, True).next_to(pos, DOWN, buff=0.08)

        easy_specs = [("car", (-2.65, 1.0)), ("tree", (2.75, 1.0)), ("mug-hot", (-2.7, -1.35)),
                      ("building", (2.8, -1.3)), ("bicycle", (-2.95, -0.15)), ("pizza-slice", (3.0, -0.15))]
        easy = VGroup(*[tile(n, GREY_A, C_NEW).move_to(P0 + np.array([x, y, 0])) for n, (x, y) in easy_specs])
        rx, ry = 1.5, 0.95
        angles = np.linspace(0, 2 * PI, 8)[:-1] + 0.35
        hard = VGroup(*[
            tile("dog", C_DOG, C_NEW).move_to(P0 + np.array([rx * np.cos(t), ry * np.sin(t), 0]))
            for t in angles
        ])
        ring = DashedVMobject(Ellipse(width=2 * rx + 0.75, height=2 * ry + 0.75, color=C_TOK, stroke_width=2),
                              num_dashes=48).move_to(P0)
        ring_lbl = sans("top-m nearest neighbours of each photo (LAION)", 13, C_TOK).next_to(ring, DOWN, buff=0.06)

        def caption(s_):
            return sans(s_, 14, WHITE, True).move_to([3.6, -2.6, 0])

        cap_a = caption("Positives only → shortcut: always answers \u201cYes\u201d")
        cap_b = caption("+ ~100 easy negatives (random) → over-generalizes: every Shiba is <bo>")
        cap_c = caption("+ ~100 hard negatives (look-alikes) → learns what makes <bo> unique")

        self.play(FadeOut(vocab_panel))
        self.play(FadeIn(data_title), FadeIn(pos, scale=0.8), FadeIn(pos_lbl))
        self.play(FadeIn(cap_a, shift=UP * 0.1))
        self.next_slide()

        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in easy], lag_ratio=0.1), run_time=1)
        self.play(ReplacementTransform(cap_a, cap_b))
        self.next_slide()

        self.play(Create(ring), run_time=1)
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in hard], lag_ratio=0.1), FadeIn(ring_lbl), run_time=1.2)
        self.play(ReplacementTransform(cap_b, cap_c))
        self.next_slide()

        # ===== Click: the three kinds of training conversations (Table 2) =====
        data_panel = Group(data_title, pos, pos_lbl, easy, hard, ring, ring_lbl, cap_c)
        qa_title = panel_title("Training conversations (single-turn)")

        def qa_card(kind, color, thumb, q, answers, note=None):
            head = txt(kind, 14, True, color)
            qline = sans("Q: " + q, 13)
            alines = VGroup(*[sans(("A: " if i == 0 else "") + l, 13, GREY_A) for i, l in enumerate(answers)])
            alines.arrange(DOWN, aligned_edge=LEFT, buff=0.04)
            for l in alines[1:]:
                l.shift(RIGHT * 0.25)
            parts = [head, qline, alines] + ([sans(note, 12, GREY_B)] if note else [])
            body = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            row = Group(thumb, body).arrange(RIGHT, buff=0.2)
            box = RoundedRectangle(corner_radius=0.15, width=6.4, height=row.height + 0.26, fill_color=C_PANEL,
                                   fill_opacity=1, stroke_color=color, stroke_width=1.5)
            row.move_to(box).align_to(box.get_left() + RIGHT * 0.18, LEFT)
            return Group(box, row)

        cards = Group(
            qa_card("1 · Positive recognition", C_GOAL,
                    ImageMobject(str(ASSETS / "bo_2.png")).set_height(0.7),
                    "<image> Can you see if <bo> is in this photo?", ["Yes, <bo> is in the photo."]),
            qa_card("2 · Negative recognition (easy + hard negatives)", C_NEW,
                    tile("dog", C_DOG, C_NEW, size=0.7),
                    "<image> Can you check if <bo> is in this photo?",
                    ["I have analyzed the image, and I can confirm", "that <bo> is not present in the photo."]),
            qa_card("3 · Attribute Q&A, text only", C_ACCENT,
                    tile("eye-slash", GREY_B, GREY_C, size=0.7),
                    "What type of object is <bo>?", ["<bo> is a dog."],
                    note="image dropped in training → details must be stored in the soft tokens"),
        ).arrange(DOWN, buff=0.12)
        cards.move_to([0, 0, 0]).align_to([RX - 0.05, 0.82, 0], UL)

        self.play(FadeOut(data_panel))
        self.play(FadeIn(qa_title), LaggedStart(*[FadeIn(c, shift=LEFT * 0.15) for c in cards], lag_ratio=0.3),
                  run_time=1.5)
        self.next_slide()

        # ===== Click: one training step - gradients reach only the new tokens =====
        fwd = VMobject().set_points_as_corners([photo.get_top(), [enc.get_x(), llm.get_y(), 0],
                                                [out.get_x(), llm.get_y(), 0], out.get_bottom()])
        fwd_txt = Line(prompt.get_top(), [prompt.get_x(), llm.get_y(), 0])
        loss_lbl = txt("LM loss on the answer", 14, True, C_FIRE).next_to(a_out, RIGHT, buff=0.3)
        bwd = VMobject().set_points_as_corners([out.get_bottom(), [out.get_x(), llm.get_y(), 0],
                                                [prompt.get_x(), llm.get_y(), 0], prompt.get_top()])

        setup_title = panel_title("Training setup")
        items = VGroup(*[
            VGroup(Dot(radius=0.04, color=GREY_B), sans(t, 16)).arrange(RIGHT, buff=0.15)
            for t in [
                "Base model: LLaVA-1.5-13B (frozen)",
                "5 photos of <bo>  ·  k = 16 soft tokens",
                "~200 negatives: ~100 hard + ~100 easy",
                "Single-turn Q&A, standard LM loss",
                "AdamW, lr = 0.001, up to 15 epochs",
                "1× A6000 GPU",
            ]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(setup_title, DOWN, buff=0.3, aligned_edge=LEFT)
        punch = sans("A new concept = 17 tokens + 1 output column", 18, C_GOAL, True).move_to([3.6, -2.55, 0])

        self.play(FadeOut(Group(qa_title, cards)))
        self.play(ShowPassingFlash(fwd.copy().set_stroke(WHITE, 6), time_width=0.4),
                  ShowPassingFlash(fwd_txt.copy().set_stroke(C_SOFT, 6), time_width=0.6), run_time=1.3)
        self.play(FadeIn(loss_lbl, shift=LEFT * 0.1))
        self.play(ShowPassingFlash(bwd.copy().set_stroke(C_FIRE, 7), time_width=0.5), run_time=1.3)
        self.play(Indicate(prompt_lines[1], color=C_FIRE, scale_factor=1.1),
                  LaggedStart(*[Indicate(c, color=C_FIRE, scale_factor=1.12) for c in strip_chips], lag_ratio=0.1),
                  Indicate(ice_llm, color=WHITE), Indicate(ice_enc, color=WHITE), run_time=1.2)
        self.play(FadeIn(setup_title), LaggedStart(*[FadeIn(i, shift=RIGHT * 0.1) for i in items], lag_ratio=0.12))
        self.play(FadeIn(punch, shift=UP * 0.1))
        self.next_slide()

    # ---------- Slide 10: Yo'LLaVA - qualitative results (paper, Table 3) ----------
    def yollava_results(self):
        self.retitle("AUGMENT · RESULTS", "Yo\u2019LLaVA: Results")
        TOKENS = {"<T>": C_PRETRAIN, "<bo>": C_S2}
        RED, GREEN = C_NEW, C_GOAL

        def para(strs, size, hl=None, color=GREY_A):
            t2c = {**(hl or {}), "[omitted]": GREY_C, **TOKENS}  # tokens last -> keep their colour
            return VGroup(*[Text(x, font=FONT, font_size=size, color=color, t2c=t2c) for x in strs]
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)

        def concept_row(tok, fname, y):
            lbl = Text(tok + ":", font="Menlo", font_size=18, weight=BOLD, color=TOKENS[tok])
            img = ImageMobject(str(ASSETS / fname)).set_height(0.68)
            g = Group(lbl, img).arrange(RIGHT, buff=0.15)
            return g.move_to([0, y, 0]).align_to([-6.35, 0, 0], LEFT)

        SPEAKERS = {
            "gpt": ("GPT-4V", "+ text prompt", GREY_B, GREY_C),
            "llava": ("LLaVA", None, C_PRETRAIN, GREY_B),
            "yo": ("Yo\u2019LLaVA", None, C_GOAL, C_GOAL),
        }

        def answer(kind, strs, top_y, size, hl=None):
            name, sub, color, bot_color = SPEAKERS[kind]
            body = para(strs, size, hl, WHITE if kind == "yo" else GREY_A)
            bub = chip(body, color, pad=0.15)
            bub.move_to([0, 0, 0]).align_to([0.15, top_y, 0], UL)
            bot = icon("robot", bot_color, 0.42).move_to([-0.7, bub.get_top()[1] - 0.3, 0])
            parts = [bot]
            if kind == "yo":
                parts.append(icon("heart", C_NEW, 0.18).move_to(bot.get_corner(UR) + [0.04, 0.0, 0]))
            labels = [txt(name, 13, True, color)]
            if sub:
                labels.append(txt(sub, 11, color=GREY_B))
            lbl = VGroup(*labels).arrange(DOWN, buff=0.03).next_to(bot, DOWN, buff=0.06)
            return VGroup(VGroup(*parts), lbl, bub)

        def play_answer(a, emphasize=None):
            who, lbl, bub = a
            self.play(FadeIn(who, shift=RIGHT * 0.15), FadeIn(lbl), GrowFromEdge(bub[0], LEFT), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in bub[1]], lag_ratio=0.25),
                      run_time=0.4 + 0.25 * len(bub[1]))
            if emphasize is not None:
                self.play(Indicate(emphasize, color=GREEN, scale_factor=1.04))

        def question_bubble(strs):
            user = icon("circle-user", GREY_A, 0.5)
            body = para(strs, 15, color=WHITE)
            bub = chip(body, GREY_B, pad=0.15)
            bub.move_to([0, 0, 0]).align_to([0.15, 2.05, 0], UL)
            user.move_to([-0.7, bub.get_y(), 0])
            return VGroup(user, bub)

        def play_setup(concepts, photo, q):
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in concepts], lag_ratio=0.2), run_time=0.8)
            self.play(FadeIn(photo[0], shift=UP * 0.15), FadeIn(photo[1]), run_time=0.6)
            self.play(FadeIn(q[0], scale=0.8), GrowFromEdge(q[1][0], LEFT), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(l) for l in q[1][1]], lag_ratio=0.3), run_time=0.6)

        def input_photo(fname, center_y):
            img = ImageMobject(str(ASSETS / fname)).set_width(4.3)
            img.move_to([0, center_y, 0]).align_to([-6.35, 0, 0], LEFT)
            lbl = txt("input image", 13, color=GREY_B).next_to(img, DOWN, buff=0.08)
            return Group(img, lbl)

        # ===== Example 1: recognize <T> (same click as the retitle) =====
        c1 = [concept_row("<T>", "yl_T_refs.png", 1.7)]
        p1 = input_photo("yl_party.png", -0.05)
        q1 = question_bubble(["Can you see if <T> is in this photo?"])
        play_setup(c1, p1, q1)
        self.next_slide()

        y = q1[1].get_bottom()[1] - 0.3
        g1 = answer("gpt", ["Sorry, I can\u2019t help with identifying or making",
                            "assumptions about people in images."], y, 14,
                    {"Sorry, I can\u2019t help": RED})
        play_answer(g1)
        self.next_slide()

        y = g1[2].get_bottom()[1] - 0.3
        l1 = answer("llava", ["I\u2019m unable to see any individuals in the photo",
                              "you\u2019ve provided\u2026 [omitted]"],
                    y, 14, {"unable to see any individuals": RED})
        play_answer(l1)
        self.next_slide()

        y = l1[2].get_bottom()[1] - 0.3
        y1 = answer("yo", ["<T> is in the photo, and she is standing next to another person.",
                           "They both appear to be having a good time, possibly at a party",
                           "or event. <T> is smiling and making a peace sign with her hand,",
                           "which suggests a positive and fun atmosphere."], y, 14,
                    {"smiling and making a peace sign with her hand": GREEN, "is in the photo": GREEN})
        play_answer(y1, emphasize=y1[2][1][2])
        self.next_slide()

        # ===== Example 2: <T> and <bo> together, with a creative request =====
        ex1 = Group(*c1, p1, q1, g1, l1, y1)
        c2 = [concept_row("<T>", "yl_T_refs.png", 1.7), concept_row("<bo>", "yl_bo_refs.png", 0.85)]
        p2 = input_photo("yl_park.png", -1.05)
        q2 = question_bubble(["Hey, can you see <T> and <bo> in this photo? Could you write",
                              "a cute and detailed Facebook caption for it?"])
        self.play(FadeOut(ex1, shift=LEFT * 0.6), run_time=0.7)
        play_setup(c2, p2, q2)
        self.next_slide()

        y = q2[1].get_bottom()[1] - 0.25
        g2 = answer("gpt", ["Yes, I can see them in the photo! Here\u2019s a cute Facebook caption:",
                            "\u201cSunshine, smiles, and Shiba snuggles! Couldn\u2019t have asked for a better",
                            "day at the park with my favorite furry friend! #DogLover #ParkDays \u2026\u201d"], y, 13)
        play_answer(g2)
        self.next_slide()

        y = g2[2].get_bottom()[1] - 0.25
        l2 = answer("llava", ["In the photo, I see a person and a dog. The person is sitting",
                              "on the grass\u2026 [omitted]"], y, 13, {"I see a person and a dog": RED})
        play_answer(l2)
        self.next_slide()

        y = l2[2].get_bottom()[1] - 0.25
        y2 = answer("yo", ["Yes, I can see <T> and <bo> in the photo. They are sitting on the grass in a",
                           "park, enjoying a beautiful day. <T> is holding <bo>, and they both seem to be",
                           "having a great time together. As for the Facebook caption: \u201cSpending the day",
                           "in the park with my furry best friend! <bo> is the cutest thing I\u2019ve ever seen,",
                           "and I\u2019m so grateful for our adventures together. <bo>, you\u2019re the best boy! \u2026\u201d"],
                    y, 13, {"Yes, I can see": GREEN, "in the photo. They are sitting on the grass in a": GREEN,
                            "park, enjoying a beautiful day": GREEN})
        play_answer(y2)
        toks = [m for line in y2[2][1] for m in [line]]
        self.play(LaggedStart(*[Indicate(l, color=C_S2, scale_factor=1.03) for l in toks], lag_ratio=0.1), run_time=1.2)
        self.next_slide()


    # ---------- Transition: back to the overview, zoom into "Knowledge Update" ----------
    def update_transition(self):
        o = overview_final_state(build_overview())
        augment = VGroup(o["blue"], o["red"], o["blue_txt"], o["red_txt"])
        update = Group(o["timeline"], o["ticks"], o["year_lbls"], o["photo"], o["photo_frame"],
                       o["question"], o["marker"], *o["robots"], *o["answers"])
        albl, ulbl = o["augment_lbl"], o["update_lbl"]

        # clear the Yo'LLaVA slide, bring back slide 2 as the audience last saw it
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.play(FadeIn(augment), FadeIn(albl), FadeIn(update), FadeIn(ulbl), run_time=0.8)
        # Augment is done -> tick it off
        done = Line(albl.get_left() + LEFT * 0.1, albl.get_right() + RIGHT * 0.1, color=GREY_B, stroke_width=4)
        self.play(VGroup(augment, albl).animate.set_opacity(0.35), Create(done), run_time=0.6)
        # Knowledge Update half takes the stage, then pops
        self.play(
            FadeOut(VGroup(augment, albl, done), shift=LEFT * 0.6),
            update.animate.scale(0.95).move_to(UP * 0.55),
            ulbl.animate.scale(1.2).move_to(DOWN * 2.6),
            run_time=1.2,
        )
        self.play(ulbl.animate.set_color(C_ACCENT), run_time=0.3)
        self.play(Indicate(ulbl, scale_factor=1.35, color=C_ACCENT), run_time=0.8)
        self.next_slide()
        self.update_lbl, self.update_group = ulbl, update

    # ---------- Slide 11: How to handle evolving knowledge? (MMEVOKE) ----------
    def evolving_knowledge(self):
        HEAD_X = -6.1
        kicker = txt("KNOWLEDGE UPDATE", size=18, bold=True, color=C_ACCENT)
        kicker.move_to([0, 3.4, 0]).align_to([HEAD_X, 0, 0], LEFT)
        heading = txt("How to handle evolving knowledge?", size=40, bold=True)
        heading.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        cite = make_cite("When Large Multimodal Models Confront Evolving Knowledge: "
                         "Challenges and Explorations (Jiang et al., 2026)")

        # ----- evolving-knowledge timeline (paper Fig. 1), chronological -----
        events = [  # (asset, category, name, date)
            ("su7", "Entity · Sports Car", "Xiaomi SU7", "Mar 28, 2024"),
            ("trump", "News · Politics", "Assassination attempt on Trump", "Jul 13, 2024"),
            ("wukong", "News · Entertainment", "Black Myth: Wukong", "Aug 20, 2024"),
            ("iphone", "Entity · Product", "iPhone 16", "Sep 10, 2024"),
            ("nobel", "News · Science", "Nobel Prize in Physics", "Oct 8, 2024"),
            ("apt", "Entity · Music", "APT.", "Oct 18, 2024"),
            ("nezha", "Entity · Film", "Ne Zha 2", "Jan 29, 2025"),
            ("falcon", "News · Technology", "Falcon 9 first-stage recovery", "Feb 19, 2025"),
            ("switch", "Entity · Gameboy", "Nintendo Switch 2", "Jun 5, 2025"),
            ("yu7", "Entity · Sports Car", "Xiaomi YU7", "Jun 26, 2025"),
        ]
        cat_colors = {"Entity": C_PRETRAIN, "News": C_S2}
        TL_Y = 0.05
        CW = 1.9  # card width

        def card(asset, cat, name, date):
            img = ImageMobject(str(ASSETS / f"evo_{asset}.png")).set_width(CW - 0.12)
            col = cat_colors[cat.split(" ")[0]]
            c = txt(cat, 12, True, col)
            n = txt(name, 12, True)
            if n.width > CW - 0.1:
                n.scale_to_fit_width(CW - 0.1)
            d = txt(date, 11, color=GREY_B)
            body = Group(c, img, n, d).arrange(DOWN, buff=0.06)
            box = RoundedRectangle(corner_radius=0.08, width=CW, height=body.height + 0.16, fill_color=C_PANEL,
                                   fill_opacity=1, stroke_color=col, stroke_width=1.2).move_to(body)
            return Group(box, body)

        cards = []
        x0 = -4.45
        for i, ev in enumerate(events):
            cd = card(*ev)
            x = x0 + i * 1.12
            if i % 2 == 0:
                cd.move_to([x, 0, 0]).align_to([0, TL_Y + 0.2, 0], DOWN)
            else:
                cd.move_to([x, 0, 0]).align_to([0, TL_Y - 0.2, 0], UP)
            cards.append(cd)

        axis = Arrow([-6.2, TL_Y, 0], [6.6, TL_Y, 0], buff=0, color=C_S2, stroke_width=5,
                     max_tip_length_to_length_ratio=0.03)
        axis.set_color(color=[C_PRETRAIN, C_GOAL, C_ACCENT, C_S2])
        stems = VGroup(*[
            DashedLine([cd.get_x(), TL_Y, 0],
                       [cd.get_x(), cd.get_bottom()[1] if i % 2 == 0 else cd.get_top()[1], 0],
                       dash_length=0.05, color=GREY_C, stroke_width=1.5)
            for i, cd in enumerate(cards)
        ])
        cutoff = Dot([-5.75, TL_Y, 0], radius=0.11, color=WHITE)
        cutoff_lbl = VGroup(txt("2023.04", 14, True), txt("LMM knowledge cutoff", 12, color=GREY_B)
                            ).arrange(DOWN, buff=0.03).next_to(cutoff, DOWN, buff=0.12)

        # (same click as the transition) the label becomes the section tag
        self.play(
            FadeOut(self.update_group, scale=0.7),
            ReplacementTransform(self.update_lbl, kicker),
            run_time=1,
        )
        self.play(Write(heading), FadeIn(cite, shift=UP * 0.2), run_time=1.2)
        self.play(GrowArrow(axis), FadeIn(cutoff, scale=1.5), FadeIn(cutoff_lbl), run_time=1)
        self.play(LaggedStart(*[AnimationGroup(Create(st), FadeIn(cd, shift=(UP if i % 2 == 0 else DOWN) * 0.2))
                                for i, (st, cd) in enumerate(zip(stems, cards))], lag_ratio=0.2), run_time=3)
        self.next_slide()

        # ===== Click: the problem - LMM is stuck at its cutoff =====
        timeline = Group(axis, stems, cutoff, cutoff_lbl, *cards)  # shades added below
        su7_box = SurroundingRectangle(cards[0], buff=0.05, corner_radius=0.08, color=C_ACCENT, stroke_width=3)
        shades = VGroup(*[
            SurroundingRectangle(c, buff=0.02, corner_radius=0.08, stroke_width=0,
                                 fill_color=BLACK, fill_opacity=0.65) for c in cards[1:]
        ])
        self.play(FadeIn(shades), Create(su7_box), run_time=0.8)
        timeline.add(shades)
        self.play(Group(timeline, su7_box).animate.scale(0.5).move_to([0, 1.3, 0]), run_time=1.0)

        PY = -1.35  # demo row centre
        left_panel = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=6.1, height=3.0, color=GREY_B,
                                                     stroke_width=2), num_dashes=70).move_to([-3.3, PY, 0])
        right_panel = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=6.4, height=3.0, color=C_NEW,
                                                      stroke_width=2), num_dashes=70).move_to([3.15, PY, 0])

        q_img = ImageMobject(str(ASSETS / "evo_q_car.png")).set_height(0.85)
        q_txt = VGroup(txt("Question:", 14, True),
                       lines(["Which company produces", "the car in the image?"], size=14, buff=0.04)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        q_row = Group(q_img, q_txt).arrange(RIGHT, buff=0.2)
        q_box = RoundedRectangle(corner_radius=0.15, width=q_row.width + 0.3, height=q_row.height + 0.26,
                                 fill_color=C_PANEL, fill_opacity=1, stroke_color=GREY_B, stroke_width=1.5)
        q_bub = Group(q_box, q_row.move_to(q_box)).move_to([-3.75, PY + 0.65, 0])
        user = icon("circle-user", GREY_A, 0.55).next_to(q_bub, RIGHT, buff=0.25)
        user_lbl = txt("User", 13, color=GREY_B).next_to(user, DOWN, buff=0.06)

        lmm_old = icon("robot", GREY_B, 0.55).move_to([-5.75, PY - 0.75, 0])
        lmm_old_lbl = txt("LMM", 13, True, GREY_B).next_to(lmm_old, DOWN, buff=0.06)
        r_old = chip(VGroup(txt("Response:", 14, True), txt("Porsche", 14)).arrange(RIGHT, buff=0.1),
                     GREY_B, pad=0.14).next_to(lmm_old, RIGHT, buff=0.3)
        wrong = VGroup(Line(UL, DR), Line(UR, DL)).set_stroke(C_NEW, 7).scale(0.18).next_to(r_old, RIGHT, buff=0.3)
        flying = cards[0][1][1].copy()  # the SU7 photo from the (shrunken) card

        self.play(Create(left_panel), run_time=0.6)
        self.play(flying.animate.replace(q_img), FadeIn(q_box), FadeIn(q_txt), FadeIn(user, scale=0.8),
                  FadeIn(user_lbl), run_time=1.0)
        self.add(q_img)
        self.remove(flying)
        self.play(FadeIn(lmm_old, shift=UP * 0.15), FadeIn(lmm_old_lbl))
        self.play(GrowFromEdge(r_old, LEFT), run_time=0.5)
        self.play(Create(wrong), Wiggle(lmm_old), run_time=0.8)
        self.next_slide()

        # ===== Click: knowledge injection fixes it =====
        inj_title = txt("Evolving knowledge for injection", 15, True, C_NEW)
        inj_title.move_to(right_panel.get_corner(UL) + [0.25, -0.3, 0], aligned_edge=UL)
        inj_img = ImageMobject(str(ASSETS / "evo_inj_car.png")).set_height(1.1)
        doc = chip(lines(["The Xiaomi SU7 is the first", "electric sedan created by",
                          "Xiaomi Auto Company, \u2026", "\u201cModena\u201d platform \u2026"], size=11, buff=0.04,
                         color=GREY_A), GREY_C, pad=0.12)
        doc[1][2].set_color(C_NEW)
        RY = PY + 0.2  # row: injected knowledge -> LMM
        doc_row = Group(inj_img, doc).arrange(RIGHT, buff=0.12)
        doc_row.move_to([0, RY, 0]).align_to([0.25, 0, 0], LEFT)

        lmm_new = icon("robot", C_GOAL, 0.6).move_to([5.55, RY, 0])
        lmm_fire = icon("fire", C_FIRE, 0.24).move_to(lmm_new.get_corner(UR) + [0.05, 0.02, 0])
        lmm_new_lbl = txt("LMM", 13, True, C_GOAL).next_to(lmm_new, DOWN, buff=0.06)
        inj_arrow = Arrow([doc_row.get_right()[0] + 0.1, RY, 0], [lmm_new.get_left()[0] - 0.12, RY, 0],
                          buff=0, color=C_FIRE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        inj_lbl = lines(["Knowledge", "Injection"], align=ORIGIN, size=13, bold=True, color=C_FIRE, buff=0.02)
        inj_lbl.next_to(inj_arrow, UP, buff=0.08)

        # response row mirrors the left panel: [Response] + mark, same height
        r_new = chip(VGroup(txt("Response:", 14, True), txt("Xiaomi Auto Company", 14)).arrange(RIGHT, buff=0.1),
                     C_GOAL, pad=0.14)
        r_new.move_to([0, r_old.get_y(), 0]).align_to([5.6, 0, 0], RIGHT)
        right = VMobject().set_points_as_corners([LEFT * 0.5 + UP * 0.05, DOWN * 0.45, RIGHT * 0.8 + UP * 0.75])
        right.set_stroke(C_GOAL, 7).scale(0.2).next_to(r_new, RIGHT, buff=0.22)

        self.play(Create(right_panel), FadeIn(inj_title), run_time=0.6)
        self.play(FadeIn(inj_img, shift=RIGHT * 0.15), FadeIn(doc, shift=RIGHT * 0.15))
        self.play(Indicate(doc[1][2], color=C_NEW, scale_factor=1.1))
        self.play(GrowArrow(inj_arrow), FadeIn(inj_lbl))
        self.play(FadeIn(lmm_new, shift=LEFT * 0.15), FadeIn(lmm_fire, scale=1.6), FadeIn(lmm_new_lbl))
        self.play(GrowFromEdge(r_new, LEFT), run_time=0.6)
        self.play(Create(right), run_time=0.5)
        self.next_slide()

        self.ek = dict(kicker=kicker, heading=heading, cite=cite, lmm_old=lmm_old, lmm_old_lbl=lmm_old_lbl,
                       lmm_new=VGroup(lmm_new, lmm_fire), lmm_new_lbl=lmm_new_lbl,
                       inj_arrow=inj_arrow, inj_lbl=inj_lbl)

    # ---------- Slide 12: the two goals - adaptation & retention (paper Eq. 1-2) ----------
    def adaptation_retention(self):
        ek = self.ek
        C_M, C_MS = GREY_A, C_GOAL  # original model / injected model
        C_DQ, C_DP = C_ACCENT, C_PRETRAIN  # new knowledge / previous knowledge
        EQ_SIZE = 0.47

        # ===== Transition: the cartoon becomes the notation =====
        ROW_Y = 1.95
        old_bot = ek["lmm_old"].copy().set_height(0.5).move_to([-3.9, ROW_Y, 0])
        new_bot = ek["lmm_new"].copy()
        new_bot.scale(0.5 / ek["lmm_new"][0].height).move_to([1.05, ROW_Y, 0])
        m_lbl = MathParts([("m", r"\mathcal{M}")], size=0.5).next_to(old_bot, RIGHT, buff=0.2)
        m_lbl.set_color(C_M)
        ms_lbl = MathParts([("ms", r"\mathcal{M}^*"), ("eq", r"=f(\mathcal{M},\mathcal{D}_K)")], size=0.45)
        ms_lbl["ms"].set_color(C_MS)
        ms_lbl.next_to(new_bot, RIGHT, buff=0.2)
        cap_old = txt("original LMM", 13, color=GREY_B).next_to(VGroup(old_bot, m_lbl), DOWN, buff=0.06)
        cap_new = VGroup(txt("after injecting the new data with method ", 13, color=GREY_B),
                         MathParts([("f", "f")], size=0.26).set_color(C_FIRE)).arrange(RIGHT, buff=0.05)
        cap_new.next_to(VGroup(new_bot, ms_lbl), DOWN, buff=0.06)
        arrow = Arrow([m_lbl.get_right()[0] + 0.25, ROW_Y, 0], [new_bot.get_left()[0] - 0.2, ROW_Y, 0],
                      buff=0, color=C_FIRE, stroke_width=4, max_tip_length_to_length_ratio=0.12)
        arrow_lbl = txt("knowledge injection", 13, True, C_FIRE).next_to(arrow, UP, buff=0.06)

        keep = {ek["kicker"], ek["heading"], ek["cite"], ek["lmm_old"], ek["lmm_old_lbl"], ek["lmm_new"],
                ek["lmm_new_lbl"], ek["inj_arrow"], ek["inj_lbl"], *ek["lmm_new"]}
        others = [m for m in self.mobjects if m not in keep]
        self.play(*[FadeOut(m) for m in others], run_time=0.6)
        self.play(
            Transform(ek["lmm_old"], old_bot), Transform(ek["lmm_new"], new_bot),
            ReplacementTransform(ek["lmm_old_lbl"], m_lbl), ReplacementTransform(ek["lmm_new_lbl"], ms_lbl),
            Transform(ek["inj_arrow"], arrow), Transform(ek["inj_lbl"], arrow_lbl),
            run_time=1.3,
        )
        self.play(FadeIn(cap_old, shift=UP * 0.1), FadeIn(cap_new, shift=UP * 0.1))
        two_goals = VGroup(txt("A good injection method ", 17, True, GREY_A),
                           MathParts([("f", "f")], size=0.36).set_color(C_FIRE),
                           txt(" has two goals", 17, True, GREY_A)).arrange(RIGHT, buff=0.06)
        two_goals.move_to([0, 1.2, 0])
        self.play(FadeIn(two_goals, shift=UP * 0.1))
        self.next_slide()

        # ===== the two equations =====
        def equation(opt, sub, dset, first, second):
            eq = MathParts([
                ("opt", r"\mathrm{%s}\;" % opt),
                ("E", r"\mathrm{𝔼}_{(i_%s,x_%s,y_%s)\sim\mathcal{D}_%s}" % (sub, sub, sub, dset)),
                ("lb", r"\,\left[\,"),
                ("t1", r"\mathrm{𝕀}\,(%s(i_%s,x_%s)=y_%s)" % (first, sub, sub, sub)),
                ("minus", r"\;-\;"),
                ("t2", r"\mathrm{𝕀}\,(%s(i_%s,x_%s)=y_%s)" % (second, sub, sub, sub)),
                ("rb", r"\,\right]"),
            ], size=EQ_SIZE)
            f = MathParts([("f", "f")], size=EQ_SIZE * 0.7)
            f.next_to(eq["opt"], DOWN, buff=0.06)
            eq.add(f)
            return eq, f

        MS, M = r"\mathcal{M}^*", r"\mathcal{M}"
        eq1, f1 = equation("max", "q", "Q", MS, M)
        eq2, f2 = equation("min", "p", "P", M, MS)
        for eq, dcol, (c1, c2) in [(eq1, C_DQ, (C_MS, C_M)), (eq2, C_DP, (C_M, C_MS))]:
            eq["E"].set_color(dcol)
            eq["t1"].set_color(c1)
            eq["t2"].set_color(c2)
        eq1.move_to([0.35, 0.42, 0])
        eq2.move_to([0.35, -1.75, 0])
        f1.set_color(C_FIRE)
        f2.set_color(C_FIRE)

        def tag(num, title, sub, color):
            return VGroup(txt(f"{num}  {title}", 18, True, color), txt(sub, 15, color=GREY_B)).arrange(RIGHT, buff=0.25)

        tag1 = tag("①", "Knowledge Adaptation", "learn the new", C_DQ)
        tag1.move_to([0, 1.08, 0]).align_to([-6.2, 0, 0], LEFT)
        tag2 = tag("②", "Knowledge Retention", "keep the old", C_DP)
        tag2.move_to([0, -2.6, 0]).align_to([-6.2, 0, 0], LEFT)

        def write_eq(eq):
            order = ["opt", "E", "lb", "t1", "minus", "t2", "rb"]
            self.play(LaggedStart(*[FadeIn(eq[n], shift=UP * 0.1) for n in order], FadeIn(eq[-1]),
                                  lag_ratio=0.12), run_time=1.6)

        # ===== chips under / over terms =====
        def explainer(eq, names, texts, colors, below=True, y=None):
            boxes, chips, links = VGroup(), VGroup(), VGroup()
            for n, t, c in zip(names, texts, colors):
                target = VGroup(eq["opt"], eq[-1]) if n == "opt" else eq[n]
                b = SurroundingRectangle(target, buff=0.06, corner_radius=0.06, color=c, stroke_width=2.5)
                body = lines(t, size=15, color=WHITE, buff=0.05, align=ORIGIN)
                ch = chip(body, c, pad=0.12)
                ch.move_to([b.get_x(), y, 0])
                boxes.add(b)
                chips.add(ch)
            # resolve overlaps left -> right, then clamp to the frame
            for i in range(1, len(chips)):
                gap = chips[i].get_left()[0] - chips[i - 1].get_right()[0]
                if gap < 0.15:
                    chips[i].shift(RIGHT * (0.15 - gap))
            over = chips.get_right()[0] - 6.7
            if over > 0:
                chips.shift(LEFT * over)
            for b, ch in zip(boxes, chips):
                start = b.get_bottom() if below else b.get_top()
                end = ch.get_top() if below else ch.get_bottom()
                links.add(Line(start, end, color=b.get_color(), stroke_width=2))
            return boxes, chips, links

        def show_explainer(boxes, chips, links):
            for b, ch, l in zip(boxes, chips, links):
                self.play(Create(b), run_time=0.4)
                self.play(Create(l), FadeIn(ch, shift=(DOWN if l.get_end()[1] < l.get_start()[1] else UP) * 0.1),
                          run_time=0.5)

        # ===== Click: equation (1) =====
        self.play(FadeOut(two_goals, shift=UP * 0.1), FadeIn(tag1, shift=RIGHT * 0.1))
        write_eq(eq1)
        self.next_slide()

        # ===== Click: what each term means (SU7 example) =====
        b1, c1, l1 = explainer(
            eq1, ["opt", "E", "t1", "t2"],
            [["pick the injection", "method f"],
             ["average over questions", "about NEW knowledge"],
             ["injected model right?", "“Xiaomi” ✓  →  1"],
             ["original model right?", "“Porsche” ✗  →  0"]],
            [C_FIRE, C_DQ, C_MS, C_M], below=True, y=-0.45)
        show_explainer(b1, c1, l1)
        self.next_slide()

        # ===== Click: the bracket = gain on new knowledge =====
        gain_body = lines(["gain on new knowledge:  1 − 0 = 1", "→ make it as large as possible"],
                          size=16, color=WHITE, buff=0.05, align=ORIGIN)
        gain = chip(gain_body, C_DQ, pad=0.14)
        gain.move_to([VGroup(eq1["t1"], eq1["t2"]).get_x(), -0.45, 0])
        bracket_box = SurroundingRectangle(VGroup(eq1["lb"], eq1["rb"]), buff=0.06, corner_radius=0.06,
                                           color=C_DQ, stroke_width=2.5)
        gain_link = Line(bracket_box.get_bottom(), gain.get_top(), color=C_DQ, stroke_width=2)
        self.play(ReplacementTransform(VGroup(c1[2], c1[3]), gain), ReplacementTransform(VGroup(b1[2], b1[3]), bracket_box),
                  ReplacementTransform(VGroup(l1[2], l1[3]), gain_link), run_time=1.0)
        self.play(Indicate(gain, color=C_DQ, scale_factor=1.05))
        self.next_slide()

        # ===== Click: equation (2) =====
        eq1_notes = VGroup(b1[0], b1[1], c1[0], c1[1], l1[0], l1[1], gain, bracket_box, gain_link)
        self.play(FadeOut(eq1_notes), FadeIn(tag2, shift=RIGHT * 0.1))
        write_eq(eq2)
        self.next_slide()

        # ===== Click: what each term means (forgetting example) =====
        b2, c2, l2 = explainer(
            eq2, ["opt", "E", "t1", "t2"],
            [["pick the injection", "method f"],
             ["average over general tests", "(knowledge it ALREADY had)"],
             ["original model right?", "✓  →  1"],
             ["injected model right?", "forgot it ✗  →  0"]],
            [C_FIRE, C_DP, C_M, C_MS], below=False, y=-0.5)
        show_explainer(b2, c2, l2)
        self.next_slide()

        # ===== Click: the bracket = forgetting =====
        gap_body = VGroup(
            Text("gap = 1 − 0 = 1  →  capability degradation", font=FONT, font_size=16,
                 t2c={"capability degradation": C_NEW}),
            txt("→ keep it as close to 0 as possible", 16),
        ).arrange(DOWN, buff=0.05)
        gap = chip(gap_body, C_DP, pad=0.14)
        gap.move_to([VGroup(eq2["t1"], eq2["t2"]).get_x(), -0.5, 0])
        bracket_box2 = SurroundingRectangle(VGroup(eq2["lb"], eq2["rb"]), buff=0.06, corner_radius=0.06,
                                            color=C_DP, stroke_width=2.5)
        gap_link = Line(bracket_box2.get_top(), gap.get_bottom(), color=C_DP, stroke_width=2)
        self.play(ReplacementTransform(VGroup(c2[2], c2[3]), gap), ReplacementTransform(VGroup(b2[2], b2[3]), bracket_box2),
                  ReplacementTransform(VGroup(l2[2], l2[3]), gap_link), run_time=1.0)
        self.play(Indicate(gap, color=C_DP, scale_factor=1.05))
        self.next_slide()

        # ===== Click: the trade-off in one line =====
        eq2_notes = VGroup(b2[0], b2[1], c2[0], c2[1], l2[0], l2[1], gap, bracket_box2, gap_link)
        left = VGroup(txt("① learn the new", 18, True, C_DQ), txt("gain on new knowledge ↑", 14, color=GREY_A)
                      ).arrange(DOWN, buff=0.05)
        right = VGroup(txt("② keep the old", 18, True, C_DP), txt("forgetting gap ↓", 14, color=GREY_A)
                       ).arrange(DOWN, buff=0.05)
        vs = txt("at the same time", 15, True, GREY_B)
        summary = VGroup(left, vs, right).arrange(RIGHT, buff=0.6)
        summary_box = chip(summary, GREY_C, pad=0.2).move_to([0.35, -0.6, 0])
        self.play(FadeOut(eq2_notes))
        self.play(FadeIn(summary_box[0]), FadeIn(left, shift=RIGHT * 0.15), FadeIn(right, shift=LEFT * 0.15))
        self.play(FadeIn(vs, scale=0.8))
        self.next_slide()

        self.ar = dict(kicker=ek["kicker"], heading=ek["heading"], cite=ek["cite"],
                       legend=[ek["lmm_old"], ek["lmm_new"], m_lbl, ms_lbl, ek["inj_arrow"]],
                       arrow_lbl=ek["inj_lbl"], caps=[cap_old, cap_new])

    # ---------- Slide 13: MMEvoke - three ways to inject knowledge ----------
    def injection_methods(self):
        ar = self.ar
        C_SFT, C_LORA, C_RAG = C_S2, "#BB8FCE", C_GOAL
        XS = [-4.5, 0.0, 4.5]  # column centres

        # ===== Transition: "which f?" =====
        new_head = txt("MMEvoke Benchmark", size=40, bold=True)
        new_head.move_to(ar["heading"]).align_to(ar["heading"], LEFT)
        which = VGroup(txt("which injection method ", 14, True, C_FIRE),
                       MathParts([("f", "f")], size=0.3).set_color(C_FIRE),
                       txt(" ?", 14, True, C_FIRE)).arrange(RIGHT, buff=0.04)
        which.move_to(ar["arrow_lbl"])
        keep = {ar["kicker"], ar["heading"], ar["cite"], ar["arrow_lbl"], *ar["legend"]}
        keep |= {m for x in ar["legend"] for m in x.get_family()}
        others = [m for m in self.mobjects if m not in keep]
        self.play(*[FadeOut(m) for m in others], Transform(ar["heading"], new_head), run_time=0.9)
        self.play(ReplacementTransform(ar["arrow_lbl"], which), run_time=0.6)
        self.play(Indicate(which, color=C_FIRE, scale_factor=1.15))

        def header(num, title, sub, color, x):
            t = txt(f"{num}  {title}", 21, True, color)
            st = txt(sub, 14, color=GREY_B)
            return VGroup(t, st).arrange(DOWN, buff=0.06).move_to([x, 1.0, 0])

        heads = [
            header("①", "Full fine-tuning", "update the model’s weights", C_SFT, XS[0]),
            header("②", "LoRA", "train a small low-rank add-on", C_LORA, XS[1]),
            header("③", "Multimodal RAG", "no training: look it up", C_RAG, XS[2]),
        ]
        seps = VGroup(*[DashedLine([x, 1.25, 0], [x, -3.0, 0], dash_length=0.08, color=GREY_D, stroke_width=1.5)
                        for x in (-2.25, 2.25)])
        self.play(Create(seps), LaggedStart(*[FadeIn(h, shift=DOWN * 0.15) for h in heads], lag_ratio=0.2),
                  run_time=1.2)
        self.next_slide()

        def caption(lines_, color, x):
            return lines(lines_, align=ORIGIN, size=15, bold=True, color=color, buff=0.04).move_to([x, -2.85, 0])

        # ===== Click: Full fine-tuning =====
        x = XS[0]
        llm = block(["Large Language Model"], 3.4, 0.62, trainable=True, size=16).move_to([x + 0.2, -0.05, 0])
        proj = block(["Projection"], 1.9, 0.4, trainable=True, size=15).move_to([x - 0.55, -0.8, 0])
        enc = block(["Vision", "Encoder"], 1.9, 0.62, size=15).move_to([x - 0.55, -1.6, 0])
        ic_llm = icon("fire", C_FIRE, 0.3).next_to(llm, LEFT, buff=0.12)
        ic_proj = icon("fire", C_FIRE, 0.3).next_to(proj, LEFT, buff=0.12)
        ic_enc = icon("snowflake", C_ICE, 0.3).next_to(enc, LEFT, buff=0.12)
        a1 = up_arrow(enc.get_top(), proj.get_bottom())
        a2 = up_arrow(proj.get_top(), [proj.get_x(), llm.get_bottom()[1], 0])
        data = chip(lines(["train on injection", "data: image +", "knowledge summary"], size=11,
                          color=GREY_A, buff=0.03, align=ORIGIN), C_ACCENT, pad=0.1)
        data.move_to([x + 1.3, -1.4, 0])
        a_data = up_arrow(data.get_top(), [data.get_x(), llm.get_bottom()[1], 0])
        cap1 = caption(["all LLM weights change", "(billions of parameters)"], C_SFT, x)

        self.play(FadeIn(enc, shift=UP * 0.1), FadeIn(ic_enc, scale=1.4))
        self.play(GrowArrow(a1), FadeIn(proj, shift=UP * 0.1), run_time=0.5)
        self.play(GrowArrow(a2), FadeIn(llm, shift=UP * 0.1), run_time=0.5)
        self.play(FadeIn(data, shift=UP * 0.1), GrowArrow(a_data), run_time=0.6)
        grad = VMobject().set_points_as_corners([llm.get_top() + UP * 0.05, [proj.get_x(), llm.get_y(), 0],
                                                 proj.get_bottom()])
        self.play(ShowPassingFlash(grad.set_stroke(C_FIRE, 8), time_width=0.6), run_time=1.1)
        self.play(FadeIn(ic_llm, scale=1.6), FadeIn(ic_proj, scale=1.6),
                  llm[0].animate.set_fill(C_FIRE, 0.45), proj[0].animate.set_fill(C_FIRE, 0.45), run_time=0.5)
        self.play(llm[0].animate.set_fill(C_FIRE_FILL, 1), proj[0].animate.set_fill(C_FIRE_FILL, 1),
                  FadeIn(cap1, shift=UP * 0.1), run_time=0.5)
        self.next_slide()

        # ===== Click: LoRA =====
        x = XS[1]
        xbar = Rectangle(width=2.6, height=0.17, fill_color=C_ACCENT, fill_opacity=0.9, stroke_width=0)
        xbar.move_to([x, -1.95, 0])
        hbar = xbar.copy().move_to([x, 0.42, 0])
        x_lbl = MathParts([("x", "x")], size=0.34).next_to(xbar, LEFT, buff=0.12)
        h_lbl = MathParts([("h", "h")], size=0.34).next_to(hbar, LEFT, buff=0.12)
        W = Rectangle(width=1.35, height=1.4, fill_color="#2E86C1", fill_opacity=0.35, stroke_color=C_PRETRAIN,
                      stroke_width=2).move_to([x - 0.65, -0.85, 0])
        W_lbl = VGroup(txt("Pretrained", 13, True, C_PRETRAIN), txt("weights", 13, True, C_PRETRAIN),
                       MathParts([("w", r"W\in\mathrm{ℝ}^{d\times d}")], size=0.27)).arrange(DOWN, buff=0.06).move_to(W)
        ic_W = icon("snowflake", C_ICE, 0.26).next_to(W, LEFT, buff=0.08)

        def trap(w_bottom, w_top, h):
            return Polygon([-w_bottom / 2, -h / 2, 0], [w_bottom / 2, -h / 2, 0], [w_top / 2, h / 2, 0],
                           [-w_top / 2, h / 2, 0], fill_color=C_LORA, fill_opacity=0.35, stroke_color=C_LORA,
                           stroke_width=2)

        A = trap(1.25, 0.45, 0.55).move_to([x + 0.8, -1.25, 0])
        B = trap(0.45, 1.25, 0.55).move_to([x + 0.8, -0.45, 0])
        A_lbl = MathParts([("a", "A")], size=0.3).move_to(A).shift(DOWN * 0.06)
        B_lbl = MathParts([("b", "B")], size=0.3).move_to(B).shift(UP * 0.06)
        r_lbl = MathParts([("r", "r")], size=0.26).set_color(GREY_B).move_to([x + 1.08, -0.85, 0])
        ic_AB = icon("fire", C_FIRE, 0.3).move_to([x + 1.65, -0.85, 0])
        init = MathParts([("i", r"B=0,\;A\sim\mathcal{N}(0,\sigma^2)")], size=0.2).set_color(GREY_B)
        plus = MathParts([("p", "+")], size=0.38).move_to([x + 0.08, 0.12, 0])

        def tiny(p0, p1, color=GREY_B):
            return Arrow(p0, p1, buff=0.04, color=color, stroke_width=2.5, tip_length=0.12)

        ar_xW = tiny([W.get_x(), xbar.get_top()[1], 0], W.get_bottom())
        ar_xA = tiny([A.get_x(), xbar.get_top()[1], 0], A.get_bottom())
        ar_AB = tiny(A.get_top(), B.get_bottom())
        ar_Wh = tiny(W.get_top(), [W.get_x(), hbar.get_bottom()[1], 0])
        ar_Bh = tiny(B.get_top(), [B.get_x(), hbar.get_bottom()[1], 0])
        lora_eq = MathParts([("h", "h"), ("eq", "="), ("wx", "Wx"), ("plus", "+"), ("bax", "BAx")], size=0.34)
        lora_eq["wx"].set_color(C_PRETRAIN)
        lora_eq["bax"].set_color(C_LORA)
        lora_eq.move_to([x, -2.25, 0])
        init.next_to(lora_eq, RIGHT, buff=0.15).shift(DOWN * 0.03)
        VGroup(lora_eq, init).move_to([x, -2.33, 0])
        cap2 = caption(["only A and B train", "(rank r ≪ d: a tiny fraction)"], C_LORA, x)

        self.play(FadeIn(xbar), FadeIn(x_lbl), FadeIn(W), FadeIn(W_lbl), FadeIn(ic_W, scale=1.4), GrowArrow(ar_xW))
        self.play(GrowArrow(ar_xA), FadeIn(A), FadeIn(A_lbl), run_time=0.6)
        self.play(GrowArrow(ar_AB), FadeIn(B), FadeIn(B_lbl), FadeIn(r_lbl), run_time=0.6)
        self.play(GrowArrow(ar_Wh), GrowArrow(ar_Bh), FadeIn(plus), FadeIn(hbar), FadeIn(h_lbl), run_time=0.6)
        frozen_path = Line([W.get_x(), xbar.get_top()[1], 0], [W.get_x(), hbar.get_bottom()[1], 0])
        lora_path = Line([A.get_x(), xbar.get_top()[1], 0], [A.get_x(), hbar.get_bottom()[1], 0])
        self.play(ShowPassingFlash(frozen_path.set_stroke(WHITE, 6), time_width=0.5),
                  ShowPassingFlash(lora_path.set_stroke(C_LORA, 6), time_width=0.5), run_time=1.1)
        self.play(FadeIn(lora_eq, shift=UP * 0.1), FadeIn(init, shift=UP * 0.1), FadeIn(ic_AB, scale=1.6))
        self.play(FadeIn(cap2, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: Multimodal RAG =====
        x = XS[2]
        TOP_W, TOP_H = 2.05, 0.62
        q_img = ImageMobject(str(ASSETS / "evo_q_car.png")).set_height(0.32)
        q_txt = lines(["Query", "image + question"], size=10, buff=0.02, color=GREY_A)
        q_txt[0].set_color(WHITE)
        q_row = Group(q_img, q_txt).arrange(RIGHT, buff=0.08)
        q_box = RoundedRectangle(corner_radius=0.1, width=TOP_W, height=TOP_H, fill_color=C_PANEL,
                                 fill_opacity=1, stroke_color=GREY_B, stroke_width=1.5)
        query = Group(q_box, q_row.move_to(q_box)).move_to([x - 1.1, 0.3, 0])

        docs = VGroup(*[
            RoundedRectangle(corner_radius=0.04, width=0.3, height=0.38, fill_color=C_PANEL, fill_opacity=1,
                             stroke_color=C_ACCENT, stroke_width=1.2).shift(RIGHT * 0.05 * i + UP * 0.04 * i)
            for i in range(3)
        ])
        kb_txt = lines(["Knowledge base", "injection data"], size=11, buff=0.02, color=GREY_A)
        kb_txt[0].set_color(WHITE)
        kb = VGroup(docs, kb_txt).arrange(RIGHT, buff=0.1)
        kb_frame = RoundedRectangle(corner_radius=0.1, width=TOP_W, height=TOP_H, fill_color=C_PANEL,
                                    fill_opacity=1, stroke_color=C_ACCENT, stroke_width=1.5)
        kb_box = VGroup(kb_frame, kb.move_to(kb_frame)).move_to([x + 1.05, 0.3, 0])

        def stage(lines_, color, y, w=2.6):
            body = lines(lines_, size=12, buff=0.02, align=ORIGIN, color=GREY_A)
            body[0].set_color(WHITE)
            box = RoundedRectangle(corner_radius=0.1, width=w, height=body.height + 0.14, fill_color=C_PANEL,
                                   fill_opacity=1, stroke_color=color, stroke_width=1.5).move_to([x, y, 0])
            return VGroup(box, body.move_to(box))

        retr = stage(["Retriever", "text-only · image-only · UniIR"], C_RAG, -0.45, 2.9)
        prompt = stage(["Prompt", "top-k docs + query"], C_RAG, -1.05)
        lmm = stage(["LLaVA-v1.5 (frozen)", "no training"], C_ICE, -1.65)
        ic_lmm = icon("snowflake", C_ICE, 0.26).next_to(lmm, LEFT, buff=0.1)
        ans = stage(["Short answer", "“Xiaomi Auto Company”"], C_GOAL, -2.25)
        ans[1][1].set_color(C_GOAL)

        r_arrows = VGroup(
            tiny(q_box.get_bottom(), [retr.get_left()[0] + 0.6, retr.get_top()[1], 0]),
            tiny(kb_frame.get_bottom(), [retr.get_right()[0] - 0.6, retr.get_top()[1], 0]),
            tiny(retr.get_bottom(), prompt.get_top()),
            tiny(prompt.get_bottom(), lmm.get_top()),
            tiny(lmm.get_bottom(), ans.get_top()),
        )
        cap3 = caption(["weights untouched:", "knowledge lives in the prompt"], C_RAG, x)

        self.play(FadeIn(query, shift=DOWN * 0.1), FadeIn(kb_box, shift=DOWN * 0.1))
        self.play(GrowArrow(r_arrows[0]), GrowArrow(r_arrows[1]), FadeIn(retr, shift=DOWN * 0.1), run_time=0.6)
        fetched = docs[-1].copy()
        self.play(fetched.animate.move_to(retr).scale(0.8), run_time=0.7)
        self.play(GrowArrow(r_arrows[2]), FadeIn(prompt, shift=DOWN * 0.1),
                  fetched.animate.move_to(prompt.get_right() + LEFT * 0.3), run_time=0.6)
        self.play(FadeOut(fetched), GrowArrow(r_arrows[3]), FadeIn(lmm, shift=DOWN * 0.1),
                  FadeIn(ic_lmm, scale=1.4), run_time=0.6)
        self.play(GrowArrow(r_arrows[4]), FadeIn(ans, shift=DOWN * 0.1), run_time=0.6)
        self.play(FadeIn(cap3, shift=UP * 0.1))
        self.next_slide()

        self.im = dict(kicker=ar["kicker"], heading=ar["heading"], cite=ar["cite"])

    # ---------- Slide 14: how MMEvoke is built (paper Sec. 3.1, Fig. 2) ----------
    def benchmark_construction(self):
        im = self.im
        C_RED = "#E74C3C"

        def pic(name, h):
            return ImageMobject(str(ASSETS / f"{name}.png")).set_height(h)

        def framed(m, color=GREY_C):
            return Group(m, SurroundingRectangle(m, buff=0, color=color, stroke_width=1.5))

        def para(strs, size=14, color=GREY_A, buff=0.05, t2c=None):
            return VGroup(*[Text(x_, font=FONT, font_size=size, color=color, t2c=t2c or {}) for x_ in strs]
                          ).arrange(DOWN, aligned_edge=LEFT, buff=buff)

        def arrow(a, b, color=GREY_B):
            return Arrow(a, b, buff=0.1, color=color, stroke_width=3, max_tip_length_to_length_ratio=0.2)

        # ===== Transition: clear the methods, show the 4-step pipeline =====
        keep = {im["kicker"], im["heading"], im["cite"]}
        others = [m for m in self.mobjects if m not in keep]
        sub = txt("how the benchmark is built", 18, color=GREY_B)
        sub.next_to(im["heading"], RIGHT, buff=0.35).align_to(im["heading"], DOWN).shift(UP * 0.05)

        steps_spec = [("1", "Collect", "new & popular knowledge"), ("2", "Summarize", "long text → summary"),
                      ("3", "Generate QA", "+ find query images"), ("4", "Manual check", "humans pick images")]
        steps = VGroup()
        for n, t, st in steps_spec:
            num = VGroup(Circle(radius=0.17, color=C_ACCENT, stroke_width=2), txt(n, 15, True, C_ACCENT))
            num[1].move_to(num[0])
            body = VGroup(txt(t, 17, True), txt(st, 12, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
            row = VGroup(num, body).arrange(RIGHT, buff=0.15)
            box = RoundedRectangle(corner_radius=0.15, width=2.75, height=0.75, fill_color=C_PANEL, fill_opacity=1,
                                   stroke_color=GREY_C, stroke_width=1.5)
            steps.add(VGroup(box, row.move_to(box)))
        steps.arrange(RIGHT, buff=0.42).move_to([0, 1.75, 0])
        links = VGroup(*[Arrow(steps[i].get_right(), steps[i + 1].get_left(), buff=0.06, color=GREY_B,
                               stroke_width=2.5, tip_length=0.14) for i in range(3)])

        self.play(*[FadeOut(m) for m in others], run_time=0.7)
        self.play(FadeIn(sub, shift=LEFT * 0.1),
                  LaggedStart(*[FadeIn(st_, shift=RIGHT * 0.15) for st_ in steps], lag_ratio=0.15),
                  LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.15), run_time=1.4)
        self.next_slide()

        def activate(i):
            anims = []
            for j, st_ in enumerate(steps):
                on = j == i
                anims.append(st_[0].animate.set_stroke(C_ACCENT if on else GREY_C, 3 if on else 1.5)
                             .set_fill(C_PANEL if on else "#111820", 1))
                op = 1 if on or j < i else 0.45
                anims.append(st_[1][0][1].animate.set_opacity(op))  # number (not the circle)
                anims.append(st_[1][1].animate.set_opacity(op))
                anims.append(st_[1][0][0].animate.set_stroke(opacity=op))
            return anims

        W_Y = -0.9  # work-area centre

        # ===== Click: step 1 - collect =====
        cnn = VGroup(RoundedRectangle(corner_radius=0.08, width=1.0, height=0.55, fill_color=C_RED, fill_opacity=1,
                                      stroke_width=0), txt("CNN", 22, True))
        cnn[1].move_to(cnn[0])
        cnn_lbl = txt("news (2024+)", 14, color=GREY_A).next_to(cnn, DOWN, buff=0.08)
        wiki = VGroup(Circle(radius=0.28, color=WHITE, stroke_width=2), txt("W", 22, True))
        wiki[1].move_to(wiki[0])
        wiki_lbl = txt("new Wikipedia entries", 14, color=GREY_A).next_to(wiki, DOWN, buff=0.08)
        srcs = Group(Group(cnn, cnn_lbl), Group(wiki, wiki_lbl)).arrange(DOWN, buff=0.4).move_to([-5.0, W_Y, 0])
        filt = chip(para(["keep only popular items:", "• news: many similar Google results",
                          "• entities: high page views"], 14, WHITE), C_ACCENT, pad=0.16).move_to([-1.5, W_Y, 0])
        filt[1][0].set_color(C_ACCENT)
        nobel = framed(pic("mm_nobel", 1.55))
        nobel_cap = VGroup(txt("News · Science", 13, True, C_S2),
                           txt("Nobel Prize in Physics 2024", 15, True),
                           txt("article: 1,000+ words", 13, color=GREY_B)).arrange(DOWN, buff=0.04)
        item = Group(nobel, nobel_cap).arrange(DOWN, buff=0.12).move_to([3.4, W_Y, 0])
        a1 = arrow(srcs.get_right(), filt.get_left())
        a2 = arrow(filt.get_right(), item.get_left())
        work1 = Group(srcs, filt, item, a1, a2)

        self.play(*activate(0))
        self.play(FadeIn(srcs, shift=RIGHT * 0.15))
        self.play(GrowArrow(a1), FadeIn(filt, shift=RIGHT * 0.15))
        self.play(GrowArrow(a2), FadeIn(item, shift=RIGHT * 0.15))
        self.next_slide()

        # ===== Click: step 2 - summarize =====
        article = VGroup(*[Line(LEFT * 0.9, RIGHT * (0.9 if k % 4 else 0.4), color=GREY_C, stroke_width=4)
                           for k in range(9)]).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        art_box = chip(article, GREY_C, pad=0.2)
        art_lbl = txt("1,000+ words", 13, True, C_RED).next_to(art_box, DOWN, buff=0.08)
        article_g = VGroup(art_box, art_lbl).move_to([-4.3, W_Y, 0])
        gpt = chip(VGroup(txt("GPT-4o", 17, True), txt("+ strict rules & examples", 12, color=GREY_B)
                          ).arrange(DOWN, buff=0.04), C_GOAL, pad=0.15).move_to([-1.3, W_Y, 0])
        summary = chip(para(["The 2024 Nobel Prize in Physics was", "awarded to John Hopfield and Geoffrey",
                             "Hinton for pioneering contributions to", "machine learning…"], 14, WHITE,
                            t2c={"John Hopfield": C_ACCENT, "Geoffrey": C_ACCENT, "Hinton": C_ACCENT}),
                       C_ACCENT, pad=0.18).move_to([3.3, W_Y, 0])
        sum_lbl = txt("Knowledge summary (avg. ~96 words)", 13, True, C_ACCENT).next_to(summary, UP, buff=0.08)
        b1 = arrow(article_g.get_right(), gpt.get_left())
        b2 = arrow(gpt.get_right(), summary.get_left())

        self.play(*activate(1), FadeOut(Group(srcs, filt, a1, a2, nobel_cap)),
                  nobel.animate.scale(0.55).move_to([-5.2, 0.62, 0]), run_time=0.8)
        self.play(FadeIn(article_g, shift=RIGHT * 0.15))
        self.play(GrowArrow(b1), FadeIn(gpt, scale=0.9))
        self.play(GrowArrow(b2), FadeIn(summary[0]), FadeIn(sum_lbl), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.1) for l in summary[1]], lag_ratio=0.2), run_time=1.0)
        self.next_slide()

        # ===== Click: step 3 - QA + query images =====
        qa = chip(para(["Q: Who shared the Nobel Prize in Physics", "    with the person in the image?",
                        "A: John Hopfield", "Object: Geoffrey Hinton  ·  Hypernym: Person"], 14, WHITE,
                       t2c={"Q:": C_ACCENT, "A:": C_GOAL, "Object:": GREY_B, "Hypernym:": GREY_B,
                            "John Hopfield": C_GOAL}), GREY_B, pad=0.16)
        qa.move_to([0, 0.2, 0]).align_to([-5.0, 0, 0], LEFT).shift(DOWN * 0.15)
        qa_lbl = txt("GPT-4o writes a VQA pair from the summary", 13, True, GREY_B).next_to(qa, UP, buff=0.08,
                                                                                           aligned_edge=LEFT)
        search = chip(VGroup(txt("Google image search:", 13, color=GREY_B),
                             txt("“Geoffrey Hinton Person”", 14, True)).arrange(DOWN, buff=0.04),
                      C_PRETRAIN, pad=0.12).move_to([0, W_Y - 1.1, 0]).align_to(qa, LEFT)
        names = ["mm_h1", "mm_h2", "mm_h3", "mm_outlier"]
        cands = Group(*[framed(pic(n, 1.05)) for n in names]).arrange(RIGHT, buff=0.15)
        cands.move_to([2.6, W_Y + 0.15, 0])
        kmeans = txt("CLIP features + K-means → drop outliers", 14, True, C_PRETRAIN).next_to(cands, DOWN, buff=0.2)
        c1 = arrow(qa.get_bottom() + LEFT * 0.6, search.get_top() + LEFT * 0.6, GREY_B)
        c2 = arrow(search.get_right(), [cands.get_left()[0], search.get_y(), 0] + UP * 0.0)
        c2 = Arrow(search.get_right(), cands.get_corner(DL) + UP * 0.3, buff=0.1, color=GREY_B, stroke_width=3,
                   tip_length=0.16)
        cross = VGroup(Line(UL, DR), Line(UR, DL)).set_stroke(C_RED, 8).scale(0.35).move_to(cands[-1])

        self.play(*activate(2), FadeOut(Group(article_g, gpt, b1, b2)),
                  FadeOut(Group(summary, sum_lbl), shift=LEFT * 0.3), run_time=0.8)
        self.play(FadeIn(qa_lbl), FadeIn(qa, shift=DOWN * 0.1))
        self.play(GrowArrow(c1), FadeIn(search, shift=DOWN * 0.1))
        self.play(GrowArrow(c2), LaggedStart(*[FadeIn(c, shift=LEFT * 0.15) for c in cands], lag_ratio=0.15))
        self.play(FadeIn(kmeans, shift=UP * 0.1))
        self.play(Create(cross), run_time=0.5)
        self.play(FadeOut(Group(cands[-1], cross), shift=DOWN * 0.3), run_time=0.6)
        self.next_slide()

        # ===== Click: step 4 - manual check =====
        human = icon("circle-user", GREY_A, 0.7).move_to([-1.0, W_Y + 0.15, 0])
        human_lbl = txt("annotator", 13, color=GREY_B).next_to(human, DOWN, buff=0.06)
        keep_q = cands[0]
        tick = VMobject().set_points_as_corners([LEFT * 0.5 + UP * 0.05, DOWN * 0.45, RIGHT * 0.8 + UP * 0.75])
        tick.set_stroke(C_GOAL, 8).scale(0.22)
        tick_q = tick.copy().next_to(keep_q, UP, buff=0.08)
        tick_k = tick.copy().next_to(nobel, RIGHT, buff=0.1)
        pick_lbl = txt("picks the cleanest knowledge image & query image", 14, True, C_GOAL)
        pick_lbl.next_to(cands[:3], DOWN, buff=0.2)

        self.play(*activate(3), FadeOut(Group(qa, qa_lbl, search, c1, c2, kmeans)), run_time=0.7)
        self.play(FadeIn(human, scale=0.8), FadeIn(human_lbl))
        self.play(cands[1].animate.set_opacity(0.3), cands[2].animate.set_opacity(0.3),
                  Create(tick_q), keep_q[1].animate.set_color(C_GOAL), run_time=0.7)
        self.play(Create(tick_k), nobel[1].animate.set_color(C_GOAL), FadeIn(pick_lbl, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: the final data format =====
        PY = -0.7

        def panel(title_m, color, x, w):
            box = RoundedRectangle(corner_radius=0.2, width=w, height=3.3, fill_color=C_PANEL, fill_opacity=1,
                                   stroke_color=color, stroke_width=2).move_to([x, PY, 0])
            title_m.move_to(box.get_top() + DOWN * 0.3)
            return box, title_m

        inj_t = VGroup(txt("Injection data", 18, True, C_ACCENT),
                       MathParts([("d", r"\mathcal{D}_K")], size=0.36).set_color(C_ACCENT)).arrange(RIGHT, buff=0.2)
        ev_t = VGroup(txt("Evaluation data", 18, True, C_GOAL),
                      MathParts([("d", r"\mathcal{D}_Q")], size=0.36).set_color(C_GOAL)).arrange(RIGHT, buff=0.2)
        inj_box, inj_t = panel(inj_t, C_ACCENT, -2.9, 7.0)
        ev_box, ev_t = panel(ev_t, C_GOAL, 3.75, 5.7)

        k_img_target = [-4.85, PY - 0.2, 0]
        k_text = para(["Heuristic query:", "Would you mind providing details", "about the science news in the image?",
                       "Knowledge summary:", "The 2024 Nobel Prize in Physics was", "awarded to John Hopfield and",
                       "Geoffrey Hinton for pioneering…"], 13, GREY_A, buff=0.05,
                      t2c={"Heuristic query:": WHITE, "Knowledge summary:": WHITE})
        k_text.move_to([-1.75, PY - 0.2, 0])
        q_text = para(["Query:", "Who shared the Nobel Prize in", "Physics with the person in", "the image?",
                       "Ground truth:", "John Hopfield"], 13, GREY_A, buff=0.05,
                      t2c={"Query:": WHITE, "Ground truth:": WHITE, "John Hopfield": C_GOAL})
        q_text.move_to([4.6, PY - 0.2, 0])
        use_inj = txt("injected via fine-tuning / LoRA, or the RAG knowledge base", 13, True, C_ACCENT)
        use_inj.next_to(inj_box, DOWN, buff=0.12)
        use_ev = txt("tests the injected model", 13, True, C_GOAL).next_to(ev_box, DOWN, buff=0.12)
        q_lbl = txt("query image", 12, color=GREY_B)

        self.play(FadeOut(Group(human, human_lbl, cands[1], cands[2], tick_q, tick_k, pick_lbl)), run_time=0.6)
        for m in (inj_box, ev_box):
            m.set_z_index(-1)
        nobel.set_z_index(3)
        keep_q.set_z_index(3)
        self.play(FadeIn(inj_box), FadeIn(inj_t), FadeIn(ev_box), FadeIn(ev_t),
                  nobel.animate.set_height(1.5).move_to(k_img_target),
                  keep_q.animate.set_height(1.6).move_to([2.3, PY - 0.2, 0]), run_time=1.2)
        q_lbl.next_to(keep_q, DOWN, buff=0.06)
        k_lbl = txt("knowledge image", 12, color=GREY_B).next_to(nobel, DOWN, buff=0.06)
        self.play(FadeIn(k_text, shift=LEFT * 0.1), FadeIn(q_text, shift=LEFT * 0.1), FadeIn(k_lbl), FadeIn(q_lbl))
        self.play(FadeIn(use_inj, shift=UP * 0.1), FadeIn(use_ev, shift=UP * 0.1))
        self.next_slide()

        self.bc = dict(kicker=im["kicker"], heading=im["heading"], cite=im["cite"], sub=sub)

    # ---------- Slide 15: metrics (CEM, F1) then results (paper Table 2 + Fig. 4) ----------
    def metrics_and_results(self):
        bc = self.bc
        C_RED = "#E74C3C"
        TRUTH = ["Thomas", "Matthew", "Crooks"]

        def word_chip(w, color, size=15):
            return chip(txt(w, size, True, color), color, pad=0.08)

        def mark(ok):
            if ok:
                m = VMobject().set_points_as_corners([LEFT * 0.5 + UP * 0.05, DOWN * 0.45, RIGHT * 0.8 + UP * 0.75])
                return m.set_stroke(C_GOAL, 6).scale(0.14)
            return VGroup(Line(UL, DR), Line(UR, DL)).set_stroke(C_RED, 6).scale(0.1)

        def new_sub(text):
            t = txt(text, 18, color=GREY_B)
            return t.move_to(bc["sub"]).align_to(bc["sub"], LEFT)

        # ===== Transition: the example question =====
        keep = {bc["kicker"], bc["heading"], bc["cite"], bc["sub"]}
        others = [m for m in self.mobjects if m not in keep]
        img = ImageMobject(str(ASSETS / "evo_trump.png")).set_height(1.3)
        img_f = Group(img, SurroundingRectangle(img, buff=0, color=GREY_C, stroke_width=1.5))
        q = VGroup(
            Text("Q: Who tried to assassinate the person in the image at a", font=FONT, font_size=16,
                 t2c={"Q:": C_ACCENT}),
            txt("    campaign rally in Butler, Pennsylvania?", 16),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        truth_lbl = txt("Ground truth:", 16, True, C_GOAL)
        truth = VGroup(*[word_chip(w, C_GOAL) for w in TRUTH]).arrange(RIGHT, buff=0.1)
        truth_row = VGroup(truth_lbl, truth).arrange(RIGHT, buff=0.2)
        qcol = VGroup(q, truth_row).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        qcard = Group(img_f, qcol).arrange(RIGHT, buff=0.35).move_to([0, 1.35, 0])

        self.play(*[FadeOut(m) for m in others], Transform(bc["sub"], new_sub("how answers are scored")),
                  run_time=0.8)
        self.play(FadeIn(img_f, shift=RIGHT * 0.15), FadeIn(q, shift=RIGHT * 0.15))
        self.play(FadeIn(truth_lbl), LaggedStart(*[FadeIn(c, scale=0.8) for c in truth], lag_ratio=0.15))
        self.next_slide()

        CY = -1.2

        def card(title, color, x):
            box = RoundedRectangle(corner_radius=0.2, width=6.3, height=3.15, fill_color=C_PANEL, fill_opacity=1,
                                   stroke_color=color, stroke_width=2).move_to([x, CY, 0])
            t = txt(title, 19, True, color).move_to(box.get_top() + DOWN * 0.3)
            return box, t

        # ===== Click: CEM =====
        cem_box, cem_t = card("Cover Exact Match (CEM)", C_ACCENT, -3.3)
        cem_def = VGroup(txt("1 if the whole ground truth appears", 15),
                         txt("in the answer, otherwise 0", 15)).arrange(DOWN, buff=0.04)
        cem_def.next_to(cem_t, DOWN, buff=0.18)

        def ex_row(answer, score, ok, x, y):
            a = Text(f"“{answer}”", font=FONT, font_size=15, color=WHITE)
            arrow = txt("→", 15, color=GREY_B)
            sc = txt(score, 16, True, C_GOAL if ok else C_RED)
            row = VGroup(a, arrow, sc, mark(ok)).arrange(RIGHT, buff=0.18)
            return row.move_to([x, y, 0])

        cem_rows = VGroup(
            ex_row("Thomas matthew crooks", "CEM = 1", True, -3.3, CY - 0.05),
            ex_row("crooks", "CEM = 0", False, -3.3, CY - 0.5),
            ex_row("Bernie Sanders", "CEM = 0", False, -3.3, CY - 0.95),
        )
        self.play(FadeIn(cem_box), FadeIn(cem_t))
        self.play(FadeIn(cem_def, shift=UP * 0.1))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in cem_rows], lag_ratio=0.3), run_time=1.4)
        note = txt("all-or-nothing: “crooks” gets no credit", 13, True, GREY_B)
        note.move_to([-3.3, CY - 1.38, 0])
        self.play(Indicate(cem_rows[1], color=C_RED, scale_factor=1.05), FadeIn(note))
        self.next_slide()

        # ===== Click: F1 =====
        f1_box, f1_t = card("F1 score (word overlap)", C_PRETRAIN, 3.3)
        f1_def = MathParts([("p", r"P=\frac{\mathrm{overlap}}{\#\,\mathrm{answer\ words}}\quad"),
                            ("r", r"R=\frac{\mathrm{overlap}}{\#\,\mathrm{truth\ words}}\quad"),
                            ("f", r"F_1=\frac{2PR}{P+R}")], size=0.3)
        f1_def.next_to(f1_t, DOWN, buff=0.18)
        ans_chip = word_chip("crooks", WHITE)
        ans_lbl = txt("answer:", 14, color=GREY_B)
        ans_row = VGroup(ans_lbl, ans_chip).arrange(RIGHT, buff=0.15)
        t_chips = VGroup(*[word_chip(w.lower() if False else w, C_GOAL, 14) for w in TRUTH]).arrange(RIGHT, buff=0.1)
        t_lbl = txt("truth:", 14, color=GREY_B)
        t_row = VGroup(t_lbl, t_chips).arrange(RIGHT, buff=0.15)
        words = VGroup(ans_row, t_row).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([2.15, CY - 0.5, 0])
        match = Line(ans_chip.get_bottom(), t_chips[2].get_top(), color=C_GOAL, stroke_width=3)
        calc = VGroup(
            txt("overlap = 1 word", 14, color=GREY_A),
            txt("P = 1/1,  R = 1/3", 14, color=GREY_A),
            txt("F1 = 0.5", 18, True, C_PRETRAIN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([5.2, CY - 0.5, 0])
        f1_note = txt("partial credit for partly-right answers", 13, True, GREY_B)
        f1_note.move_to([3.3, CY - 1.38, 0])

        self.play(FadeIn(f1_box), FadeIn(f1_t))
        self.play(FadeIn(f1_def, shift=UP * 0.1))
        self.play(FadeIn(words, shift=RIGHT * 0.1))
        self.play(Create(match), Indicate(t_chips[2], color=C_GOAL), Indicate(ans_chip, color=C_GOAL))
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.1) for c in calc], lag_ratio=0.3), FadeIn(f1_note))
        self.next_slide()

        # ===== Results: CEM bar chart, built group by group =====
        groups = [
            ("Fine-\ntuning", C_S2, [("Vanilla LLaVA-v1.5", 4.89, None, None),
                                   ("Full-FT", 18.02, "“A man was arrested after…”", False),
                                   ("LoRA", 15.23, "“Anthony Gualtieri”", False)]),
            ("MM-RAG", C_GOAL, [("RAG · text-only", 24.05, "“Gunman”", False),
                                ("RAG · image-only", 25.25, "“Trump”", False),
                                ("RAG · UniIR", 40.68, "“Bernie Sanders”", False)]),
            ("AI web\nsearch", C_PRETRAIN, [("Gemini-2.0-Flash", 18.21, "“No one”", False),
                                            ("Gemini-2.5-Pro", 44.19, "“Thomas Matthew Crooks”", True),
                                            ("Perplexity AI", 48.27, "“Thomas matthew crooks”", True),
                                            ("GPT-4.1", 39.61, "“Thomas matthew crooks”", True)]),
            ("Sufficient\ncontext", C_ACCENT, [("LLaVA-v1.5", 56.13, "“Thomas matthew crooks”", True),
                                              ("Qwen-VL-Chat", 48.96, "“crooks”  (F1 = 0.5)", False),
                                              ("Gemini-2.5-Pro", 72.15, "“Thomas matthew crooks”", True),
                                              ("GPT-4.1", 75.02, "“Thomas matthew crooks”", True)]),
        ]
        X0, UNIT, STEP, BAR_H = -2.75, 0.042, 0.268, 0.19
        NAME_X, GROUP_X, ANS_X = -2.9, -6.35, 1.9
        y = 1.5
        rows_by_group = []
        for gname, col, rows in groups:
            items = []
            gy0 = y
            for name, v, ans, ok in rows:
                nm = txt(name, 13, color=GREY_A).move_to([0, y, 0]).align_to([NAME_X, 0, 0], RIGHT)
                bar = RoundedRectangle(corner_radius=0.04, width=max(v * UNIT, 0.05), height=BAR_H, fill_color=col,
                                       fill_opacity=1, stroke_width=0).move_to([X0 + v * UNIT / 2, y, 0])
                val = txt(f"{v:.1f}", 13, True).next_to(bar, RIGHT, buff=0.08)
                a_m = None
                if ans is not None:
                    a_txt = txt(ans, 12, color=GREY_A)
                    a_m = VGroup(mark(ok).scale(0.8), a_txt).arrange(RIGHT, buff=0.1)
                    a_m.move_to([0, y, 0]).align_to([ANS_X, 0, 0], LEFT)
                items.append((nm, bar, val, a_m))
                y -= STEP
            gl = lines(gname.split("\n"), size=14, bold=True, color=col, buff=0.03)
            gl.move_to([0, (gy0 + y + STEP) / 2, 0]).align_to([GROUP_X, 0, 0], LEFT)
            brace = Line([GROUP_X + 1.55, gy0 + 0.1, 0], [GROUP_X + 1.55, y + STEP - 0.1, 0], color=col, stroke_width=2)
            rows_by_group.append((gl, brace, items))
            y -= 0.12

        axis = Line([X0, 1.67, 0], [X0, y + STEP - 0.05, 0], color=GREY_D, stroke_width=2)
        chart_t = txt("CEM (%) on MMEvoke", 15, True).move_to([0, 1.92, 0]).align_to([X0, 0, 0], LEFT)
        ans_t = txt("answer to the Butler question", 13, True, GREY_B).move_to([0, 1.92, 0]).align_to([ANS_X, 0, 0], LEFT)

        def show_group(k, takeaway=None):
            gl, brace, items = rows_by_group[k]
            self.play(FadeIn(gl), Create(brace), run_time=0.4)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(nm), GrowFromEdge(bar, LEFT)) for nm, bar, _, _ in items],
                                  lag_ratio=0.2), run_time=1.0)
            self.play(*[FadeIn(val, shift=LEFT * 0.1) for _, _, val, _ in items],
                      *[FadeIn(a, shift=LEFT * 0.1) for *_, a in items if a is not None], run_time=0.6)

        metric_stuff = Group(img_f, q, truth_row, cem_box, cem_t, cem_def, cem_rows, note, f1_box, f1_t, f1_def,
                             words, match, calc, f1_note)
        self.play(FadeOut(metric_stuff), Transform(bc["sub"], new_sub("results (Table 2, Figure 4)")), run_time=0.8)
        self.play(Create(axis), FadeIn(chart_t), FadeIn(ans_t))
        show_group(0)
        take = txt("fine-tuning barely injects the new knowledge (≤ 18% CEM)", 15, True, C_S2)
        take.move_to([0, -2.98, 0])
        self.play(FadeIn(take, shift=UP * 0.1))
        self.next_slide()

        def swap_take(text, color):
            t = txt(text, 15, True, color).move_to([0, -2.98, 0])
            self.play(Transform(take, t), run_time=0.5)

        show_group(1)
        swap_take("retrieval helps, but the best RAG reaches only 40.7% CEM", C_GOAL)
        self.next_slide()

        show_group(2)
        swap_take("commercial search engines: at most 48.3% CEM", C_PRETRAIN)
        self.next_slide()

        show_group(3)
        swap_take("even with the right document in context: at most 75% CEM", C_ACCENT)
        self.next_slide()

        # ===== Click: spotlight the fine-tuning failure =====
        others_ = VGroup()
        for k, (gl, brace, items) in enumerate(rows_by_group):
            if k == 0:
                others_.add(items[0][0], items[0][1], items[0][2])  # vanilla row
            else:
                others_.add(gl, brace, *[m for it in items for m in it if m is not None])
        _, _, ft_items = rows_by_group[0]
        ft = VGroup(*[m for it in ft_items[1:] for m in it if m is not None])
        ft_box = SurroundingRectangle(ft, buff=0.1, corner_radius=0.08, color=C_RED, stroke_width=3)
        ft_lbl = txt("trained on the new fact, still can\u2019t answer it", 12, True, C_RED)
        ft_lbl.move_to([0.0, ft.get_y(), 0]).align_to([1.75, 0, 0], RIGHT)
        self.play(others_.animate.set_opacity(0.25), run_time=0.6)
        self.play(Create(ft_box), run_time=0.8)
        self.play(FadeIn(ft_lbl, shift=UP * 0.1), Indicate(ft, color=C_RED, scale_factor=1.03))
        swap_take("Full-FT & LoRA fail to inject the knowledge\u2026 and that\u2019s not their only problem", C_RED)
        self.next_slide()


    # ---------- Slide 16: capability degradation after injection (paper Table 3) ----------
    def degradation(self):
        bc = self.bc
        C_RED = "#E74C3C"
        dims = [("Comprehensive", ["MME", "MMBench"]), ("OCR", ["SEED-B2P", "OCRBench"]),
                ("Multidisciplinary", ["ScienceQA", "MMMU"]), ("Instruction", ["MIA-Bench"]),
                ("Multi-Round", ["MMDU"]), ("Mathematical", ["MathVista", "MathVision"]),
                ("Hallucination", ["POPE", "HallusionB."])]
        vanilla = [1865.56, 64.60, 38.78, 30.80, 69.83, 28.60, 66.33, 26.37, 25.50, 13.16, 86.87, 21.76]
        rows = [
            ("Full-FT", [956.80, 52.92, 31.44, 28.10, 67.13, 24.20, 25.25, 13.03, 24.70, 11.94, 74.22, 9.27],
             [48.71, 18.08, 18.93, 8.77, 3.87, 15.38, 61.93, 50.59, 3.14, 9.27, 14.56, 57.40]),
            ("LoRA", [1233.54, 53.87, 30.22, 25.70, 66.18, 21.40, 29.66, 13.70, 23.20, 12.83, 73.97, 8.78],
             [33.88, 16.61, 22.07, 16.56, 5.23, 25.17, 55.28, 48.05, 9.02, 2.51, 14.85, 59.65]),
        ]
        X_M, X0, STEP = -6.4, -4.75, 0.99
        cx = [X0 + STEP * k for k in range(12)]
        Y_DIM, Y_BENCH, Y_VAN = 1.75, 1.42, 1.0
        Y_ROWS = [0.38, -0.42]

        def fmt(v):
            return f"{v:,.2f}" if v >= 1000 else f"{v:.2f}"

        # header
        dim_lbls, rules, k = VGroup(), VGroup(), 0
        for name, benches in dims:
            span = cx[k:k + len(benches)]
            mid = sum(span) / len(span)
            dim_lbls.add(txt(name, 11, True, GREY_B).move_to([mid, Y_DIM, 0]))
            rules.add(Line([span[0] - STEP * 0.45, Y_DIM - 0.15, 0], [span[-1] + STEP * 0.45, Y_DIM - 0.15, 0],
                           color=GREY_D, stroke_width=1.5))
            k += len(benches)
        bench_lbls = VGroup(*[txt(b, 11, True).move_to([cx[i], Y_BENCH, 0])
                              for i, b in enumerate(sum([d[1] for d in dims], []))])
        top = Line([X_M, Y_BENCH - 0.2, 0], [cx[-1] + 0.5, Y_BENCH - 0.2, 0], color=GREY_C, stroke_width=1.5)
        method_hdr = txt("Method", 13, True).move_to([0, Y_BENCH, 0]).align_to([X_M, 0, 0], LEFT)

        def row(name, vals, y, color=WHITE):
            nm = txt(name, 14, True, color).move_to([0, y, 0]).align_to([X_M, 0, 0], LEFT)
            cells = VGroup(*[txt(fmt(v), 13, color=GREY_A).move_to([cx[i], y, 0]) for i, v in enumerate(vals)])
            return nm, cells

        van_nm, van_cells = row("Vanilla", vanilla, Y_VAN, C_PRETRAIN)
        van_cells.set_color(WHITE)
        body = []
        for (name, vals, drops), y in zip(rows, Y_ROWS):
            nm, cells = row(name, vals, y + 0.12, C_S2 if name == "Full-FT" else "#BB8FCE")
            chips = VGroup()
            for i, d in enumerate(drops):
                sev = d / 62
                t = txt(f"↓{d:.1f}%", 11, True)
                bg = RoundedRectangle(corner_radius=0.05, width=0.82, height=0.24, stroke_width=0,
                                      fill_color=C_RED, fill_opacity=0.12 + 0.75 * sev).move_to([cx[i], y - 0.17, 0])
                chips.add(VGroup(bg, t.move_to(bg)))
            body.append((nm, cells, chips))
        mid_rule = Line([X_M, Y_VAN - 0.28, 0], [cx[-1] + 0.5, Y_VAN - 0.28, 0], color=GREY_D, stroke_width=1)
        bottom = Line([X_M, Y_ROWS[-1] - 0.4, 0], [cx[-1] + 0.5, Y_ROWS[-1] - 0.4, 0], color=GREY_C, stroke_width=1.5)

        # ===== Transition + table with raw numbers only =====
        keep = {bc["kicker"], bc["heading"], bc["cite"], bc["sub"]}
        others = [m for m in self.mobjects if m not in keep]
        sub = txt("general capabilities after injection (Table 3)", 18, color=GREY_B)
        sub.move_to(bc["sub"]).align_to(bc["sub"], LEFT)
        self.play(*[FadeOut(m) for m in others], Transform(bc["sub"], sub), run_time=0.8)
        setup = txt("LLaVA-v1.5 on 12 standard benchmarks, before vs. after injecting MMEvoke", 15, color=GREY_A)
        setup.move_to([0, 2.25, 0]).align_to([X_M, 0, 0], LEFT)
        self.play(FadeIn(setup), FadeIn(dim_lbls), Create(rules), FadeIn(bench_lbls), FadeIn(method_hdr),
                  Create(top), run_time=1.0)
        self.play(FadeIn(van_nm), LaggedStart(*[FadeIn(c) for c in van_cells], lag_ratio=0.05), Create(mid_rule))
        for nm, cells, _ in body:
            self.play(FadeIn(nm), LaggedStart(*[FadeIn(c, shift=DOWN * 0.08) for c in cells], lag_ratio=0.05),
                      run_time=0.9)
        self.play(Create(bottom), run_time=0.4)
        question = txt("What’s wrong?", 44, True, C_RED).move_to([0, -1.75, 0])
        self.play(FadeIn(question, scale=1.4), run_time=0.6)
        self.play(Wiggle(question))
        self.next_slide()

        # ===== Click: reveal - every score dropped =====
        self.play(FadeOut(question, shift=DOWN * 0.2), run_time=0.4)
        for _, _, chips in body:
            self.play(LaggedStart(*[FadeIn(c, scale=0.7) for c in chips], lag_ratio=0.05), run_time=1.0)
        worst = [0, 6, 7, 11]  # MME, MIA-Bench, MMDU, HallusionBench
        boxes = VGroup(*[
            SurroundingRectangle(VGroup(*[b[1][i] for b in body], *[b[2][i] for b in body]),
                                 buff=0.07, corner_radius=0.06, color=C_RED, stroke_width=3)
            for i in worst
        ])
        self.play(LaggedStart(*[Create(b) for b in boxes], lag_ratio=0.25), run_time=1.2)
        self.play(*[Indicate(bench_lbls[i], color=C_RED, scale_factor=1.2) for i in worst])
        call = VGroup(
            txt("Every benchmark gets worse: average drop ↓ 25.9% (Full-FT), ↓ 25.7% (LoRA)", 17, True),
            txt("worst: instruction following, multi-round QA, hallucination, comprehensive", 14, color=GREY_A),
        ).arrange(DOWN, buff=0.08).move_to([0, -1.55, 0])
        self.play(FadeIn(call, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: name it - capability degradation = goal ② violated =====
        name = Text("Capability degradation", font=FONT, font_size=30, weight=BOLD, color=C_RED)
        link = VGroup(txt("→ goal ② ", 17, True, C_PRETRAIN), txt("Knowledge Retention", 17, True, C_PRETRAIN),
                      txt(" is violated: learning the new erases the old", 17, color=GREY_A)).arrange(RIGHT, buff=0.06)
        punch = VGroup(name, link).arrange(DOWN, buff=0.12).move_to([0, -2.45, 0])
        self.play(call.animate.shift(UP * 0.25).set_opacity(0.7), run_time=0.4)
        self.play(Write(name), run_time=0.8)
        self.play(FadeIn(link, shift=UP * 0.1))
        self.next_slide()

        self.dg = dict(name=name)

    # ---------- Slide 17: catastrophic forgetting (concept) ----------
    def catastrophic_forgetting(self):
        bc, dg = self.bc, self.dg
        C_OLD, C_NEWK, C_RED = C_PRETRAIN, C_ACCENT, "#E74C3C"

        # ===== Transition: "capability degradation" has a name =====
        new_head = txt("Catastrophic Forgetting", size=40, bold=True)
        new_head.move_to(bc["heading"]).align_to(bc["heading"], LEFT)
        keep = {bc["kicker"], bc["heading"], bc["cite"], dg["name"]}
        others = [m for m in self.mobjects if m not in keep]
        self.play(*[FadeOut(m) for m in others], run_time=0.6)
        self.play(ReplacementTransform(dg["name"], new_head), FadeOut(bc["heading"]), run_time=1.1)
        defn = VGroup(
            Text("When a trained model is fine-tuned on new data, its performance on", font=FONT, font_size=19),
            Text("previously learned tasks or knowledge drops sharply.", font=FONT, font_size=19,
                 t2c={"drops sharply": C_RED}),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        defn.next_to(new_head, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(defn, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: parameter space + the pretrained model =====
        space = RoundedRectangle(corner_radius=0.2, width=7.2, height=3.9, fill_color="#0E1116", fill_opacity=1,
                                 stroke_color=GREY_D, stroke_width=1.5).move_to([-2.7, -1.0, 0])
        space_t = VGroup(txt("parameter space", 14, True, GREY_B),
                         MathParts([("t", r"(\theta_1,\theta_2,\ldots)")], size=0.26).set_color(GREY_B)
                         ).arrange(RIGHT, buff=0.12)
        space_t.move_to(space.get_corner(UL) + [0.2, -0.25, 0], aligned_edge=UL)

        C_O, R_O = np.array([-4.0, -1.15, 0]), (1.75, 1.25)
        C_N, R_N = np.array([-1.3, -0.95, 0]), (1.75, 1.25)

        def basin(center, radii, color):
            return VGroup(*[
                Ellipse(width=2 * radii[0] * f, height=2 * radii[1] * f, stroke_color=color, stroke_width=2,
                        fill_color=color, fill_opacity=0.08 + 0.1 * k).move_to(center)
                for k, f in enumerate([1.0, 0.68, 0.36])
            ])

        old_basin = basin(C_O, R_O, C_OLD)
        new_basin = basin(C_N, R_N, C_NEWK)
        old_lbl = txt("low loss on OLD data", 13, True, C_OLD).move_to([C_O[0] - 0.4, -2.68, 0])
        new_lbl = txt("low loss on NEW data", 13, True, C_NEWK).move_to([C_N[0] + 0.4, -2.68, 0])

        theta0 = C_O + np.array([-0.15, 0.05, 0])
        probe = Dot(theta0, radius=0.11, color=WHITE).set_z_index(5)
        probe_lbl = MathParts([("t", r"\theta_{\mathrm{pretrained}}")], size=0.26).next_to(probe, DOWN, buff=0.1)

        def perf(pt, c, r):
            d = np.hypot((pt[0] - c[0]) / r[0], (pt[1] - c[1]) / r[1])
            return 5 + 90 / (1 + d ** 6)

        # live meters
        MX, MW = 2.05, 3.9

        def meter(title, sub, color, y, fn):
            t = txt(title, 17, True, color).move_to([0, y + 0.45, 0]).align_to([MX, 0, 0], LEFT)
            st = txt(sub, 12, color=GREY_B).next_to(t, DOWN, buff=0.05, aligned_edge=LEFT)
            track = RoundedRectangle(corner_radius=0.08, width=MW, height=0.3, fill_color="#1A1F26", fill_opacity=1,
                                     stroke_color=GREY_D, stroke_width=1).move_to([MX + MW / 2, y - 0.25, 0])

            def bar():
                v = fn(probe.get_center())
                w = max(MW * v / 100, 0.02)
                b = RoundedRectangle(corner_radius=0.08, width=w, height=0.3, fill_color=color, fill_opacity=1,
                                     stroke_width=0)
                b.move_to(track.get_left() + RIGHT * w / 2)
                pct = txt(f"{v:.0f}%", 16, True).next_to(track, RIGHT, buff=0.12)
                return VGroup(b, pct)

            return VGroup(t, st, track), always_redraw(bar)

        old_m, old_bar = meter("Old knowledge & skills", "instruction following, OCR, hallucination tests…",
                               C_OLD, 0.15, lambda q: perf(q, C_O, R_O))
        new_m, new_bar = meter("New knowledge", "e.g. “who makes the Xiaomi SU7?”", C_NEWK, -1.65,
                               lambda q: perf(q, C_N, R_N))

        self.play(FadeIn(space), FadeIn(space_t))
        self.play(LaggedStart(*[GrowFromCenter(e) for e in old_basin[::-1]], lag_ratio=0.15), FadeIn(old_lbl))
        self.play(GrowFromCenter(probe), FadeIn(probe_lbl))
        self.play(FadeIn(old_m), FadeIn(new_m), FadeIn(old_bar), FadeIn(new_bar))
        self.next_slide()

        # ===== Click: fine-tune on the new data only =====
        path = CubicBezier(theta0, theta0 + np.array([0.9, 1.05, 0]), C_N + np.array([-0.9, 0.9, 0]), C_N)
        trail = TracedPath(probe.get_center, stroke_color=C_RED, stroke_width=4)
        step_lbl = txt("fine-tune on NEW data only", 14, True, C_RED)
        step_lbl.move_to([-1.0, 0.55, 0])
        self.play(LaggedStart(*[GrowFromCenter(e) for e in new_basin[::-1]], lag_ratio=0.15), FadeIn(new_lbl))
        self.add(trail)
        self.play(FadeOut(probe_lbl), FadeIn(step_lbl), run_time=0.4)
        self.play(MoveAlongPath(probe, path), run_time=3.0, rate_func=smooth)
        end_lbl = MathParts([("t", r"\theta_{\mathrm{fine\text{-}tuned}}")], size=0.26).next_to(probe, DOWN, buff=0.1)
        self.play(FadeIn(end_lbl))
        self.play(Indicate(old_bar, color=C_RED, scale_factor=1.05))
        why = VGroup(
            txt("The weights leave the region that", 15, True, C_RED),
            txt("encoded the old knowledge", 15, True, C_RED),
            txt("cf. Table 3: MIA-Bench ↓62%, HallusionBench ↓60%", 12, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        why.move_to([0, -2.6, 0]).align_to([MX, 0, 0], LEFT)
        self.play(FadeIn(why, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: what we actually want =====
        ghost = VGroup(trail.copy().set_stroke(opacity=0.45), Dot(C_N, radius=0.09, color=C_RED).set_opacity(0.6))
        self.remove(trail)
        self.add(ghost)
        overlap = np.array([-2.65, -1.05, 0])
        good_path = CubicBezier(theta0, theta0 + np.array([0.5, -0.45, 0]), overlap + np.array([-0.5, -0.4, 0]),
                                overlap)
        good_trail = TracedPath(probe.get_center, stroke_color=C_GOAL, stroke_width=4)
        self.play(FadeOut(end_lbl), FadeOut(step_lbl), FadeOut(why), probe.animate.move_to(theta0), run_time=0.8)
        self.add(good_trail)
        self.play(MoveAlongPath(probe, good_path), run_time=2.2, rate_func=smooth)
        star = Star(n=5, outer_radius=0.2, inner_radius=0.09, color=C_GOAL, fill_opacity=1).move_to(overlap)
        star.set_z_index(6)
        goal_lbl = lines(["good on", "both"], align=ORIGIN, size=13, bold=True, color=C_GOAL, buff=0.02)
        goal_lbl.next_to(star, RIGHT, buff=0.12).shift(DOWN * 0.45)
        self.play(GrowFromCenter(star), FadeIn(goal_lbl))
        ideal = VGroup(
            txt("Goal: learn the new fact without", 15, True, C_GOAL),
            txt("leaving the old region", 15, True, C_GOAL),
            txt("= adaptation ① + retention ② at the same time", 12, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        ideal.move_to([0, -2.6, 0]).align_to([MX, 0, 0], LEFT)
        self.play(FadeIn(ideal, shift=UP * 0.1))
        self.next_slide()

        self.cf = dict(heading=new_head)

    # ---------- Slide 18: preventing forgetting with data replay (paper Sec. 5.2, Table 3) ----------
    def replay(self):
        bc, cf = self.bc, self.cf
        C_OLD, C_NEWK, C_RED = C_PRETRAIN, C_ACCENT, "#E74C3C"

        # ===== Transition =====
        head = txt("Preventing Catastrophic Forgetting", size=40, bold=True)
        head.move_to(cf["heading"]).align_to(cf["heading"], LEFT)
        c1 = txt("When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations "
                 "(Jiang et al., 2026)", 15, color=C_LINK)
        c2 = txt("iCaRL: Incremental Classifier and Representation Learning (Rebuffi et al., 2017)", 15, color=C_LINK)
        cites = VGroup(*[VGroup(c, Underline(c, buff=0.04, color=C_LINK, stroke_width=1.2)) for c in (c1, c2)])
        cites.arrange(DOWN, buff=0.1).move_to([0, -3.45, 0])
        keep = {bc["kicker"], bc["cite"], cf["heading"]}
        others = [m for m in self.mobjects if m not in keep]
        self.play(*[FadeOut(m) for m in others], Transform(cf["heading"], head),
                  ReplacementTransform(bc["cite"], cites), run_time=1.0)

        idea = VGroup(
            Text("Idea — replay (rehearsal): while learning the new, keep reviewing a little of the old",
                 font=FONT, font_size=19, weight=BOLD, t2c={"replay (rehearsal)": C_GOAL}),
            txt("iCaRL (2017): keep a small memory of old examples and mix it into every new training stage",
                13, color=GREY_A),
        ).arrange(DOWN, buff=0.06)
        idea = chip(idea, C_GOAL, pad=0.16).move_to([0, 1.85, 0])
        self.play(FadeIn(idea, shift=DOWN * 0.1))
        self.next_slide()

        # ===== Click: the paper's recipe - mix ~942 old samples into the new data =====
        def pile(color, n, label, sub, center):
            cards = VGroup(*[
                RoundedRectangle(corner_radius=0.05, width=0.55, height=0.4, fill_color=C_PANEL, fill_opacity=1,
                                 stroke_color=color, stroke_width=1.5).shift(RIGHT * 0.07 * k + UP * 0.06 * k)
                for k in range(n)
            ])
            t = VGroup(txt(label, 15, True, color), txt(sub, 12, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT,
                                                                                       buff=0.03)
            g = VGroup(cards, t).arrange(RIGHT, buff=0.2).move_to(center)
            return g

        new_p = pile(C_NEWK, 5, "Injection data  \U0001D4DF_K", "9,422 new facts (MMEvoke)", [-4.75, 0.6, 0])
        new_p[1][0].become(VGroup(txt("Injection data ", 15, True, C_NEWK),
                                  MathParts([("d", r"\mathcal{D}_K")], size=0.3).set_color(C_NEWK)
                                  ).arrange(RIGHT, buff=0.05).move_to(new_p[1][0], aligned_edge=LEFT))
        old_p = pile(C_OLD, 3, "Replay data", "≈942 samples = 10% of MMEvoke’s size,", [-4.75, -1.0, 0])
        extra = lines(["randomly drawn from LLaVA-v1.5’s", "own training data"], size=12, color=GREY_A, buff=0.03)
        extra.next_to(old_p[1], DOWN, buff=0.03, aligned_edge=LEFT)
        old_p.add(extra)

        # mixed training set: 10 new : 1 old, shuffled
        order = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        slots = VGroup(*[
            RoundedRectangle(corner_radius=0.04, width=0.24, height=0.34, fill_color=C_NEWK if o == 0 else C_OLD,
                             fill_opacity=0.85, stroke_width=0) for o in order
        ]).arrange(RIGHT, buff=0.06).move_to([-0.45, -2.3, 0])
        mix_lbl = lines(["shuffled into one", "training set (≈ 10 : 1)"], align=ORIGIN, size=13, bold=True,
                        color=GREY_B, buff=0.02).next_to(slots, UP, buff=0.12)
        a_new = Arrow([new_p.get_right()[0] + 0.1, new_p.get_y(), 0], mix_lbl.get_top() + UP * 0.05, buff=0.08,
                      color=C_NEWK, stroke_width=2.5, tip_length=0.14)
        a_old = Arrow([old_p.get_right()[0] + 0.1, old_p[1][0].get_y(), 0], mix_lbl.get_left() + LEFT * 0.05,
                      buff=0.08, color=C_OLD, stroke_width=2.5, tip_length=0.14)

        self.play(FadeIn(new_p, shift=RIGHT * 0.15))
        self.play(FadeIn(old_p, shift=RIGHT * 0.15))
        self.play(GrowArrow(a_new), GrowArrow(a_old), run_time=0.5)
        flyers = []
        for k, (o, slot) in enumerate(zip(order, slots)):
            src = (new_p if o == 0 else old_p)[0][-1].copy()
            flyers.append(src)
        self.play(LaggedStart(*[f.animate.become(sl) for f, sl in zip(flyers, slots)], lag_ratio=0.08), run_time=1.6)
        self.add(slots)
        for f in flyers:
            self.remove(f)
        self.play(FadeIn(mix_lbl, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: train on the mix - loss on both =====
        MXc = 4.0
        enc = block(["Vision encoder"], 2.3, 0.45, size=14).move_to([MXc - 0.8, -1.9, 0])
        proj = block(["Projection"], 2.3, 0.45, trainable=True, size=14).move_to([MXc - 0.8, -1.15, 0])
        llm_box = RoundedRectangle(corner_radius=0.15, width=4.6, height=0.85, fill_color=C_PANEL, fill_opacity=1,
                                   stroke_color=C_ICE, stroke_width=2).move_to([MXc, -0.15, 0])
        llm_t = VGroup(txt("Large language model", 15, True, C_ICE), txt("frozen weights W", 11, color=GREY_B)
                       ).arrange(DOWN, buff=0.02).move_to(llm_box)
        lora_l = chip(txt("LoRA: BA", 12, True, C_FIRE), C_FIRE, pad=0.07, fill=C_FIRE_FILL)
        lora_r = lora_l.copy()
        lora_l.move_to(llm_box.get_left() + RIGHT * 0.55)
        lora_r.move_to(llm_box.get_right() + LEFT * 0.55)
        ic_enc = icon("snowflake", C_ICE, 0.24).next_to(enc, LEFT, buff=0.1)
        ic_proj = icon("fire", C_FIRE, 0.24).next_to(proj, LEFT, buff=0.1)
        ic_llm = icon("snowflake", C_ICE, 0.24).next_to(llm_box, LEFT, buff=0.1)
        target = chip(VGroup(
            Text("target: knowledge summary (new) or original answer (old)", font=FONT, font_size=13,
                 t2c={"(new)": C_NEWK, "(old)": C_OLD}),
            txt("next-token loss on both", 12, color=GREY_B)).arrange(DOWN, buff=0.03), GREY_C, pad=0.12)
        target.move_to([MXc, 0.95, 0])
        a_e = up_arrow(enc.get_top(), proj.get_bottom())
        a_p = up_arrow(proj.get_top(), [proj.get_x(), llm_box.get_bottom()[1], 0])
        a_t = up_arrow(llm_box.get_top(), target.get_bottom())
        img_in = Arrow([slots.get_right()[0] + 0.05, enc.get_y(), 0], ic_enc.get_left() + LEFT * 0.05, buff=0.05,
                       color=GREY_B, stroke_width=2.5, tip_length=0.14)
        tx_y = -2.45
        txt_in = VGroup(
            Line([slots.get_right()[0] + 0.05, tx_y, 0], [MXc + 1.45, tx_y, 0], color=GREY_B, stroke_width=2.5),
            Arrow([MXc + 1.45, tx_y, 0], [MXc + 1.45, llm_box.get_bottom()[1], 0], buff=0.02, color=GREY_B,
                  stroke_width=2.5, tip_length=0.14),
        )
        in_lbl = txt("image", 11, color=GREY_B).next_to(img_in, UP, buff=0.03)
        tin_lbl = txt("text (question)", 11, color=GREY_B).next_to(txt_in[0], DOWN, buff=0.03)
        variant = txt("Replay-Full-FT: same mix, but all LLM weights train", 12, color=GREY_B)
        variant.move_to([MXc, -2.9, 0])

        self.play(FadeIn(enc), FadeIn(ic_enc), GrowArrow(a_e), FadeIn(proj), FadeIn(ic_proj), run_time=0.7)
        self.play(GrowArrow(a_p), FadeIn(llm_box), FadeIn(llm_t), FadeIn(ic_llm), run_time=0.6)
        self.play(FadeIn(lora_l, scale=1.3), FadeIn(lora_r, scale=1.3))
        self.play(GrowArrow(img_in), Create(txt_in[0]), FadeIn(in_lbl), FadeIn(tin_lbl), run_time=0.5)
        self.play(GrowArrow(txt_in[1]), run_time=0.3)
        self.play(GrowArrow(a_t), FadeIn(target, shift=UP * 0.1))
        # a few training steps: alternate new / old samples, gradient into LoRA + projection
        for k in (3, 0, 5):
            col = C_OLD if order[k] == 1 else C_NEWK
            self.play(Indicate(slots[k], color=WHITE, scale_factor=1.4), run_time=0.35)
            bwd = VMobject().set_points_as_corners([target.get_bottom(), llm_box.get_top(), lora_l.get_center(),
                                                    proj.get_top()])
            self.play(ShowPassingFlash(bwd.set_stroke(col, 7), time_width=0.6),
                      Indicate(VGroup(lora_l, lora_r), color=col, scale_factor=1.1), run_time=0.8)
        self.play(FadeIn(variant))
        self.next_slide()

        # ===== Click: retention result (Table 3) =====
        work = VGroup(new_p, old_p, a_new, a_old, slots, mix_lbl, enc, proj, llm_box, llm_t, lora_l, lora_r, ic_enc,
                      ic_proj, ic_llm, target, a_e, a_p, a_t, img_in, txt_in, in_lbl, tin_lbl, variant)
        self.play(FadeOut(work), run_time=0.6)

        def hbars(title, sub, rows, x0, unit, top_y, fmt_):
            t = VGroup(txt(title, 17, True), txt(sub, 12, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
            t.move_to([0, top_y, 0]).align_to([x0 - 2.2, 0, 0], LEFT)
            items = VGroup()
            y = top_y - 0.65
            for name, v, col in rows:
                nm = txt(name, 14, color=GREY_A).move_to([0, y, 0]).align_to([x0 - 0.12, 0, 0], RIGHT)
                b = RoundedRectangle(corner_radius=0.05, width=max(v * unit, 0.05), height=0.3, fill_color=col,
                                     fill_opacity=1, stroke_width=0).move_to([x0 + v * unit / 2, y, 0])
                val = txt(fmt_(v), 14, True).next_to(b, RIGHT, buff=0.08)
                items.add(VGroup(nm, b, val))
                y -= 0.48
            return t, items

        ret_t, ret = hbars("Retention: average drop on 12 benchmarks", "lower is better (Table 3)",
                           [("Full-FT", 25.89, C_RED), ("Replay-Full-FT", 9.14, C_GOAL),
                            ("LoRA", 25.74, C_RED), ("Replay-LoRA", 6.80, C_GOAL)],
                           -3.6, 0.075, 1.0, lambda v: f"↓{v:.1f}%")
        mia = chip(VGroup(txt("e.g. instruction following (MIA-Bench)", 13, True),
                          Text("LoRA ↓55.3%  →  Replay-LoRA ↓6.0%", font=FONT, font_size=14,
                               t2c={"↓55.3%": C_RED, "↓6.0%": C_GOAL})).arrange(DOWN, buff=0.04),
                   GREY_C, pad=0.12)
        mia.next_to(ret, DOWN, buff=0.25).align_to(ret_t, LEFT)
        rank = txt("Replay-LoRA ranks #1 of all retention methods", 13, True, C_GOAL).next_to(mia, DOWN, buff=0.12,
                                                                                             aligned_edge=LEFT)
        self.play(FadeIn(ret_t))
        for k in range(0, 4, 2):
            self.play(FadeIn(ret[k][0]), GrowFromEdge(ret[k][1], LEFT), FadeIn(ret[k][2]), run_time=0.6)
            self.play(FadeIn(ret[k + 1][0]), GrowFromEdge(ret[k + 1][1], LEFT), FadeIn(ret[k + 1][2]), run_time=0.6)
        self.play(FadeIn(mia, shift=UP * 0.1), FadeIn(rank))
        self.next_slide()

        # ===== Click: the trade-off - adaptation goes down =====
        ad_t, ad = hbars("Adaptation: new-knowledge CEM", "higher is better (Appendix F)",
                         [("Full-FT", 18.02, C_S2), ("Replay-Full-FT", 11.07, GREY_B),
                          ("LoRA", 15.23, C_S2), ("Replay-LoRA", 11.36, GREY_B)],
                         3.7, 0.12, 1.0, lambda v: f"{v:.1f}")
        self.play(FadeIn(ad_t))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in ad],
                              lag_ratio=0.2), run_time=1.4)
        trade = VGroup(
            Text("Trade-off: replay protects the old (②)", font=FONT, font_size=15, weight=BOLD,
                 t2c={"(②)": C_OLD}),
            Text("but the new fact sticks less (①)", font=FONT, font_size=15, weight=BOLD,
                 t2c={"(①)": C_NEWK}),
            txt("CEM drops, F1 roughly flat; more replay data doesn’t fix it (App. F.2)", 12, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        trade.next_to(ad, DOWN, buff=0.25).align_to(ad_t, LEFT)
        self.play(FadeIn(trade, shift=UP * 0.1))
        self.next_slide()

        self.rp = dict(heading=cf["heading"], cites=cites, kicker=bc["kicker"])

    # ================= KORE (Jiang et al., 2026) =================
    def two_cites(self, a, b):
        cs = VGroup(*[VGroup(c, Underline(c, buff=0.04, color=C_LINK, stroke_width=1.2))
                      for c in (txt(a, 15, color=C_LINK), txt(b, 15, color=C_LINK))])
        for c in cs:
            if c.width > 12.8:
                c.scale_to_fit_width(12.8)
        return cs.arrange(DOWN, buff=0.1).move_to([0, -3.45, 0])

    # ---------- Slide 19: KORE stage 1 - knowledge-oriented augmentation ----------
    def kore_augmentation(self):
        rp = self.rp
        C_A, C_R = C_ACCENT, C_PRETRAIN
        head = txt("KORE: Knowledge-Oriented Controls", size=40, bold=True)
        head.move_to(rp["heading"]).align_to(rp["heading"], LEFT)
        cites = self.two_cites(
            "KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls "
            "(Jiang et al., 2026)",
            "Physics of Language Models: Part 3.1, Knowledge Storage and Extraction (Allen-Zhu & Li, 2024)")
        keep = {rp["kicker"], rp["heading"], rp["cites"]}
        others = [m for m in self.mobjects if m not in keep]
        self.play(*[FadeOut(m) for m in others], Transform(rp["heading"], head),
                  ReplacementTransform(rp["cites"], cites), run_time=1.0)

        def stage_chip(n, name, goal, color):
            g = VGroup(txt(f"Stage {n} · {name}", 16, True, color), txt(goal, 12, color=GREY_A)
                       ).arrange(DOWN, buff=0.04)
            return chip(g, color, pad=0.13)

        st1 = stage_chip(1, "KORE-Augmentation", "→ ① adaptation: learn the new accurately", C_A)
        st2 = stage_chip(2, "KORE-Constraint", "→ ② retention: don’t disturb the old", C_R)
        stages = VGroup(st1, st2).arrange(RIGHT, buff=0.5).move_to([0, 1.95, 0])
        self.play(LaggedStart(FadeIn(st1, shift=DOWN * 0.1), FadeIn(st2, shift=DOWN * 0.1), lag_ratio=0.3))
        self.next_slide()

        # ===== Click: why augment? (Physics of LMs 3.1) =====
        self.play(st2.animate.set_opacity(0.35), st1[0].animate.set_stroke(width=3), run_time=0.5)
        why_t = VGroup(txt("Why augment? Physics of LMs 3.1", 17, True, C_A),
                       txt("GPT-2 pretrained on synthetic biographies, then QA-finetuned on half the people;",
                           12, color=GREY_A),
                       txt("test: answer questions about the OTHER half", 12, color=GREY_A)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        why_t.move_to([0, 1.0, 0]).align_to([-6.3, 0, 0], LEFT)

        def bio(lines_, color=GREY_C):
            return chip(lines(lines_, size=12, color=WHITE, buff=0.04), color, pad=0.12)

        bio1 = bio(["Anya Briar Forger was born on October 2, 1996.",
                    "She spent her early years in Princeton, NJ."])
        bio2 = bio(["Born on October 2, 1996, Anya Briar Forger",
                    "grew up in Princeton, NJ."])
        bio3 = bio(["Anya Briar Forger’s childhood was spent in",
                    "Princeton, NJ. She was born on October 2, 1996."])
        VGroup(bio1, bio2, bio3).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([0, -0.85, 0]).align_to(
            [-6.3, 0, 0], LEFT)
        bio_lbl = txt("same facts, different wording & order (illustrative)", 11, color=GREY_B)
        bio_lbl.next_to(bio3, DOWN, buff=0.08, aligned_edge=LEFT)

        BX, UNIT = 1.85, 0.042
        bars_spec = [("one fixed biography", 9.7, C_RED if False else "#E74C3C"),
                     ("5 rephrasings", 41.0, C_S2),
                     ("5 rephrasings + shuffled sentences", 96.6, C_GOAL)]
        bars_t = txt("QA accuracy on the unseen half (%)", 14, True).move_to([0, 0.75, 0]).align_to([BX - 0.9, 0, 0], LEFT)
        bars = []
        y = 0.15
        for name, v, col in bars_spec:
            nm = txt(name, 13, color=GREY_A).move_to([0, y + 0.27, 0]).align_to([BX - 0.9, 0, 0], LEFT)
            b = RoundedRectangle(corner_radius=0.05, width=v * UNIT, height=0.26, fill_color=col, fill_opacity=1,
                                 stroke_width=0).move_to([BX - 0.9 + v * UNIT / 2, y - 0.02, 0])
            val = txt(f"{v:.1f}", 14, True).next_to(b, RIGHT, buff=0.08)
            bars.append(VGroup(nm, b, val))
            y -= 0.82
        lesson = VGroup(txt("Seen in one fixed form → memorized, but not usable.", 13, True),
                        txt("Seen in many forms → the fact can actually be extracted.", 13, True, C_GOAL)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        lesson.move_to([0, -2.75, 0]).align_to([BX - 0.9, 0, 0], LEFT)

        self.play(FadeIn(why_t, shift=DOWN * 0.1))
        self.play(FadeIn(bio1, shift=RIGHT * 0.1), FadeIn(bars_t))
        self.play(FadeIn(bars[0][0]), GrowFromEdge(bars[0][1], LEFT), FadeIn(bars[0][2]))
        self.play(FadeIn(bio2, shift=DOWN * 0.1), FadeIn(bio3, shift=DOWN * 0.1), FadeIn(bio_lbl))
        self.play(FadeIn(bars[1][0]), GrowFromEdge(bars[1][1], LEFT), FadeIn(bars[1][2]))
        self.play(Indicate(bio3, color=C_GOAL, scale_factor=1.04), run_time=0.7)
        self.play(FadeIn(bars[2][0]), GrowFromEdge(bars[2][1], LEFT), FadeIn(bars[2][2]))
        self.play(FadeIn(lesson, shift=UP * 0.1))
        self.next_slide()

        # ===== Click: KORE-Augmentation - one fact becomes a knowledge tree =====
        pol = VGroup(why_t, bio1, bio2, bio3, bio_lbl, bars_t, *bars, lesson)
        self.play(FadeOut(pol), run_time=0.6)
        img = ImageMobject(str(ASSETS / "evo_trump.png")).set_height(0.9)
        know = lines(["During a campaign rally in Butler,", "Pennsylvania on July 13, 2024, a",
                      "gunman attempted to assassinate", "former President Donald Trump…",
                      "the shooter, Thomas Matthew Crooks…"], size=11, color=GREY_A, buff=0.03)
        know[-1].set_color(C_A)
        orig_body = Group(txt("Original knowledge", 13, True), img, know).arrange(DOWN, buff=0.1)
        orig_box = RoundedRectangle(corner_radius=0.15, width=orig_body.width + 0.3, height=orig_body.height + 0.25,
                                    fill_color=C_PANEL, fill_opacity=1, stroke_color=GREY_B, stroke_width=1.5)
        orig = Group(orig_box, orig_body.move_to(orig_box)).move_to([-5.15, -0.5, 0])

        pipe = VGroup(txt("GPT-4o +", 10, color=GREY_B), txt("image search", 10, color=GREY_B))
        pipe.arrange(DOWN, buff=0.02)

        def qa_card(title, q, a_, color, w):
            body = VGroup(txt(title, 12, True, color),
                          Text(q, font=FONT, font_size=11, color=WHITE, t2c={"Q:": C_ACCENT}),
                          Text(a_, font=FONT, font_size=11, color=GREY_A, t2c={"A:": C_GOAL})
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
            box = RoundedRectangle(corner_radius=0.1, width=w, height=body.height + 0.2, fill_color=C_PANEL,
                                   fill_opacity=1, stroke_color=color, stroke_width=1.5)
            body.move_to(box).align_to(box.get_left() + RIGHT * 0.12, LEFT)
            return VGroup(box, body)

        trunk_t = txt("Trunk: multi-round dialogue (up to ~10 rounds)", 13, True, C_A)
        r1 = qa_card("Round 1", "H.Q: Please explain the politics news shown in the image.",
                     "A: …how the assassination attempt on Trump unfolded.", C_A, 4.1)
        r2 = qa_card("Round 2", "Q: Who was identified as the gunman?", "A: Thomas Matthew Crooks.", C_A, 3.0)
        rounds = VGroup(r1, r2, txt("…", 20, True, C_A)).arrange(RIGHT, buff=0.15)
        trunk = VGroup(trunk_t, rounds).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([1.85, 0.75, 0])

        br_t = txt("Branches: instruction tasks", 13, True, C_GOAL)
        b1 = qa_card("Visual recognition", "Q: Is the image depicting the", "assassination of Trump?  A: Yes", C_GOAL, 2.65)
        b2 = qa_card("Image caption", "Q: Caption this politics news.", "A: <knowledge summary>", C_GOAL, 2.45)
        b3 = qa_card("VQA", "Q: Who attempted to assassinate the", "person in the image?  A: T. M. Crooks", C_GOAL, 3.0)
        branches = VGroup(b1, b2, b3).arrange(RIGHT, buff=0.15)
        branch = VGroup(br_t, branches).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.0, -1.3, 0])

        node = Dot([-2.75, -0.5, 0], radius=0.07, color=WHITE)
        a_root = Arrow(orig_box.get_right(), node.get_center(), buff=0.05, color=GREY_B, stroke_width=3,
                       tip_length=0.14)
        pipe.next_to(a_root, UP, buff=0.05)
        to_trunk = VMobject().set_points_as_corners([node.get_center(), [node.get_x(), trunk.get_y(), 0],
                                                     [trunk.get_left()[0] - 0.08, trunk.get_y(), 0]])
        to_branch = VMobject().set_points_as_corners([node.get_center(), [node.get_x(), branch.get_y(), 0],
                                                      [branch.get_left()[0] - 0.08, branch.get_y(), 0]])
        to_trunk.set_stroke(C_A, 3)
        to_branch.set_stroke(C_GOAL, 3)
        stat = txt("→ KORE-74K training set, built from MMEvoke’s knowledge", 13, True, GREY_B)
        stat.move_to([1.85, -2.65, 0])

        self.play(FadeIn(orig, shift=RIGHT * 0.15))
        self.play(GrowArrow(a_root), FadeIn(pipe), GrowFromCenter(node), run_time=0.6)
        self.play(Create(to_trunk), FadeIn(trunk_t), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rounds], lag_ratio=0.25), run_time=1.0)
        self.play(Create(to_branch), FadeIn(br_t), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in branches], lag_ratio=0.25), run_time=1.0)
        self.play(FadeIn(stat))
        self.next_slide()

        # ===== Click: structure > surface variations (Table 5) =====
        tree = Group(orig, a_root, pipe, node, to_trunk, to_branch, trunk, branch, stat)
        self.play(FadeOut(tree), run_time=0.6)
        t5_t = VGroup(txt("Does the structure matter? Knowledge adaptation (K.A), Table 5", 16, True),
                      txt("K.A = mean of CEM and F1 on MMEvoke; all variants trained the same way", 12, color=GREY_B)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        t5_t.move_to([0, 1.0, 0]).align_to([-4.6, 0, 0], LEFT)
        spec = [("KORE-Augmentation (dialogue tree)", 38.82, C_A),
                ("text rephrasing (knowledge-aware)", 20.29, GREY_B),
                ("image edits (knowledge-aware)", 18.33, GREY_B),
                ("synonym swap (knowledge-agnostic)", 15.60, GREY_B)]
        X0, U = -0.6, 0.13
        rows = VGroup()
        y = 0.2
        for name, v, col in spec:
            nm = txt(name, 14, color=WHITE if name.startswith("KORE") else GREY_A).move_to([0, y, 0]).align_to([X0 - 0.15, 0, 0],
                                                                                                 RIGHT)
            b = RoundedRectangle(corner_radius=0.05, width=v * U, height=0.32, fill_color=col, fill_opacity=1,
                                 stroke_width=0).move_to([X0 + v * U / 2, y, 0])
            rows.add(VGroup(nm, b, txt(f"{v:.1f}", 15, True).next_to(b, RIGHT, buff=0.1)))
            y -= 0.55
        note = txt("both text methods use GPT-4o → the gain comes from structure, not from a stronger generator",
                   13, True, C_A).move_to([0, -2.4, 0])
        self.play(FadeIn(t5_t))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2])) for r in rows],
                              lag_ratio=0.2), run_time=1.4)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.next_slide()
        self.kr = dict(heading=rp["heading"], cites=cites, kicker=rp["kicker"], st1=st1, st2=st2, stages=stages,
                       t5=VGroup(t5_t, rows, note))

    # ---------- Slide 20: KORE stage 2 - null-space constrained LoRA ----------
    def kore_constraint(self):
        kr = self.kr
        st1, st2 = kr["st1"], kr["st2"]
        C_A, C_R = C_ACCENT, C_PRETRAIN
        self.play(FadeOut(kr["t5"]), st1.animate.set_opacity(0.35), st2.animate.set_opacity(1), run_time=0.7)
        self.play(st2[0].animate.set_stroke(width=3), st1[0].animate.set_stroke(width=1.5), run_time=0.3)

        # ----- the condition -----
        cond = MathParts([
            ("lora", r"W^*=W_0+BA"), ("sp", r"\qquad"),
            ("keep", r"(W_0+BA)\,x_{\mathrm{old}}\approx W_0\,x_{\mathrm{old}}"), ("sp2", r"\quad\Rightarrow\quad"),
            ("goal", r"A\,x_{\mathrm{old}}\approx 0"),
        ], size=0.36)
        cond["goal"].set_color(C_R)
        cond.move_to([0, 1.05, 0])
        cond_lbl = txt("keep the outputs for old inputs unchanged", 12, color=GREY_B).next_to(cond["keep"], DOWN,
                                                                                              buff=0.08)

        # ----- activation-space picture -----
        P0 = np.array([-3.6, -1.05, 0])
        space = RoundedRectangle(corner_radius=0.2, width=5.6, height=3.0, fill_color="#0E1116", fill_opacity=1,
                                 stroke_color=GREY_D, stroke_width=1.5).move_to(P0)
        sp_t = txt("input activations of one linear layer", 11, color=GREY_B)
        sp_t.move_to(space.get_corner(DL) + [0.15, 0.15, 0], aligned_edge=DL)
        ang = 10 * DEGREES
        u1 = np.array([np.cos(ang), np.sin(ang), 0])
        u2 = np.array([-np.sin(ang), np.cos(ang), 0])
        rng = np.random.default_rng(3)
        pts = [P0 + u1 * rng.normal(0, 0.85) + u2 * rng.normal(0, 0.07) for _ in range(46)]
        olds = VGroup(*[Dot(q, radius=0.045, color=C_R) for q in pts])
        old_lbl = lines(["old-knowledge activations:", "spread along large-σ directions"], size=11, bold=True,
                        color=C_R, buff=0.02)
        old_lbl.move_to(P0 + RIGHT * 1.45 + UP * 0.6)
        ax1 = Arrow(P0 - u1 * 2.3, P0 + u1 * 2.3, buff=0, color=GREY_C, stroke_width=2, tip_length=0.14)
        ax2 = Arrow(P0 - u2 * 1.25, P0 + u2 * 1.25, buff=0, color=C_FIRE, stroke_width=3, tip_length=0.16)
        ax1_l = VGroup()
        ax2_l = lines(["smallest σ ≈ null space"], size=12, bold=True, color=C_FIRE, buff=0.02)
        ax2_l.next_to(ax2.get_end(), LEFT, buff=0.1)

        self.play(FadeIn(cond["lora"]), run_time=0.5)
        self.play(FadeIn(cond["sp"]), FadeIn(cond["keep"]), FadeIn(cond_lbl), run_time=0.6)
        self.play(FadeIn(cond["sp2"]), FadeIn(cond["goal"], scale=1.2), run_time=0.6)
        self.play(FadeIn(space), FadeIn(sp_t))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in olds], lag_ratio=0.02), FadeIn(old_lbl), run_time=1.0)
        self.play(GrowArrow(ax1), FadeIn(ax1_l))
        self.play(GrowArrow(ax2), FadeIn(ax2_l))
        # A reads only along the null direction -> old inputs project to ~0
        proj_lines = VGroup(*[DashedLine(q, P0 + u2 * np.dot(q - P0, u2), dash_length=0.04, color=GREY_B,
                                         stroke_width=1) for q in pts[::3]])
        a_lbl = VGroup(txt("A reads only this direction", 11, color=C_FIRE)).next_to(ax2_l, DOWN, buff=0.06,
                                                                                    aligned_edge=RIGHT)
        self.play(FadeIn(a_lbl))
        self.play(LaggedStart(*[Create(l) for l in proj_lines], lag_ratio=0.05), run_time=0.9)
        zero = txt("A x_old ≈ 0", 13, True, C_R).move_to(P0 + LEFT * 1.35 + DOWN * 0.6)
        self.play(Indicate(Dot(P0, radius=0.08, color=C_R), scale_factor=2.0), FadeIn(zero))
        new_pt = P0 + u1 * 1.2 + u2 * (-0.9)
        new_d = Dot(new_pt, radius=0.08, color=C_A)
        new_proj = DashedLine(new_pt, P0 + u2 * np.dot(new_pt - P0, u2), dash_length=0.05, color=C_A, stroke_width=2)
        new_lbl = lines(["new-knowledge input has a", "null-space part → B learns from it"], size=11,
                        color=C_A, buff=0.02).next_to(new_d, DOWN, buff=0.08)
        new_lbl.align_to([P0[0] + 0.4, 0, 0], LEFT)
        self.play(GrowFromCenter(new_d), Create(new_proj), FadeIn(new_lbl))
        self.next_slide()

        # ===== Click: the recipe =====
        RX = 0.4

        def step(n, parts):
            num = VGroup(Circle(radius=0.15, color=C_R, stroke_width=2), txt(str(n), 13, True, C_R))
            num[1].move_to(num[0])
            body = VGroup(*parts).arrange(RIGHT, buff=0.08)
            return VGroup(num, body).arrange(RIGHT, buff=0.15)

        steps_t = txt("KORE-Constraint recipe (per linear layer)", 13, True, C_R)
        steps = VGroup(
            steps_t,
            step(1, [txt("activations X of 256 OneVision samples;", 12), MathParts([("c", r"C=XX^{\top}")],
                                                                                    size=0.24)]),
            step(2, [txt("SVD of C, keep the r smallest-σ directions", 12)]),
            VGroup(MathParts([("u", r"\hat{U}\approx\mathrm{null}(C),\;\;P=\hat{U}\hat{U}^{\top}")], size=0.24)),
            step(3, [txt("initialize LoRA from the projected weights", 12)]),
            VGroup(MathParts([("w", r"\mathrm{SVD}(W_0P)=U^*\Sigma^*V^{*\top}\;\rightarrow\;"
                                    r"B=U^*\sqrt{\Sigma^*},\;A=\sqrt{\Sigma^*}V^{*\top}")], size=0.24)),
            VGroup(MathParts([("r", r"W_0'=W_0-BA")], size=0.24), txt("(model unchanged at the start)", 11,
                                                                       color=GREY_B)).arrange(RIGHT, buff=0.12),
            step(4, [txt("freeze A", 12, True, C_ICE), icon("snowflake", C_ICE, 0.18), txt(", train only B", 12, True,
                                                                                          C_FIRE),
                     icon("fire", C_FIRE, 0.18)]),
            VGroup(MathParts([("z", r"AC\approx 0\;\Rightarrow\;BAC\approx 0\ \ \mathrm{for\ any}\ B")], size=0.24)),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
        for k in (3, 5, 6, 8):
            steps[k].shift(RIGHT * 0.42)
        steps[5][0].set_color(GREY_A)
        steps[8][0]["z"].set_color(C_R)
        steps.move_to([0, 0, 0]).align_to([RX, 0, 0], LEFT).align_to([0, 0.42, 0], UP)
        steps_box = SurroundingRectangle(steps, buff=0.14, corner_radius=0.15, color=C_R, stroke_width=1.5)
        src = VGroup()
        self.play(Create(steps_box), FadeIn(steps_t))
        for k in range(1, len(steps)):
            self.play(FadeIn(steps[k], shift=RIGHT * 0.1), run_time=0.4)
        self.play(Indicate(steps[8], color=C_R, scale_factor=1.05))
        self.next_slide()
        self.kc = dict(stuff=Group(cond, cond_lbl, space, sp_t, olds, old_lbl, ax1, ax2, ax1_l, ax2_l, proj_lines,
                                   a_lbl, zero, new_d, new_proj, new_lbl, steps, steps_box, steps_t, src))

    # ---------- Slide 21: KORE results (Table 1 & Table 4) ----------
    def kore_results(self):
        kr, kc = self.kr, self.kc
        C_A, C_R, C_RED = C_ACCENT, C_PRETRAIN, "#E74C3C"
        head = txt("KORE: Results", size=40, bold=True)
        head.move_to(kr["heading"]).align_to(kr["heading"], LEFT)
        self.play(FadeOut(kc["stuff"]), FadeOut(kr["stages"]), Transform(kr["heading"], head), run_time=0.8)

        # scatter: x = retention K.R, y = adaptation K.A (LLaVA-v1.5 7B, Table 1)
        XMIN, XMAX, YMIN, YMAX = 28, 50, 0, 42
        L, R_, B, T = -5.3, 1.9, -2.55, 1.85

        def P(x, y):
            return np.array([L + (x - XMIN) / (XMAX - XMIN) * (R_ - L), B + (y - YMIN) / (YMAX - YMIN) * (T - B), 0])

        axes = VGroup(Arrow(P(XMIN, YMIN), P(XMAX, YMIN) + RIGHT * 0.15, buff=0, color=GREY_C, stroke_width=2,
                            tip_length=0.12),
                      Arrow(P(XMIN, YMIN), P(XMIN, YMAX) + UP * 0.15, buff=0, color=GREY_C, stroke_width=2,
                            tip_length=0.12))
        xl = txt("knowledge retention K.R  (12 benchmarks) →", 12, True, C_R).next_to(axes[0], DOWN, buff=0.3)
        yl = txt("knowledge adaptation K.A (MMEvoke) →", 12, True, C_A).rotate(PI / 2).next_to(axes[1], LEFT,
                                                                                                  buff=0.38)
        ticks = VGroup()
        for xv in (30, 35, 40, 45):
            ticks.add(txt(str(xv), 10, color=GREY_B).next_to(P(xv, YMIN), DOWN, buff=0.05))
        for yv in (10, 20, 30, 40):
            ticks.add(txt(str(yv), 10, color=GREY_B).next_to(P(XMIN, yv), LEFT, buff=0.05))
        ideal = txt("better ↗", 12, True, GREY_B).move_to(P(48.3, 40.5))

        base = [("Vanilla LLaVA-v1.5", 46.74, 7.12, UP), ("Full-FT", 31.66, 16.60, LEFT),
                ("LoRA", 33.47, 16.77, DOWN), ("Replay", 43.00, 14.67, UP), ("EWC", 35.14, 17.46, None),
                ("LwF", 35.44, 17.29, None), ("CIA", 35.99, 17.39, None), ("SEFE", 33.02, 20.17, UP),
                ("MoELoRA", 40.51, 9.33, None), ("O-LoRA", 41.25, 9.26, None)]
        dots, labels = VGroup(), VGroup()
        for name, x, y, d in base:
            dd = Dot(P(x, y), radius=0.07, color=GREY_B if name != "Vanilla LLaVA-v1.5" else WHITE)
            dots.add(dd)
            if d is not None:
                labels.add(txt(name, 11, color=GREY_A).next_to(dd, d, buff=0.08))
        labels.add(txt("EWC · LwF · CIA", 11, color=GREY_A).next_to(P(35.5, 17.4), UP, buff=0.12))
        labels.add(txt("MoELoRA · O-LoRA", 11, color=GREY_A).next_to(P(40.9, 9.3), DOWN, buff=0.1))

        self.play(Create(axes), FadeIn(xl), FadeIn(yl), FadeIn(ticks), FadeIn(ideal))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.08), FadeIn(labels), run_time=1.2)
        side_x = 2.6
        base_note = VGroup(txt("Baselines: stuck at K.A ≤ 20", 15, True),
                           txt("they trade one goal for the other", 12, color=GREY_A)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.04).move_to([0, 1.5, 0]).align_to([side_x, 0, 0], LEFT)
        self.play(FadeIn(base_note))
        self.next_slide()

        # ===== Click: KORE =====
        kp = P(40.00, 35.96)
        star = Star(n=5, outer_radius=0.2, inner_radius=0.09, color=C_GOAL, fill_opacity=1).move_to(kp)
        k_lbl = txt("KORE", 14, True, C_GOAL).next_to(star, UP, buff=0.08)
        jump = DashedLine(P(33.47, 16.77), kp, dash_length=0.08, color=C_GOAL, stroke_width=2)
        facts = VGroup(
            txt("KORE (340M trainable params, same as LoRA)", 14, True, C_GOAL),
            Text("MMEvoke CEM 30.7 vs LoRA 15.2  (≈ 2×)", font=FONT, font_size=13, color=WHITE),
            Text("F1 41.3 vs LoRA 18.3", font=FONT, font_size=13, color=WHITE),
            Text("retention K.R 40.0 vs LoRA 33.5", font=FONT, font_size=13, color=WHITE),
            Text("balance score HARS 82.8 vs best baseline 67.6", font=FONT, font_size=13, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        facts.next_to(base_note, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(Create(jump), run_time=0.8)
        self.play(GrowFromCenter(star), FadeIn(k_lbl))
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.1) for f in facts], lag_ratio=0.15), run_time=1.2)
        self.next_slide()

        # ===== Click: ablation - each stage does its job (Table 4) =====
        p_noaug = P(40.16, 14.57)
        p_nocon = P(35.78, 38.82)
        a1 = Arrow(kp, p_noaug, buff=0.15, color=C_A, stroke_width=3, tip_length=0.15)
        a2 = Arrow(kp, p_nocon, buff=0.15, color=C_R, stroke_width=3, tip_length=0.15)
        d1 = Dot(p_noaug, radius=0.08, color=C_A)
        d2 = Dot(p_nocon, radius=0.08, color=C_R)
        l1 = lines(["w/o augmentation:", "K.A 36.0 → 14.6"], size=11, bold=True, color=C_A, buff=0.02)
        l1.next_to(d1, RIGHT, buff=0.1).shift(DOWN * 0.15)
        l2 = lines(["w/o constraint:", "K.R 40.0 → 35.8"], size=11, bold=True, color=C_R, buff=0.02)
        l2.next_to(d2, DOWN, buff=0.1).shift(LEFT * 0.35)
        abl = VGroup(txt("Ablation: each stage protects its own goal", 14, True),
                     Text("augmentation → ① adaptation", font=FONT, font_size=13, t2c={"augmentation": C_A}),
                     Text("constraint → ② retention", font=FONT, font_size=13, t2c={"constraint": C_R}),
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        abl.next_to(facts, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(GrowArrow(a1), GrowFromCenter(d1), FadeIn(l1))
        self.play(GrowArrow(a2), GrowFromCenter(d2), FadeIn(l2))
        self.play(FadeIn(abl, shift=UP * 0.1))
        self.next_slide()

    # ---------- Transition: from training-based injection to Knowledge Editing ----------
    def editing_transition(self):
        HEAD_X = -6.1
        C_RED = "#E74C3C"
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

        kicker = txt("KNOWLEDGE UPDATE", size=18, bold=True, color=C_ACCENT)
        kicker.move_to([0, 3.4, 0]).align_to([HEAD_X, 0, 0], LEFT)
        head = txt("Fine-tuning / LoRA might be expensive", size=40, bold=True)
        head.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)

        # --- the new fact (left) ---
        FX = -5.2
        photo = ImageMobject(str(ASSETS / "person_crop.png")).set_height(1.25).move_to([FX, -0.05, 0])
        frame = SurroundingRectangle(photo, buff=0, color=GREY_C, stroke_width=2)
        years = ["2023", "2024", "2025"]
        answers = ["Former U.S. President", "President-elect", "U.S. President"]
        year = chip(txt(years[0], 15, True, C_ACCENT), C_ACCENT, pad=0.08).next_to(frame, UP, buff=0.12)
        ans = txt(answers[0], 14, True).next_to(frame, DOWN, buff=0.15)
        fact_lbl = txt("new fact", 13, color=GREY_B).next_to(ans, DOWN, buff=0.1)

        # --- augmented training data (stack of cards) ---
        DX = -2.85
        cards = VGroup()
        for i in range(4):
            r = RoundedRectangle(corner_radius=0.08, width=1.25, height=0.8, fill_color=C_PANEL, fill_opacity=1,
                                 stroke_color=C_NEW, stroke_width=1.5)
            ls = VGroup(*[Line(LEFT * 0.4, RIGHT * (0.4 - 0.2 * (k % 2)), color=GREY_B, stroke_width=2)
                          for k in range(3)]).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(r)
            cards.add(VGroup(r, ls).shift(np.array([0.1 * i, -0.1 * i, 0])))
        cards.move_to([DX, -0.05, 0])
        for i, c in enumerate(cards):
            c.set_z_index(4 - i)
        data_lbl = lines(["augmented data", "rephrasings, dialogues, VQA…"], align=ORIGIN, size=13, buff=0.04,
                         color=GREY_A).next_to(cards, DOWN, buff=0.2)
        data_lbl[0].set_color(C_NEW)

        # --- the model as a grid of weights ---
        MX = 0.15
        ROWS, COLS, S = 6, 9, 0.22
        cells = VGroup(*[Square(S, stroke_width=0, fill_color=C_FIRE, fill_opacity=0.35) for _ in range(ROWS * COLS)])
        cells.arrange_in_grid(ROWS, COLS, buff=0.05).move_to([MX, -0.05, 0])
        grid_box = SurroundingRectangle(cells, buff=0.12, corner_radius=0.1, color=C_FIRE, stroke_width=2)
        m_lbl = txt("LMM weights", 15, True, C_FIRE)
        fire = icon("fire", C_FIRE, 0.28)
        m_head = VGroup(fire, m_lbl).arrange(RIGHT, buff=0.1).next_to(grid_box, UP, buff=0.15)
        bar_bg = RoundedRectangle(corner_radius=0.05, width=grid_box.width, height=0.12, fill_color=GREY_E,
                                  fill_opacity=1, stroke_width=0).next_to(grid_box, DOWN, buff=0.3)
        bar_lbl = txt("gradient training on GPUs", 13, color=GREY_A).next_to(bar_bg, DOWN, buff=0.1)
        runs = txt("1", 18, True, C_FIRE)
        runs_lbl = txt("training runs:", 14, True, C_FIRE)
        runs_row = VGroup(runs_lbl, runs).arrange(RIGHT, buff=0.12).next_to(bar_lbl, DOWN, buff=0.18)

        a1 = Arrow(frame.get_right(), cards.get_left(), buff=0.15, color=GREY_B, stroke_width=3, tip_length=0.15)
        a2 = Arrow(cards.get_right(), grid_box.get_left(), buff=0.15, color=GREY_B, stroke_width=3, tip_length=0.15)

        def train_run(rt=1.6):
            fill = Line(bar_bg.get_left(), bar_bg.get_right(), color=C_FIRE, stroke_width=6)
            self.play(Create(fill, rate_func=linear),
                      cells.animate(rate_func=there_and_back, lag_ratio=0.015).set_fill(C_FIRE, 1),
                      run_time=rt)
            return fill

        self.play(FadeIn(kicker), FadeIn(head, shift=UP * 0.1))
        self.play(FadeIn(photo), Create(frame), FadeIn(year), FadeIn(ans), FadeIn(fact_lbl))
        self.play(GrowArrow(a1), LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in cards], lag_ratio=0.15),
                  FadeIn(data_lbl))
        self.play(GrowArrow(a2), FadeIn(cells), Create(grid_box), FadeIn(m_head), FadeIn(bar_bg), FadeIn(bar_lbl))
        fill = train_run()
        self.play(FadeIn(runs_row))
        self.next_slide()

        # ===== Click: the world keeps changing -> do it all again; the costs =====
        PX = 2.45
        def cost_row(ic, color, title, sub):
            t = VGroup(txt(title, 15, True, color), txt(sub, 12, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT,
                                                                                        buff=0.05)
            i = icon(ic, color, 0.36)
            i.move_to(t).align_to([PX, 0, 0], LEFT)
            t.next_to(i, RIGHT, buff=0.22)
            t.shift(LEFT * (t.get_left()[0] - (PX + 0.6)))
            return VGroup(i, t)

        costs = VGroup(
            cost_row("database", C_NEW, "Lots of data per fact", "KORE builds 74K training samples for MMEvoke"),
            cost_row("microchip", C_FIRE, "GPU training", "backprop through a 7B model (LoRA: 340M params)"),
            cost_row("rotate", C_ACCENT, "Again for every change", "the world keeps changing → retrain"),
            cost_row("triangle-exclamation", C_RED, "Risk: forgetting", "old knowledge degrades (as we saw)"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.3, 0]).align_to([PX, 0, 0], LEFT)

        self.play(FadeIn(costs[0], shift=LEFT * 0.1), Indicate(cards, color=C_NEW, scale_factor=1.05))
        self.play(FadeIn(costs[1], shift=LEFT * 0.1))
        for k in (1, 2):
            new_year = chip(txt(years[k], 15, True, C_ACCENT), C_ACCENT, pad=0.08).move_to(year)
            new_ans = txt(answers[k], 14, True).move_to(ans)
            anims = [Transform(year, new_year), Transform(ans, new_ans), FadeOut(fill)]
            if k == 1:
                anims.append(FadeIn(costs[2], shift=LEFT * 0.1))
            self.play(*anims, run_time=0.5)
            self.play(Indicate(cards, color=C_NEW, scale_factor=1.05), run_time=0.4)
            fill = train_run(1.0)
            self.play(Transform(runs, txt(str(k + 1), 18, True, C_FIRE).move_to(runs)), run_time=0.3)
        self.play(FadeIn(costs[3], shift=LEFT * 0.1), cells[::7].animate.set_fill(GREY_D, 0.6))
        self.next_slide()

        # ===== Click: can we do it cheaper - or training-free? =====
        q_head = txt("Can we do it cheaper, or even training-free?", size=40, bold=True)
        q_head.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        q_head[-1].set_color(C_ACCENT)
        skip = VGroup(Cross(VGroup(cards, data_lbl), stroke_color=C_RED, stroke_width=5),
                      Cross(VGroup(bar_bg, bar_lbl), stroke_color=C_RED, stroke_width=5))
        self.play(Transform(head, q_head), FadeOut(costs), FadeOut(runs_row), run_time=0.8)
        self.play(Create(skip), run_time=0.6)
        self.play(FadeOut(VGroup(cards, data_lbl, skip, bar_bg, bar_lbl, fill, a1, a2)), run_time=0.6)

        # the model is frozen, and only the weights holding the fact change
        snow = icon("snowflake", C_ICE, 0.28).move_to(fire)
        e_lbl = txt("LMM weights", 15, True, C_ICE).move_to(m_lbl)
        model = VGroup(cells, grid_box, m_head)
        self.play(cells.animate.set_fill(C_ICE, 0.25), grid_box.animate.set_color(C_ICE),
                  ReplacementTransform(fire, snow), Transform(m_lbl, e_lbl), run_time=0.8)
        fact = Group(photo, frame, year, ans)
        self.play(model.animate.shift(RIGHT * 1.4), fact.animate.shift(RIGHT * 2.0), run_time=0.6)

        old = txt(answers[0], 14, True, GREY_C).move_to(ans)
        strike = Line(old.get_left(), old.get_right(), color=C_RED, stroke_width=3)
        new = txt(answers[2], 14, True, C_ACCENT).next_to(old, DOWN, buff=0.12)
        self.play(FadeOut(ans), FadeOut(fact_lbl), FadeIn(old), Transform(year, chip(
            txt("edit", 15, True, C_ACCENT), C_ACCENT, pad=0.08).move_to(year)))
        self.play(Create(strike), FadeIn(new, shift=DOWN * 0.1))

        target = cells[2 * COLS + 4]
        hit = Square(S, stroke_color=C_ACCENT, stroke_width=3, fill_color=C_ACCENT, fill_opacity=1).move_to(target)
        path = CurvedArrow(frame.get_right() + RIGHT * 0.1 + UP * 0.2, target.get_top() + UP * 0.08, angle=-PI / 3,
                           color=C_ACCENT, stroke_width=3, tip_length=0.15)
        self.play(Create(path), run_time=0.8)
        self.play(FadeIn(hit), Flash(target, color=C_ACCENT, line_length=0.18, flash_radius=0.25), run_time=0.6)
        cap = lines(["locate where the fact is stored", "→ change only those weights"], align=ORIGIN, size=15,
                    bold=True, color=C_ACCENT, buff=0.05).next_to(grid_box, DOWN, buff=0.3)
        self.play(FadeIn(cap, shift=UP * 0.1))

        self.next_slide()

        # ===== Click: pop -> Knowledge Editing =====
        title = txt("Knowledge Editing", size=60, bold=True).move_to(UP * 0.4)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (hit,)],
                  hit.animate.scale(3).set_opacity(0), run_time=0.7)
        self.remove(hit)
        self.play(GrowFromCenter(title), run_time=0.6)
        self.play(title.animate.set_color(C_ACCENT), run_time=0.3)
        self.play(Indicate(title, scale_factor=1.25, color=C_ACCENT), run_time=0.8)

        road = VGroup()
        for name in ("ROME", "MultEdit", "M2Edit"):
            road.add(chip(txt(name, 20, True), GREY_B, pad=0.16))
        road.arrange(RIGHT, buff=0.7).next_to(title, DOWN, buff=0.8)
        arrows = VGroup(*[Arrow(road[i].get_right(), road[i + 1].get_left(), buff=0.1, color=GREY_B,
                                stroke_width=2.5, tip_length=0.12) for i in range(2)])
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in road], lag_ratio=0.3),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3, run_time=1.2))
        self.ed = dict(title=title, road=road, arrows=arrows)
        self.next_slide()

    # ---------- ROME (1/2): locate the fact with causal tracing ----------
    def rome_locate(self):
        HEAD_X = -6.1
        C_H = "#BB8FCE"      # clean hidden state (purple, as in the paper)
        C_COR = "#F0B27A"    # corrupted state
        C_DARK = "#2C2C3A"
        C_FLOW = "#85C1E9"
        ed = self.ed
        kicker = txt("KNOWLEDGE EDITING", size=18, bold=True, color=C_ACCENT)
        kicker.move_to([0, 3.4, 0]).align_to([HEAD_X, 0, 0], LEFT)
        head = txt("ROME: where is the fact stored?", size=40, bold=True)
        head.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        cite = make_cite("Locating and Editing Factual Associations in GPT (Meng et al., 2022)")
        rome_chip = ed["road"][0]
        self.play(FadeOut(ed["title"]), FadeOut(ed["road"][1:]), FadeOut(ed["arrows"]),
                  ReplacementTransform(rome_chip, head), FadeIn(kicker), FadeIn(cite), run_time=0.9)

        toks = ["The", "capital", "of", "Indonesia", "is", "in"]
        SUBJ, LAST = 3, 5
        NR, NC, SP = 6, 8, 0.5
        ROW_Y = [1.2 - i * SP for i in range(NR)]

        def make_grid(x0, labels_right):
            circ = [[Circle(0.12, stroke_color=GREY_C, stroke_width=1.5, fill_color=C_DARK, fill_opacity=1)
                     .move_to([x0 + c * SP, ROW_Y[r], 0]) for c in range(NC)] for r in range(NR)]
            wires = VGroup(*[Line(circ[r][c].get_right(), circ[r][c + 1].get_left(), color=GREY_D, stroke_width=1.5)
                             for r in range(NR) for c in range(NC - 1)])
            labs = VGroup(*[txt(t, 15, True, C_ACCENT if i == SUBJ else GREY_A)
                            .move_to([0, ROW_Y[i], 0]).align_to([labels_right, 0, 0], RIGHT)
                            for i, t in enumerate(toks)])
            cells = VGroup(*[m for row in circ for m in row])
            lay = Arrow([x0, ROW_Y[-1] - 0.42, 0], [x0 + (NC - 1) * SP, ROW_Y[-1] - 0.42, 0], buff=0,
                        color=GREY_C, stroke_width=2, tip_length=0.1)
            lay_l = txt("layers →", 11, color=GREY_B).next_to(lay, DOWN, buff=0.05)
            return circ, cells, VGroup(wires, labs, VGroup(lay, lay_l))

        def bar(x, y, frac, color, label):
            bg = Rectangle(width=1.3, height=0.13, stroke_width=0, fill_color=GREY_E, fill_opacity=1)
            bg.move_to([x, y, 0]).align_to([x, 0, 0], LEFT)
            fg = Rectangle(width=max(1.3 * frac, 0.02), height=0.13, stroke_width=0, fill_color=color, fill_opacity=1)
            fg.align_to(bg, LEFT).align_to(bg, DOWN)
            lb = txt(label, 11, color=GREY_B).next_to(bg, DOWN, buff=0.05).align_to(bg, LEFT)
            return VGroup(bg, fg, lb)

        def out_word(word, color, x):
            return txt(word, 17, True, color).move_to([0, ROW_Y[LAST], 0]).align_to([x, 0, 0], LEFT)

        # ---- (a) clean run ----
        LX0, RX0 = -5.3, 1.3
        lc, lcells, lrest = make_grid(LX0, -5.58)
        l_title = txt("(a) clean run", 17, True, C_H).move_to([LX0 + 1.75, 1.75, 0])
        l_arrow = Arrow(lc[LAST][-1].get_right(), lc[LAST][-1].get_right() + RIGHT * 0.4, buff=0.05,
                        color=GREY_B, stroke_width=2, tip_length=0.1)
        LOX = l_arrow.get_end()[0] + 0.1
        l_out = out_word("Jakarta", C_GOAL, LOX)
        l_bar = bar(LOX, ROW_Y[LAST] - 0.42, 0.9, C_GOAL, "P(Jakarta)")
        cap = txt("Run the fact through the model and record every hidden state (token × layer)", 16, color=GREY_A)
        cap.move_to([0, -2.35, 0])

        self.play(FadeIn(l_title), FadeIn(lcells), FadeIn(lrest), FadeIn(cap))
        cols = [VGroup(*[lc[r][c] for r in range(NR)]) for c in range(NC)]
        self.play(LaggedStart(*[col.animate.set_fill(C_H, 1) for col in cols], lag_ratio=0.25), run_time=1.4)
        self.play(GrowArrow(l_arrow), FadeIn(l_out), FadeIn(l_bar))
        self.next_slide()

        # ---- (b) corrupted subject run ----
        rc, rcells, rrest = make_grid(RX0, 1.03)
        r_title = txt("(b) corrupted subject run", 17, True, C_COR).move_to([RX0 + 1.75, 1.75, 0])
        star_lbl = txt("Indonesia*", 15, True, C_COR).move_to(rrest[1][SUBJ]).align_to(rrest[1][SUBJ], RIGHT)
        noise = chip(txt("+ noise ε", 12, True, C_COR), C_COR, pad=0.06).next_to(star_lbl, LEFT, buff=0.12)
        r_arrow = Arrow(rc[LAST][-1].get_right(), rc[LAST][-1].get_right() + RIGHT * 0.4, buff=0.05,
                        color=GREY_B, stroke_width=2, tip_length=0.1)
        ROX = r_arrow.get_end()[0] + 0.1
        r_out = out_word("?", C_COR, ROX)
        r_bar = bar(ROX, ROW_Y[LAST] - 0.42, 0.06, C_COR, "P(Jakarta)")
        cap2 = txt("Blur the subject's embedding with noise (ν = 3σ) → the model loses the fact", 16, color=GREY_A)
        cap2.move_to(cap)
        self.play(FadeIn(r_title), FadeIn(rcells), FadeIn(rrest), FadeOut(cap))
        self.play(Transform(rrest[1][SUBJ], star_lbl), FadeIn(noise, shift=DOWN * 0.1), FadeIn(cap2))
        rcols = [VGroup(*[rc[r][c] for r in range(NR)]) for c in range(NC)]
        self.play(LaggedStart(*[AnimationGroup(*[rc[r][c].animate.set_fill(C_H if r < SUBJ else C_COR, 1)
                                                 for r in range(NR)]) for c in range(NC)], lag_ratio=0.25),
                  run_time=1.4)
        self.play(GrowArrow(r_arrow), FadeIn(r_out), FadeIn(r_bar))
        self.next_slide()

        # ---- (c) patch ONE clean state into the corrupted run ----
        PR, PC = SUBJ, 3
        src, dst = lc[PR][PC], rc[PR][PC]
        patch = CurvedArrow(src.get_top() + UP * 0.05, dst.get_top() + UP * 0.05, angle=-PI / 2,
                            color=C_H, stroke_width=6, tip_length=0.2)
        p_lbl = txt("(c) patch ONE clean state", 14, True, C_H).next_to(patch, UP, buff=0.08)
        ring = Circle(0.17, color=C_H, stroke_width=3).move_to(src)
        self.play(Create(ring), FadeOut(cap2))
        self.play(Create(patch), FadeIn(p_lbl), run_time=1.0)
        self.play(dst.animate.set_fill(C_H, 1).set_stroke(WHITE, 2.5), ring.animate.move_to(dst), run_time=0.6)
        flow_pts = [dst.get_center(), rc[PR][5].get_center(), rc[LAST][5].get_center(), rc[LAST][-1].get_center(),
                    r_arrow.get_end()]
        flow = VMobject(stroke_color=C_FLOW, stroke_width=10, stroke_opacity=0.45).set_points_as_corners(flow_pts)
        restored = [rc[PR][c] for c in range(PC + 1, NC)] + [rc[LAST][c] for c in range(5, NC)]
        r_out2 = out_word("Jakarta", C_GOAL, ROX)
        r_bar2 = bar(ROX, ROW_Y[LAST] - 0.42, 0.7, C_GOAL, "P(Jakarta)")
        self.play(Create(flow), *[m.animate.set_fill(C_H, 0.75) for m in restored], run_time=1.2)
        self.play(Transform(r_out, r_out2), Transform(r_bar, r_bar2), run_time=0.6)

        ie = MathParts([("ie", r"\mathrm{IE}\;=\;"), ("pc", r"P^{*,\,\mathrm{clean}\ h_i^{(l)}}[o]"),
                        ("m", r"\;-\;"), ("p", r"P^{*}[o]")], size=0.42)
        ie["pc"].set_color(C_H)
        ie["p"].set_color(C_COR)
        ie.move_to([0, -2.3, 0])
        ie_note = Text("indirect effect: how much does restoring this ONE state bring the answer back?",
                       font=FONT, font_size=15, color=GREY_A, t2c={"ONE": C_H}).next_to(ie, DOWN, buff=0.18)
        self.play(Write(ie), run_time=0.8)
        self.play(FadeIn(ie_note))
        self.next_slide()

        # ---- (d) sweep every (token, layer) -> causal-effect map ----
        def heat(color, r, c, table):
            v = table.get(r, [0] * NC)[c]
            return interpolate_color(ManimColor(C_DARK), ManimColor(color), min(1, 0.08 + v))

        STATE = {SUBJ: [0.1, 0.55, 0.9, 0.95, 0.8, 0.45, 0.15, 0.05], 4: [0, 0, 0.05, 0.1, 0.1, 0.1, 0.05, 0],
                 LAST: [0, 0, 0, 0.05, 0.2, 0.55, 0.85, 0.95]}
        MLP = {SUBJ: [0.05, 0.5, 0.9, 0.75, 0.3, 0.05, 0, 0]}
        ATT = {SUBJ: [0, 0.15, 0.05, 0, 0, 0, 0, 0], LAST: [0, 0, 0, 0.05, 0.4, 0.85, 0.7, 0.2]}

        cap3 = txt("Repeat for every token × layer → a map of where the fact is recalled", 16, color=GREY_A)
        cap3.move_to(cap)
        self.play(FadeOut(VGroup(patch, p_lbl, ring, flow, ie, ie_note)), FadeOut(r_out), FadeOut(r_bar),
                  *[m.animate.set_fill(C_DARK, 1).set_stroke(GREY_C, 1.5) for m in rcells], FadeIn(cap3),
                  run_time=0.7)
        scan = SurroundingRectangle(rcols[0], buff=0.08, color=C_FLOW, stroke_width=2.5, corner_radius=0.08)
        self.play(Create(scan), run_time=0.3)
        for c in range(NC):
            self.play(scan.animate.move_to(rcols[c]),
                      *[rc[r][c].animate.set_fill(heat(C_H, r, c, STATE), 1) for r in range(NR)], run_time=0.28)
        self.play(FadeOut(scan), run_time=0.2)

        def site_box(r, c0, c1, color, label, side):
            g = VGroup(*[rc[r][c] for c in range(c0, c1 + 1)])
            box = SurroundingRectangle(g, buff=0.07, color=color, stroke_width=2.5, corner_radius=0.08)
            lb = txt(label, 12, True, color)
            lb.next_to(box, side, buff=0.06)
            bgr = BackgroundRectangle(lb, color=BLACK, fill_opacity=0.85, buff=0.04)
            return VGroup(box, bgr, lb)

        early = site_box(SUBJ, 1, 4, WHITE, "early site", DOWN)
        late = site_box(LAST, 5, 7, WHITE, "late site", DOWN)
        self.play(Create(early[0]), FadeIn(early[1:]), run_time=0.6)
        self.play(Create(late[0]), FadeIn(late[1:]), run_time=0.6)
        self.next_slide()

        # ---- (e,f,g) which component? state vs MLP vs attention ----
        C_ATT = "#EC7063"
        self.play(FadeOut(VGroup(lcells, lrest, l_title, l_arrow, l_out, l_bar, r_arrow, cap3, early, late, noise,
                                 r_title, rrest[0], rrest[2])), run_time=0.6)
        MW, MY = 3.0, 0.4
        CS = MW / NC
        XS = [-4.05, -0.15, 3.75]

        def sq_map(table, color, cx):
            return VGroup(*[Square(CS * 0.86, stroke_width=0, fill_opacity=1, fill_color=heat(color, r, c, table))
                            .move_to([cx - MW / 2 + CS * (c + 0.5), MY + CS * (NR / 2 - r - 0.5), 0])
                            for r in range(NR) for c in range(NC)])

        st_map, mlp_map, att_map = sq_map(STATE, C_H, XS[0]), sq_map(MLP, C_GOAL, XS[1]), sq_map(ATT, C_ATT, XS[2])
        names = ["The", "capital", "of", "Indonesia*", "is", "in"]
        labs = VGroup(*[txt(t, 14, True, C_COR if i == SUBJ else GREY_A)
                        .move_to([0, st_map[i * NC].get_y(), 0]).align_to([XS[0] - MW / 2 - 0.12, 0, 0], RIGHT)
                        for i, t in enumerate(names)])
        self.play(ReplacementTransform(rcells, st_map), ReplacementTransform(rrest[1], labs), run_time=0.9)

        def map_title(s_, color, m):
            return txt(s_, 15, True, color).next_to(m, UP, buff=0.2)

        def lay_lbl(m):
            return txt("layers →", 11, color=GREY_B).next_to(m, DOWN, buff=0.1)

        def mbox(m, r, c0, c1):
            return SurroundingRectangle(VGroup(*[m[r * NC + c] for c in range(c0, c1 + 1)]), buff=0.04,
                                        color=WHITE, stroke_width=2.5)

        t_e = map_title("restore hidden state", C_H, st_map)
        t_f = map_title("restore MLP only", C_GOAL, mlp_map)
        t_g = map_title("restore attention only", C_ATT, att_map)
        st_boxes = VGroup(mbox(st_map, SUBJ, 1, 4), mbox(st_map, LAST, 5, 7))
        self.play(FadeIn(t_e), FadeIn(lay_lbl(st_map)), Create(st_boxes))
        self.play(FadeIn(t_f), FadeIn(lay_lbl(mlp_map)),
                  LaggedStart(*[FadeIn(c) for c in mlp_map], lag_ratio=0.01), run_time=1.0)
        self.play(FadeIn(t_g), FadeIn(lay_lbl(att_map)),
                  LaggedStart(*[FadeIn(c) for c in att_map], lag_ratio=0.01), run_time=1.0)
        e_box, a_box = mbox(mlp_map, SUBJ, 1, 4), mbox(att_map, LAST, 4, 7)
        e_l = txt("early site = MLP", 14, True, C_GOAL).next_to(mlp_map, DOWN, buff=0.45)
        a_l = txt("late site = attention", 14, True, C_ATT).next_to(att_map, DOWN, buff=0.45)
        self.play(Create(e_box), FadeIn(e_l), Create(a_box), FadeIn(a_l))

        take = VGroup(
            Text("Facts are recalled by mid-layer MLPs at the last subject token", font=FONT, font_size=19,
                 weight=BOLD, t2c={"mid-layer MLPs": C_GOAL, "last subject token": C_ACCENT}),
            Text("1000 facts, GPT-2 XL: at that token, MLP effect 6.6% vs attention 1.6%  →  ROME edits one MLP "
                 "(layer 18 of 48)", font=FONT, font_size=14, color=GREY_A),
        ).arrange(DOWN, buff=0.12).move_to([0, -2.25, 0])
        take_box = SurroundingRectangle(take, buff=0.18, corner_radius=0.12, color=C_GOAL, stroke_width=2)
        note = txt("maps illustrative, pattern from the paper's Fig. 1–2", 10, color=GREY_C)
        note.next_to(take_box, DOWN, buff=0.1).align_to(take_box, RIGHT)
        self.play(Create(take_box), FadeIn(take, shift=UP * 0.1), FadeIn(note))
        self.next_slide()
        self.rm = dict(kicker=kicker, head=head, cite=cite, mlp_map=mlp_map, e_box=e_box)

    # ---------- ROME (2/x): shared helpers ----------
    def rome_clear(self, title, keep=()):
        rm = self.rm
        head = txt(title, size=40, bold=True).next_to(rm["kicker"], DOWN, buff=0.15, aligned_edge=LEFT)
        keepset = {rm["kicker"], rm["head"], rm["cite"], *keep}
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keepset], Transform(rm["head"], head),
                  run_time=0.8)

    @staticmethod
    def cellvec(vals, color, cell=0.2, horizontal=False):
        g = VGroup(*[Square(cell, stroke_color=color, stroke_width=1.2, fill_color=color, fill_opacity=v)
                     for v in vals])
        return g.arrange(RIGHT if horizontal else DOWN, buff=0)

    @staticmethod
    def pbar(name, frac, color, x, y, w=1.4):
        """Probability bar; the bar starts at x, the name is right-aligned just left of it."""
        nm = (txt(name, 13, True, color).move_to([x - 0.12, y, 0]).align_to([x - 0.12, 0, 0], RIGHT) if name
              else VMobject().move_to([x, y, 0]))
        bg = Rectangle(width=w, height=0.16, stroke_width=0, fill_color=GREY_E, fill_opacity=1)
        bg.move_to([x + w / 2, y, 0])
        fg = Rectangle(width=max(w * frac, 0.02), height=0.16, stroke_width=0, fill_color=color, fill_opacity=1)
        fg.align_to(bg, LEFT).align_to(bg, DOWN)
        return VGroup(nm, bg, fg)

    # ---------- ROME: the MLP we will rewrite ----------
    def rome_mlp(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("ROME: rewrite one MLP")

        ex = Text("The capital of Indonesia is in", font=FONT, font_size=22, weight=BOLD, t2c={"Indonesia": C_K})
        old = txt("Jakarta", 22, True, GREY_B)
        arr = txt("→", 22, True, GREY_B)
        new = txt("IKN", 22, True, C_V)
        sent = VGroup(ex, old, arr, new).arrange(RIGHT, buff=0.22).move_to([0, 2.1, 0])
        strike = Line(old.get_left(), old.get_right(), color=C_K, stroke_width=4)
        trip = VGroup(txt("subject s", 11, True, C_K).next_to(ex[12:21], DOWN, buff=0.08),
                      txt("target o*", 11, True, C_V).next_to(new, DOWN, buff=0.08))
        self.play(FadeIn(ex), FadeIn(old))
        self.play(Create(strike), FadeIn(arr), FadeIn(new, shift=LEFT * 0.1), FadeIn(trip))

        Y, OX = -0.35, 3.6

        def trap(xl, xr, hl, hr, color, fill_op):
            return Polygon([xl + OX, Y + hl / 2, 0], [xr + OX, Y + hr / 2, 0], [xr + OX, Y - hr / 2, 0],
                           [xl + OX, Y - hl / 2, 0], color=color, stroke_width=2.5, fill_color=color,
                           fill_opacity=fill_op)

        def vec(x, h, color=WHITE):
            x += OX
            return VGroup(Line([x - 0.1, Y + h / 2, 0], [x + 0.1, Y + h / 2, 0], color=color, stroke_width=3),
                          DashedLine([x, Y + h / 2, 0], [x, Y - h / 2, 0], dash_length=0.06, color=color,
                                     stroke_width=2),
                          Line([x - 0.1, Y - h / 2, 0], [x + 0.1, Y - h / 2, 0], color=color, stroke_width=3))

        def arrow(a, b):
            return Arrow([a + OX, Y, 0], [b + OX, Y, 0], buff=0, color=GREY_B, stroke_width=3, tip_length=0.13)

        h_in = vec(-6.2, 0.9, GREY_A)
        wfc = trap(-5.75, -4.45, 0.9, 2.3, GREY_A, 0.08)
        sig = Rectangle(width=0.2, height=2.3, color=GREY_A, stroke_width=2).move_to([-4.2 + OX, Y, 0])
        k_v = vec(-3.55, 2.3, GREY_A)
        wpr = trap(-3.0, -1.7, 2.3, 0.9, C_W, 0.18)
        v_v = vec(-1.05, 0.9, GREY_A)
        arrows = VGroup(arrow(-6.05, -5.8), arrow(-4.08, -3.7), arrow(-3.4, -3.05), arrow(-1.65, -1.2))
        l_fc = MathParts([("w", r"W_{fc}")], size=0.36).move_to(wfc)
        l_sig = MathParts([("s", r"\sigma")], size=0.32).move_to(sig)
        l_pr = MathParts([("w", r"W_{proj}")], size=0.36, color=C_W).move_to(wpr)
        l_k = MathParts([("k", r"k\in\mathbb{R}^{D}")], size=0.28, color=GREY_A).move_to([-3.4 + OX, Y - 1.4, 0])
        l_v = MathParts([("v", r"v\in\mathbb{R}^{H}")], size=0.28, color=GREY_A).move_to([-1.05 + OX, Y - 1.4, 0])
        l_h = lines(["hidden state", "of “Indonesia”"], align=ORIGIN, size=11, buff=0.03, color=GREY_B
                    ).move_to([-6.2 + OX, Y - 1.4, 0])
        mlp = VGroup(h_in, wfc, sig, k_v, wpr, v_v, arrows, l_fc, l_sig, l_pr, l_k, l_v, l_h)
        mlp_t = txt("MLP at layer l*  (found by causal tracing; GPT-2 XL: layer 18)", 15, True, C_GOAL)
        mlp_t.move_to([0, 1.35, 0])
        self.play(FadeIn(mlp_t), LaggedStart(*[FadeIn(m) for m in mlp], lag_ratio=0.08), run_time=1.4)
        self.next_slide()

        # ===== Click: focus on W_proj: key in, value out =====
        dim = VGroup(h_in, wfc, sig, arrows[0], arrows[1], l_fc, l_sig, l_h)
        focus = MathParts([("a", r"\mathrm{focus\ on}\ \ W\equiv W_{proj}^{(l^*)}")], size=0.38, color=C_W)
        focus.next_to(wpr, UP, buff=0.1)
        k_note = lines(["key k", "“which subject is this?”"], align=ORIGIN, size=13, buff=0.04, color=C_K)
        k_note[0].set_color(C_K)
        k_note[1].set_color(GREY_A)
        k_note.next_to(l_k, DOWN, buff=0.15)
        v_note = lines(["value v", "“what do I know about it?”"], align=ORIGIN, size=13, buff=0.04)
        v_note[0].set_color(C_V)
        v_note[1].set_color(GREY_A)
        v_note.next_to(l_v, DOWN, buff=0.15)
        self.play(dim.animate.set_opacity(0.2), FadeOut(mlp_t), k_v.animate.set_color(C_K), v_v.animate.set_color(C_V),
                  l_k.animate.set_color(C_K), l_v.animate.set_color(C_V), wpr.animate.set_fill(C_W, 0.35),
                  run_time=0.8)
        self.play(FadeIn(focus, shift=DOWN * 0.1), FadeIn(k_note), FadeIn(v_note))
        memo = Text("ROME's assumption: W behaves like a key → value memory", font=FONT, font_size=18,
                    weight=BOLD, t2c={"key": C_K, "value": C_V}).move_to([0, -2.75, 0])
        self.play(FadeIn(memo, shift=UP * 0.1))
        self.next_slide()

    # ---------- ROME: what is a (linear) key-value memory? ----------
    def rome_memory(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("Key → value memory: a lookup table?")

        # ---- left: an ordinary lookup table ----
        LX = -3.6
        t_l = txt("① Lookup table", 18, True, GREY_A).move_to([LX, 1.75, 0])
        rows_ = [("Indonesia", "capital: Jakarta"), ("France", "capital: Paris"), ("Japan", "capital: Tokyo")]
        KW, VW, RH = 1.7, 2.3, 0.5
        table = VGroup()
        for i, (k, v) in enumerate(rows_):
            y = 0.85 - i * RH
            kc = Rectangle(width=KW, height=RH, stroke_color=GREY_C, stroke_width=1.5).move_to([LX - VW / 2, y, 0])
            vc = Rectangle(width=VW, height=RH, stroke_color=GREY_C, stroke_width=1.5).move_to([LX + KW / 2, y, 0])
            table.add(VGroup(kc, vc, txt(k, 15, True, C_K).move_to(kc), txt(v, 15, color=C_V).move_to(vc)))
        hdr = VGroup(txt("key", 12, True, GREY_B).next_to(table[0][0], UP, buff=0.08),
                     txt("value", 12, True, GREY_B).next_to(table[0][1], UP, buff=0.08))
        query = chip(txt("query: Indonesia", 14, True, C_K), C_K, pad=0.1).move_to([LX - 1.2, -0.85, 0])
        q_arrow = CurvedArrow(query.get_top() + UP * 0.05, table[0][0].get_left() + LEFT * 0.05, angle=-PI / 2,
                              color=C_K, stroke_width=2.5, tip_length=0.13)
        hit = SurroundingRectangle(table[0], buff=0.04, color=C_ACCENT, stroke_width=3)
        result = chip(txt("capital: Jakarta", 14, True, C_V), C_V, pad=0.1).move_to([LX + 1.3, -0.85, 0])
        r_arrow = CurvedArrow(hit.get_right() + RIGHT * 0.05, result.get_right() + RIGHT * 0.05, angle=-PI / 2,
                              color=C_V, stroke_width=2.5, tip_length=0.13)
        self.play(FadeIn(t_l), FadeIn(hdr), LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.2))
        self.play(FadeIn(query, shift=UP * 0.1), Create(q_arrow))
        self.play(Create(hit), run_time=0.5)
        self.play(Create(r_arrow), FadeIn(result, shift=DOWN * 0.1))
        self.next_slide()

        # ---- right: a linear associative memory ----
        RX = 3.4
        t_r = txt("② Linear associative memory", 18, True, C_W).move_to([RX, 1.75, 0])
        Wm = Square(1.5, stroke_color=C_W, stroke_width=2.5, fill_color=C_W, fill_opacity=0.18).move_to([RX, 0.45, 0])
        grid = VGroup(*[Line(Wm.get_left() + RIGHT * 0.3 * i + UP * 0.75 + RIGHT * 0, Wm.get_left() +
                             RIGHT * 0.3 * i + DOWN * 0.75, color=C_W, stroke_width=0.6, stroke_opacity=0.5)
                        for i in range(1, 5)],
                      *[Line(Wm.get_bottom() + UP * 0.3 * i + LEFT * 0.75, Wm.get_bottom() + UP * 0.3 * i +
                             RIGHT * 0.75, color=C_W, stroke_width=0.6, stroke_opacity=0.5) for i in range(1, 5)])
        w_l = MathParts([("w", "W")], size=0.5, color=C_W).move_to(Wm)
        w_bg = BackgroundRectangle(w_l, color=BLACK, fill_opacity=0.7, buff=0.05)
        KEYS = {"Indonesia": [0.9, 0.2, 0.7, 0.1, 0.5], "France": [0.2, 0.8, 0.3, 0.9, 0.1],
                "Japan": [0.5, 0.1, 0.2, 0.6, 0.95]}
        VALS = {"Indonesia": [0.3, 0.9, 0.5, 0.15, 0.7], "France": [0.85, 0.2, 0.6, 0.4, 0.1],
                "Japan": [0.1, 0.5, 0.95, 0.7, 0.3]}
        VTXT = {"Indonesia": "≈ capital: Jakarta, …", "France": "≈ capital: Paris, …", "Japan": "≈ capital: Tokyo, …"}
        KX, VX = RX - 1.75, RX + 1.75

        def k_vec(n):
            return self.cellvec(KEYS[n], C_K).move_to([KX, 0.45, 0])

        def v_vec(n):
            return self.cellvec(VALS[n], C_V).move_to([VX, 0.45, 0])

        def k_lab(n):
            return txt(f"k({n})", 13, True, C_K).next_to(k_vec(n), DOWN, buff=0.12)

        def v_lab(n):
            return txt(VTXT[n], 12, True, C_V).next_to(v_vec(n), DOWN, buff=0.12)

        kv, vv, kl, vl = k_vec("Indonesia"), v_vec("Indonesia"), k_lab("Indonesia"), v_lab("Indonesia")
        a_in = Arrow([KX + 0.15, 0.45, 0], Wm.get_left(), buff=0.08, color=GREY_B, stroke_width=3, tip_length=0.13)
        a_out = Arrow(Wm.get_right(), [VX - 0.15, 0.45, 0], buff=0.08, color=GREY_B, stroke_width=3, tip_length=0.13)
        rule = MathParts([("v", r"v"), ("e", r"\;=\;"), ("w", r"W"), ("k", r"\,k")], size=0.42)
        rule["v"].set_color(C_V)
        rule["w"].set_color(C_W)
        rule["k"].set_color(C_K)
        rule.move_to([RX, -0.85, 0])
        rule_n = txt("lookup = one matrix multiply", 12, color=GREY_A).next_to(rule, RIGHT, buff=0.2)
        self.play(FadeIn(t_r), FadeIn(Wm), FadeIn(grid), FadeIn(w_bg), FadeIn(w_l))
        self.play(FadeIn(kv, shift=RIGHT * 0.2), FadeIn(kl), GrowArrow(a_in))
        self.play(Indicate(Wm, color=C_W, scale_factor=1.05), GrowArrow(a_out), run_time=0.6)
        self.play(FadeIn(vv, shift=RIGHT * 0.2), FadeIn(vl), FadeIn(rule), FadeIn(rule_n))
        for n in ("France", "Japan"):
            self.play(Transform(kv, k_vec(n)), Transform(kl, k_lab(n)), run_time=0.5)
            self.play(Indicate(Wm, color=C_W, scale_factor=1.05), Transform(vv, v_vec(n)), Transform(vl, v_lab(n)),
                      run_time=0.6)
        same = txt("the SAME W answers every key", 13, True, C_W).next_to(Wm, UP, buff=0.12)
        self.play(FadeIn(same))
        self.next_slide()

        # ---- comparison ----
        def bullets(items, color, x):
            g = VGroup(*[Text("• " + t_, font=FONT, font_size=16, color=GREY_A) for t_ in items])
            g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([x, -1.7, 0])
            return g

        b_l = bullets(["exact match on the key", "one separate slot per fact", "adding a row touches nothing else"],
                      GREY_A, LX)
        b_r = bullets(["keys are vectors; similar keys → similar values",
                       "all facts superimposed in one matrix",
                       "adding a fact can disturb the others!"], C_W, RX)
        b_r[2].set_color(C_K)
        self.play(FadeOut(query), FadeOut(q_arrow), FadeOut(result), FadeOut(r_arrow), FadeOut(rule_n),
                  rule.animate.next_to(Wm, UP, buff=0.12), FadeOut(same))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in b_l], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in b_r], lag_ratio=0.2), run_time=1.2)
        assume = Text("ROME assumes W_proj is a linear associative memory (Kohonen 1972; Anderson 1972)",
                      font=FONT, font_size=15, weight=BOLD, t2c={"linear associative memory": C_W})
        assume.move_to([0, -2.85, 0]).scale(1.1)
        self.play(FadeIn(assume, shift=UP * 0.1))
        self.next_slide()

    # ---------- ROME: many pairs -> least squares ----------
    def rome_lsq(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("Storing many pairs: a least-squares memory")
        Y = 1.3
        n = 6
        W = Rectangle(width=1.6, height=1.0, stroke_color=C_W, stroke_width=2.5, fill_color=C_W, fill_opacity=0.25)
        Kcols = VGroup(*[Rectangle(width=0.36, height=1.6, stroke_color=C_K, stroke_width=1.2, fill_color=C_K,
                                   fill_opacity=0.25 + 0.6 * ((i * 37) % 10) / 10) for i in range(n)]).arrange(RIGHT, buff=0)
        Vcols = VGroup(*[Rectangle(width=0.36, height=1.0, stroke_color=C_V, stroke_width=1.2, fill_color=C_V,
                                   fill_opacity=0.25 + 0.6 * ((i * 53) % 10) / 10) for i in range(n)]).arrange(RIGHT, buff=0)
        times, approx = txt("×", 26, True, GREY_B), MathParts([("a", r"\approx")], size=0.5)
        row = VGroup(W, times, Kcols, approx, Vcols).arrange(RIGHT, buff=0.3).move_to([0, Y, 0])
        k_lbls = VGroup(*[MathParts([("k", rf"k_{{{i + 1}}}" if i < 3 else (r"\cdots" if i == 3 else r"k_n"))],
                                    size=0.26, color=C_K).next_to(Kcols[i], DOWN, buff=0.08)
                          for i in range(n) if i != 4])
        v_lbls = VGroup(*[MathParts([("v", rf"v_{{{i + 1}}}" if i < 3 else (r"\cdots" if i == 3 else r"v_n"))],
                                    size=0.26, color=C_V).next_to(Vcols[i], DOWN, buff=0.08)
                          for i in range(n) if i != 4])
        names = VGroup(
            MathParts([("a", r"W")], size=0.32, color=C_W).next_to(W, DOWN, buff=0.15),
            MathParts([("a", r"K=[\,k_1\,|\,k_2\,|\,\ldots\,]")], size=0.3, color=C_K).next_to(Kcols, DOWN, buff=0.42),
            MathParts([("a", r"V=[\,v_1\,|\,v_2\,|\,\ldots\,]")], size=0.3, color=C_V).next_to(Vcols, DOWN, buff=0.42),
        )
        sub = VGroup(txt("keys (one per fact)", 11, color=GREY_B).next_to(names[1], DOWN, buff=0.06),
                     txt("values", 11, color=GREY_B).next_to(names[2], DOWN, buff=0.06))
        self.play(FadeIn(W), FadeIn(names[0]))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in Kcols], lag_ratio=0.1), FadeIn(times),
                  FadeIn(k_lbls), FadeIn(names[1]), FadeIn(sub[0]))
        self.play(FadeIn(approx), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in Vcols], lag_ratio=0.1),
                  FadeIn(v_lbls), FadeIn(names[2]), FadeIn(sub[1]))
        hk = SurroundingRectangle(Kcols[0], buff=0.03, color=WHITE, stroke_width=2.5)
        hv = SurroundingRectangle(Vcols[0], buff=0.03, color=WHITE, stroke_width=2.5)
        pair = MathParts([("a", r"W k_1\approx v_1")], size=0.3).next_to(row, UP, buff=0.2)
        self.play(Create(hk), Create(hv), FadeIn(pair))
        for i in (1, 2):
            self.play(hk.animate.move_to(Kcols[i]), hv.animate.move_to(Vcols[i]),
                      Transform(pair, MathParts([("a", rf"W k_{i + 1}\approx v_{i + 1}")], size=0.3).move_to(pair)),
                      run_time=0.45)
        self.play(FadeOut(hk), FadeOut(hv), FadeOut(pair))
        self.next_slide()

        # ===== Click: least squares =====
        eq1 = MathParts([("a", r"WK\approx V"), ("b", r"\quad\Rightarrow\quad"),
                         ("c", r"W=\arg\min_{W}\ \Vert WK-V\Vert_F^{2}")], size=0.44)
        eq1["c"].set_color(C_W)
        eq1.move_to([0, -1.0, 0])
        n1 = txt("more facts than W can store exactly → W is the best fit over ALL stored pairs", 14,
                 color=GREY_A).next_to(eq1, DOWN, buff=0.15)
        self.play(Write(eq1), run_time=1.0)
        self.play(FadeIn(n1))
        self.next_slide()

        # ===== Click: closed form / normal equation =====
        eq2 = MathParts([("a", r"W=VK^{+}"), ("b", r"\quad\Leftrightarrow\quad"), ("c", r"W\,"),
                         ("d", r"KK^{\top}"), ("e", r"=VK^{\top}")], size=0.44)
        eq2.move_to([-0.6, -2.35, 0])
        lab_a = txt("pseudoinverse", 11, color=GREY_B).next_to(eq2["a"], DOWN, buff=0.1)
        lab_b = txt("normal equation", 11, color=GREY_B).next_to(VGroup(eq2["c"], eq2["e"]), DOWN, buff=0.1)
        self.play(FadeIn(eq2), FadeIn(lab_a), FadeIn(lab_b))
        cbox = SurroundingRectangle(eq2["d"], buff=0.06, color=C_K, stroke_width=2.5)
        c_note = VGroup(MathParts([("c", r"C=KK^{\top}")], size=0.34, color=C_K),
                        txt("key statistics (paper: 100K", 11, color=GREY_A),
                        txt("Wikipedia tokens) – keep this!", 11, color=GREY_A)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05).next_to(eq2, RIGHT, buff=0.5)
        self.play(Create(cbox), FadeIn(c_note, shift=LEFT * 0.1))
        self.next_slide()

    # ---------- ROME: rank-one update ----------
    def rome_rank1(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("Insert one new fact: the rank-one update")
        prob = MathParts([("o", r"\hat{W}=\arg\min_{\hat{W}}\ \Vert\hat{W}K-V\Vert_F^{2}"),
                          ("s", r"\quad\mathrm{subject\ to}\quad"), ("c", r"\hat{W}k_*=v_*")], size=0.44)
        prob["o"].set_color(C_W)
        prob["c"].set_color(C_V)
        prob.move_to([0, 1.75, 0])
        br_o = Brace(prob["o"], DOWN, buff=0.08, color=C_W)
        br_c = Brace(prob["c"], DOWN, buff=0.08, color=C_V)
        lb_o = txt("disturb old memories as little as possible", 13, True, C_W).next_to(br_o, DOWN, buff=0.06)
        lb_c = txt("store the new pair: k(Indonesia) → IKN", 13, True, C_V).next_to(br_c, DOWN, buff=0.06)
        self.play(Write(prob), run_time=1.0)
        self.play(GrowFromCenter(br_o), FadeIn(lb_o), GrowFromCenter(br_c), FadeIn(lb_c))
        self.next_slide()

        # ===== Click: derivation =====
        EX, LXL = -2.6, -6.4
        steps = [
            ("Lagrange multiplier Λ", [("a", r"\mathcal{J}=\Vert\hat{W}K-V\Vert_F^{2}-2\Lambda^{\top}(\hat{W}k_*-v_*)")]),
            ("set ∂J/∂Ŵ = 0", [("a", r"\hat{W}KK^{\top}-VK^{\top}=\Lambda k_*^{\top}")]),
            ("use C = KKᵀ and VKᵀ = WC", [("a", r"\hat{W}C=WC+\Lambda k_*^{\top}"), ("b", r"\quad\Rightarrow\quad"),
                                          ("c", r"\hat{W}=W+\Lambda\,(C^{-1}k_*)^{\top}")]),
            ("plug into Ŵk* = v*", [("a", r"\Lambda=\frac{v_*-Wk_*}{(C^{-1}k_*)^{\top}k_*}")]),
        ]
        ys = [0.25, -0.45, -1.15, -2.05]
        der = VGroup()
        for (lab, parts), y in zip(steps, ys):
            eq = MathParts(parts, size=0.38)
            eq.move_to([0, y, 0]).align_to([EX, 0, 0], LEFT)
            l_ = txt(lab, 13, True, GREY_B).move_to([0, y, 0]).align_to([LXL, 0, 0], LEFT)
            der.add(VGroup(l_, eq))
        der[2][1]["c"].set_color(C_W)
        der[3][1].set_color(C_V)
        self.play(FadeOut(VGroup(br_o, lb_o, br_c, lb_c)), run_time=0.4)
        for i, d in enumerate(der):
            self.play(FadeIn(d[0]), Write(d[1]), run_time=0.9)
            if i == 1:
                self.next_slide()
        self.next_slide()

        # ===== Click: the result, intuitively =====
        res = MathParts([("w", r"\hat{W}=W+"), ("l", r"\Lambda"), ("c", r"\,(C^{-1}k_*)^{\top}")], size=0.5)
        res["l"].set_color(C_V)
        res["c"].set_color(C_K)
        res.move_to([-1.6, 0.55, 0])
        box = SurroundingRectangle(res, buff=0.2, corner_radius=0.1, color=C_W, stroke_width=2.5)
        self.play(FadeOut(der[:2]), FadeOut(der[2][0]), FadeOut(der[3][0]), FadeOut(der[2][1][:2]),
                  ReplacementTransform(der[2][1]["c"], res), FadeOut(der[3][1]), run_time=0.9)
        self.play(Create(box))
        e_l = VGroup(txt("Λ = how wrong the old W is for k*", 13, True, C_V),
                     MathParts([("a", r"\propto\ v_*-Wk_*\quad(\mathrm{residual})")], size=0.3, color=GREY_A)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([-4.3, -1.0, 0])
        e_r = VGroup(txt("(C⁻¹k*)ᵀ = which inputs trigger the edit", 13, True, C_K),
                     txt("k* “whitened” by key statistics C: responds to k*,", 12, color=GREY_A),
                     txt("disturbs typical old keys as little as possible", 12, color=GREY_A)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([0.6, -1.05, 0])
        a_l = Arrow(res["l"].get_bottom(), e_l.get_top(), buff=0.12, color=C_V, stroke_width=2.5, tip_length=0.13)
        a_r = Arrow(res["c"].get_bottom(), e_r.get_top(), buff=0.12, color=C_K, stroke_width=2.5, tip_length=0.13)
        self.play(GrowArrow(a_l), FadeIn(e_l))
        self.play(GrowArrow(a_r), FadeIn(e_r))

        # rank-one picture
        u = Rectangle(width=0.14, height=1.0, stroke_width=0, fill_color=C_V, fill_opacity=1)
        rrow = Rectangle(width=1.0, height=0.14, stroke_width=0, fill_color=C_K, fill_opacity=1)
        r1 = Square(1.0, stroke_width=1.5, stroke_color=C_W, fill_color=C_W, fill_opacity=0.35)
        outer = VGroup(u, txt("×", 16, True, GREY_B), rrow, txt("=", 18, True), r1).arrange(RIGHT, buff=0.12)
        outer.move_to([4.6, 0.55, 0])
        o_l = txt("rank-one ΔW: one column × one row", 12, True, C_W).next_to(outer, DOWN, buff=0.15)
        self.play(FadeIn(outer), FadeIn(o_l))
        chk = MathParts([("a", r"\mathrm{new\ key:}\ \ \hat{W}k_*=v_*"),
                         ("b", r"\qquad\mathrm{old\ key:}\ \ \hat{W}k_i=Wk_i+\Lambda\,"),
                         ("c", r"(C^{-1}k_*)^{\top}k_i")], size=0.34)
        chk["a"].set_color(C_V)
        chk["c"].set_color(C_K)
        chk.move_to([0, -2.3, 0])
        br = Brace(chk["c"], DOWN, buff=0.06, color=C_K)
        br_l = txt("small on average → old facts kept", 11, True, C_K).next_to(br, DOWN, buff=0.04)
        self.play(FadeIn(chk, shift=UP * 0.1))
        self.play(GrowFromCenter(br), FadeIn(br_l))
        self.next_slide()

    # ---------- ROME: choosing k* ----------
    def rome_key(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("Step 1 · choosing k* (the subject's key)")
        f = MathParts([("k", r"k_*"), ("a", r"=\frac{1}{N}\sum_{j=1}^{N}k(x_j+s),"), ("q", r"\qquad"),
                       ("b", r"k(x)=\sigma\left(W_{fc}^{(l^*)}\,\gamma\left(a^{(l^*)}_{[x],i}+h^{(l^*-1)}_{[x],i}"
                             r"\right)\right)")], size=0.4)
        f["k"].set_color(C_K)
        f.move_to([0, 1.65, 0])
        note = txt("s = subject (Indonesia), i = its last token, l* = edit layer.  "
                   "k* should mean “Indonesia” in ANY context, not one exact prompt", 13, color=GREY_A)
        note.next_to(f, DOWN, buff=0.25)
        self.play(Write(f), run_time=1.2)
        self.play(FadeIn(note))

        prefixes = ["Last week, the minister said", "I", "The report found that", "According to new data,"]
        ys = [0.1, -0.5, -1.1, -1.7]
        rows = VGroup()
        keys = VGroup()
        vals = [[0.9, 0.3, 0.7, 0.2, 0.55, 0.4], [0.8, 0.25, 0.6, 0.35, 0.5, 0.3],
                [0.95, 0.4, 0.65, 0.15, 0.6, 0.5], [0.85, 0.2, 0.75, 0.25, 0.45, 0.35]]
        for j, (pre, y) in enumerate(zip(prefixes, ys)):
            t_ = Text(f"x{j + 1}: “{pre}” + Indonesia", font=FONT, font_size=14, color=GREY_A,
                      t2c={"Indonesia": C_K, f"x{j + 1}:": GREY_B}).move_to([0, y, 0]).align_to([-6.4, 0, 0], LEFT)
            a_ = Arrow([-1.75, y, 0], [-0.75, y, 0], buff=0, color=GREY_B, stroke_width=2.5, tip_length=0.12)
            kvec = self.cellvec(vals[j], C_K, cell=0.22, horizontal=True).move_to([0.2, y, 0])
            rows.add(VGroup(t_, a_))
            keys.add(kvec)
        mlp_lbl = txt("run model, read k at last subject token", 11, color=GREY_B).move_to([-1.25, 0.45, 0])
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0], shift=RIGHT * 0.1), GrowArrow(r[1]), FadeIn(k))
                                for r, k in zip(rows, keys)], lag_ratio=0.3), FadeIn(mlp_lbl), run_time=2.0)
        avg = [sum(v[i] for v in vals) / 4 for i in range(6)]
        kstar = self.cellvec(avg, C_K, cell=0.3, horizontal=True).move_to([4.3, -0.8, 0])
        kbox = SurroundingRectangle(kstar, buff=0.06, color=C_K, stroke_width=3)
        k_l = MathParts([("k", r"k_*")], size=0.4, color=C_K).next_to(kbox, UP, buff=0.15)
        brace = Brace(keys, RIGHT, buff=0.15, color=GREY_B)
        avg_l = txt("average", 13, True, GREY_A).next_to(brace, RIGHT, buff=0.08)
        self.play(GrowFromCenter(brace), FadeIn(avg_l))
        self.play(*[ReplacementTransform(k.copy(), kstar) for k in keys], Create(kbox), FadeIn(k_l), run_time=1.2)
        paper = txt("paper: N = 20 prefixes sampled from the model (ten of length 5, ten of length 10); "
                    "prefixes shown are illustrative", 11, color=GREY_C).move_to([0, -2.6, 0])
        self.play(FadeIn(paper))
        self.next_slide()

    # ---------- ROME: choosing v* ----------
    def rome_value(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("Step 2 · choosing v* (the new fact's value)")
        f = MathParts([("v", r"v_*=\arg\min_{z}\ "),
                       ("a", r"\frac{1}{N}\sum_{j=1}^{N}-\log P_{G(m_i^{(l^*)}:=z)}[\,o^*\,|\,x_j+p\,]"),
                       ("p", r"\ +\ "),
                       ("b", r"D_{KL}\left(P_{G(m_i^{(l^*)}:=z)}[x\,|\,p^{\prime}]\ \Vert\ P_G[x\,|\,p^{\prime}]\right)")],
                      size=0.36)
        f["v"].set_color(C_V)
        f.move_to([0, 1.75, 0])
        br_a = Brace(f["a"], DOWN, buff=0.08, color=C_V)
        br_b = Brace(f["b"], DOWN, buff=0.08, color=C_PRETRAIN)
        lb_a = txt("(a) make it say IKN", 14, True, C_V).next_to(br_a, DOWN, buff=0.06)
        lb_b = txt("(b) don't change what Indonesia is", 14, True, C_PRETRAIN).next_to(br_b, DOWN, buff=0.06)
        z_note = txt("z replaces the MLP output at the subject's last token; no closed form → gradient descent on z",
                     13, color=GREY_A).move_to([0, 0.35, 0])
        self.play(Write(f), run_time=1.2)
        self.play(FadeIn(z_note))
        self.next_slide()

        # ---- (a) panel ----
        AX, BX = -3.4, 3.4
        self.play(GrowFromCenter(br_a), FadeIn(lb_a))
        pa = Text("x_j + “The capital of Indonesia is in”", font=FONT, font_size=15, color=GREY_A,
                  t2c={"Indonesia": C_K}).move_to([AX, -0.3, 0])
        BAR_X = AX - 0.4
        pj = self.pbar("Jakarta", 0.85, GREY_B, BAR_X, -0.9)
        pi_ = self.pbar("IKN", 0.03, C_V, BAR_X, -1.3)
        step = txt("Adam step 0", 13, True, GREY_B).move_to([AX, -1.85, 0])
        self.play(FadeIn(pa), FadeIn(pj), FadeIn(pi_), FadeIn(step))
        for k_, (fj, fi) in enumerate([(0.6, 0.25), (0.25, 0.6), (0.05, 0.92)]):
            self.play(Transform(pj, self.pbar("Jakarta", fj, GREY_B, BAR_X, -0.9)),
                      Transform(pi_, self.pbar("IKN", fi, C_V, BAR_X, -1.3)),
                      Transform(step, txt(f"Adam step {[5, 10, 20][k_]}", 13, True, GREY_B).move_to(step)),
                      run_time=0.5)

        # ---- (b) panel ----
        self.play(GrowFromCenter(br_b), FadeIn(lb_b))
        pb = Text("p′ = “Indonesia is a”", font=FONT, font_size=15, color=GREY_A,
                  t2c={"Indonesia": C_K}).move_to([BX, -0.3, 0])
        BBX = BX - 0.2
        pc = self.pbar("country", 0.8, C_PRETRAIN, BBX, -0.9)
        pt = self.pbar("city", 0.08, GREY_B, BBX, -1.3)
        keep = txt("stays a country, doesn't turn into a city ✓", 13, True, C_PRETRAIN).move_to([BX, -1.85, 0])
        self.play(FadeIn(pb), FadeIn(pc), FadeIn(pt))
        self.play(Wiggle(VGroup(pc, pt), scale_value=1.03, rotation_angle=0.01), FadeIn(keep))
        bottom = txt("only the vector z is optimized (Adam, ≤ 20 steps) · model weights are untouched", 14, True,
                     C_ACCENT).move_to([0, -2.65, 0])
        self.play(FadeIn(bottom, shift=UP * 0.1))
        self.next_slide()

    # ---------- ROME: whole pipeline ----------
    def rome_pipeline(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("ROME: the whole pipeline")

        def stage(content, color, sub):
            box = RoundedRectangle(corner_radius=0.12, width=content.width + 0.4, height=1.0, stroke_color=color,
                                   stroke_width=2.5, fill_color=C_PANEL, fill_opacity=1)
            content.move_to(box)
            s_ = txt(sub, 13, color=GREY_B).next_to(box, DOWN, buff=0.12)
            return VGroup(box, content, s_)

        st = [stage(txt("causal trace", 18, True, C_PRETRAIN), C_PRETRAIN, "pick layer l*"),
              stage(MathParts([("a", r"k_*\ (\mathrm{avg.\ key})")], size=0.4, color=C_K), C_K, "Eq. 3"),
              stage(MathParts([("a", r"v_*\ (\mathrm{optimize}\ z)")], size=0.4, color=C_V), C_V, "Eq. 4"),
              stage(MathParts([("a", r"\hat{W}=W+\Lambda(C^{-1}k_*)^{\top}")], size=0.4, color=C_W), C_W, "Eq. 2")]
        pipe = VGroup(*st).arrange(RIGHT, buff=0.55).move_to([0, 0.7, 0])
        arrs = VGroup(*[Arrow(st[i][0].get_right(), st[i + 1][0].get_left(), buff=0.06, color=GREY_B,
                              stroke_width=3, tip_length=0.13) for i in range(3)])
        self.play(LaggedStart(*[FadeIn(s_, shift=RIGHT * 0.15) for s_ in st], lag_ratio=0.3),
                  LaggedStart(*[GrowArrow(a_) for a_ in arrs], lag_ratio=0.3), run_time=1.8)
        dot = Dot(st[0][0].get_left(), radius=0.09, color=WHITE)
        path = Line(st[0][0].get_left(), st[3][0].get_right())
        self.play(MoveAlongPath(dot, path), run_time=1.5, rate_func=linear)
        self.play(FadeOut(dot), Indicate(st[3][0], color=C_W, scale_factor=1.05))
        self.next_slide()

        # ===== Click: result =====
        q = Text("“The capital of Indonesia is in …”", font=FONT, font_size=20, weight=BOLD,
                 t2c={"Indonesia": C_K})
        before = VGroup(txt("before:", 15, color=GREY_B), txt("Jakarta", 20, True, GREY_B)).arrange(RIGHT, buff=0.15)
        after = VGroup(txt("after ROME:", 15, color=GREY_B), txt("IKN ✓", 20, True, C_V)).arrange(RIGHT, buff=0.15)
        res = VGroup(q, before, after).arrange(RIGHT, buff=0.6).move_to([0, -1.2, 0])
        cost = Text("≈ 2 s per edit on one GPU (GPT-2 XL, A6000): no dataset, no training run", font=FONT,
                    font_size=16, color=GREY_A, t2c={"≈ 2 s per edit": C_GOAL}).move_to([0, -2.05, 0])
        self.play(FadeIn(q))
        self.play(FadeIn(before))
        self.play(FadeIn(after, shift=LEFT * 0.1), Flash(after[1], color=C_V, flash_radius=0.5))
        self.play(FadeIn(cost, shift=UP * 0.1))
        self.next_slide()

    # ---------- Multimodal (1/3): localization in an MLLM ----------
    def mm_locate(self):
        rm = self.rm
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        C_COR, C_RED = "#F0B27A", "#E74C3C"
        kicker = txt("KNOWLEDGE EDITING · MULTIMODAL", size=18, bold=True, color=C_ACCENT)
        kicker.move_to(rm["kicker"]).align_to(rm["kicker"], LEFT)
        head = txt("Multimodal: where is the fact stored?", size=40, bold=True)
        head.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        cite = make_cite("Understanding Information Storage and Transfer in Multi-modal Large Language Models "
                         "(Basu et al., 2024)")
        keep = {rm["kicker"], rm["head"], rm["cite"]}
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], Transform(rm["kicker"], kicker),
                  Transform(rm["head"], head), Transform(rm["cite"], cite), run_time=0.9)

        # ---- LLaVA input: image -> CLIP -> 576 visual tokens + question tokens ----
        flag = id_flag(0.75).move_to([-5.7, 1.5, 0])
        clip = chip(txt("CLIP + projector", 13, True, C_ICE), C_ICE, pad=0.1).move_to([-3.4, 1.5, 0])
        a0 = Arrow(flag.get_right(), clip.get_left(), buff=0.12, color=GREY_B, stroke_width=2.5, tip_length=0.12)
        TY = 0.3
        vis = VGroup(*[Square(0.2, stroke_color=C_ICE, stroke_width=1, fill_color=C_ICE, fill_opacity=0.45)
                       for _ in range(10)]).arrange(RIGHT, buff=0.04)
        vis.move_to([0, TY, 0]).align_to([-6.4, 0, 0], LEFT)
        vis_l = txt("576 visual tokens", 11, color=C_ICE).next_to(vis, DOWN, buff=0.1)
        a1 = Arrow(clip.get_bottom(), vis.get_top() + RIGHT * 0.6, buff=0.1, color=GREY_B, stroke_width=2.5,
                   tip_length=0.12)
        words = ["The", "capital", "of", "this country", "is", "in?"]
        toks = VGroup(*[chip(txt(w, 14, True, C_K if w == "this country" else GREY_A),
                             C_K if w == "this country" else GREY_D, pad=0.07) for w in words])
        toks.arrange(RIGHT, buff=0.08).next_to(vis, RIGHT, buff=0.2)
        CONS = 3
        br = Brace(toks[CONS], DOWN, buff=0.05, color=C_K)
        br_l = txt("visual constraint", 11, True, C_K).next_to(br, DOWN, buff=0.04)
        llm = chip(lines(["LLaVA's LLM", "(Vicuna-7B)"], align=ORIGIN, size=13, bold=True, buff=0.03, color=WHITE),
                   GREY_B, pad=0.12).next_to(toks, RIGHT, buff=0.45)
        a2 = Arrow(toks.get_right(), llm.get_left(), buff=0.08, color=GREY_B, stroke_width=2.5, tip_length=0.12)
        a3 = Arrow(llm.get_right(), llm.get_right() + RIGHT * 0.45, buff=0.08, color=GREY_B, stroke_width=2.5,
                   tip_length=0.12)
        OX = a3.get_end()[0] + 0.1

        def out(word, color):
            return txt(word, 17, True, color).move_to([0, TY + 0.1, 0]).align_to([OX, 0, 0], LEFT)

        def pj(frac, color):
            return self.pbar("", frac, color, OX, TY - 0.25, w=1.0)

        o_word, o_bar = out("Jakarta", C_GOAL), pj(0.85, C_GOAL)
        o_lbl = txt("P(Jakarta)", 10, color=GREY_B).next_to(o_bar[1], DOWN, buff=0.05).align_to(o_bar[1], LEFT)
        self.play(FadeIn(flag), FadeIn(clip), GrowArrow(a0))
        self.play(GrowArrow(a1), LaggedStart(*[FadeIn(v) for v in vis], lag_ratio=0.05), FadeIn(vis_l),
                  LaggedStart(*[FadeIn(t_, shift=UP * 0.1) for t_ in toks], lag_ratio=0.08), run_time=1.2)
        self.play(GrowFromCenter(br), FadeIn(br_l))
        self.play(GrowArrow(a2), FadeIn(llm), GrowArrow(a3), FadeIn(o_word), FadeIn(o_bar), FadeIn(o_lbl))

        # ROME's recipe fails here
        cap = VGroup(Text("ROME's corruption: add Gaussian noise to the subject  →  here: all 576 visual tokens "
                          "+ the constraint", font=FONT, font_size=15, color=GREY_A),
                     Text("too much noise: no layer window restores the answer (even a window of 10 layers)",
                          font=FONT, font_size=15, weight=BOLD, color=C_RED)
                     ).arrange(DOWN, buff=0.1).move_to([0, -1.1, 0])
        noise = chip(txt("+ noise ε", 11, True, C_COR), C_COR, pad=0.06).next_to(vis, UP, buff=0.1)
        self.play(FadeIn(cap[0]), FadeIn(noise), vis.animate.set_fill(C_COR, 0.8).set_stroke(C_COR),
                  toks[CONS][0].animate.set_fill(C_COR, 0.35), run_time=0.8)
        self.play(Transform(o_word, out("?", C_COR)), Transform(o_bar, pj(0.03, C_COR)))
        self.play(FadeIn(cap[1], shift=UP * 0.1))
        self.next_slide()

        # ===== Click: MultimodalCausalTrace =====
        swap = chip(txt("France", 14, True, C_COR), C_COR, pad=0.07).move_to(toks[CONS])
        self.play(FadeOut(cap), FadeOut(noise), vis.animate.set_fill(C_ICE, 0.45).set_stroke(C_ICE),
                  toks[CONS][0].animate.set_fill(C_PANEL, 1), Transform(o_word, out("Jakarta", C_GOAL)),
                  Transform(o_bar, pj(0.85, C_GOAL)), run_time=0.6)
        cap2 = VGroup(
            Text("MultimodalCausalTrace: swap the constraint's tokens for another entity → the image is ignored",
                 font=FONT, font_size=15, weight=BOLD, t2c={"MultimodalCausalTrace:": C_W}),
            Text("(paper's example: “this building” → “Taj Mahal” for a photo of the Space Needle)", font=FONT,
                 font_size=12, color=GREY_B)).arrange(DOWN, buff=0.06).move_to([0, -0.75, 0])
        self.play(ReplacementTransform(toks[CONS], swap), FadeIn(cap2[0]), run_time=0.8)
        self.play(Transform(o_word, out("Paris", C_COR)), Transform(o_bar, pj(0.04, C_COR)), FadeIn(cap2[1]))

        NL, CW = 32, 0.22
        strip = VGroup(*[Square(CW - 0.03, stroke_color=GREY_C, stroke_width=1, fill_color=GREY_E, fill_opacity=1)
                         for _ in range(NL)]).arrange(RIGHT, buff=0.03).move_to([-0.6, -1.75, 0])
        s_lbl = txt("LLM layers", 12, True, GREY_A).next_to(strip, LEFT, buff=0.2)
        l1 = txt("1", 10, color=GREY_B).next_to(strip[0], DOWN, buff=0.06)
        l32 = txt("32", 10, color=GREY_B).next_to(strip[-1], DOWN, buff=0.06)
        s_t = txt("copy a window of layers from the clean run into the corrupted run", 12, color=GREY_A)
        s_t.next_to(strip, UP, buff=0.12)
        rbar_x = strip.get_right()[0] + 0.45
        rbar = self.pbar("", 0.04, C_COR, rbar_x, -1.75, w=1.0)
        r_lbl = txt("P(Jakarta) restored", 10, color=GREY_B).next_to(rbar[1], DOWN, buff=0.05).align_to(rbar[1], LEFT)
        self.play(FadeIn(strip), FadeIn(s_lbl), FadeIn(l1), FadeIn(l32), FadeIn(s_t), FadeIn(rbar), FadeIn(r_lbl))
        IE = [0.75, 0.8, 0.7, 0.55, 0.25, 0.12, 0.08, 0.05, 0.05, 0.04]
        win = SurroundingRectangle(strip[0], buff=0.03, color=WHITE, stroke_width=2.5)
        self.play(Create(win), run_time=0.3)
        for i, v in enumerate(IE):
            col = C_GOAL if v > 0.3 else C_COR
            self.play(win.animate.move_to(strip[i]), Transform(rbar, self.pbar("", v, col, rbar_x, -1.75, w=1.0)),
                      strip[i].animate.set_fill(interpolate_color(ManimColor(GREY_E), ManimColor(C_GOAL), v), 1),
                      run_time=0.22)
        self.play(FadeOut(win), *[strip[i].animate.set_fill(GREY_E, 1) for i in range(10, NL)], run_time=0.3)
        ie = MathParts([("a", r"\mathrm{IE}=P_{\mathrm{restored}}(O)-P_{\mathrm{corr}}(O)")], size=0.32)
        ie.move_to([0, -2.55, 0])
        early = Brace(strip[0:4], UP, buff=0.04, color=C_GOAL)
        self.play(FadeIn(ie), FadeOut(s_t))
        e_l = txt("very early layers", 12, True, C_GOAL).next_to(early, UP, buff=0.04)
        self.play(GrowFromCenter(early), FadeIn(e_l))
        self.next_slide()

        # ===== Click: findings vs LLMs =====
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.6)

        def layer_strip(y, lo, hi, color, label, sub, note):
            st = VGroup(*[Square(0.18, stroke_color=GREY_C, stroke_width=1, fill_color=color if lo <= i + 1 <= hi
                                 else GREY_E, fill_opacity=1) for i in range(NL)]).arrange(RIGHT, buff=0.04)
            st.move_to([0, y, 0]).align_to([-3.3, 0, 0], LEFT)
            lb = VGroup(txt(label, 15, True, color), txt(sub, 11, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT,
                                                                                       buff=0.03)
            lb.move_to([0, y, 0]).align_to([-6.4, 0, 0], LEFT)
            nt = txt(note, 12, True, color).next_to(st, RIGHT, buff=0.2)
            ticks = VGroup(txt(str(lo), 9, color=GREY_B).next_to(st[lo - 1], DOWN, buff=0.04),
                           txt(str(hi), 9, color=GREY_B).next_to(st[hi - 1], DOWN, buff=0.04))
            return VGroup(lb, st, nt, ticks)

        t_st = txt("Which MLP layers are causal? (same language backbone)", 15, True).move_to([0, 2.0, 0])
        llama = layer_strip(1.35, 4, 7, C_PRETRAIN, "LLaMA (Vicuna)-7B", "text-only question", "layers 4–7 · window 5")
        llava = layer_strip(0.6, 1, 4, C_GOAL, "LLaVA-7B", "image + question", "layers 1–4 · window 1")
        self.play(FadeIn(t_st), FadeIn(llama))
        self.play(FadeIn(llava))

        t_tr = txt("How does the image information travel?", 15, True).move_to([0, -0.25, 0])
        steps_ = [(["late visual tokens", "(#540–576 of 576)"], C_ICE), (["1st self-", "attention layer"], GREY_A),
                  (["early MLPs", "(storage)"], C_GOAL), (["mid self-attention", "(layers 16–17)"], GREY_A),
                  (["last token", "→ answer"], C_ACCENT)]
        chain = VGroup(*[chip(lines(t_, align=ORIGIN, size=12, bold=True, buff=0.03, color=c), c, pad=0.1)
                         for t_, c in steps_]).arrange(RIGHT, buff=0.45).move_to([0, -0.95, 0])
        carrs = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.05, color=GREY_B,
                               stroke_width=2.5, tip_length=0.12) for i in range(4)])
        self.play(FadeIn(t_tr), LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in chain], lag_ratio=0.25),
                  LaggedStart(*[GrowArrow(a_) for a_ in carrs], lag_ratio=0.25), run_time=1.6)
        take = VGroup(
            Text("MLLMs retrieve facts from EARLIER layers and need less parametric memory", font=FONT,
                 font_size=17, weight=BOLD, t2c={"EARLIER": C_GOAL}),
            Text("…but move the information to the last token like LLMs do (mid-layer attention)", font=FONT,
                 font_size=14, color=GREY_A)).arrange(DOWN, buff=0.08).move_to([0, -2.25, 0])
        tbox = SurroundingRectangle(take, buff=0.16, corner_radius=0.12, color=C_GOAL, stroke_width=2)
        self.play(Create(tbox), FadeIn(take, shift=UP * 0.1))
        self.next_slide()

    # ---------- Multimodal (2/3): MultEdit on the running example ----------
    def mm_multedit(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("MultEdit: edit LLaVA's early MLP")
        flag = id_flag(0.42)
        q = Text("The capital of this country is in?", font=FONT, font_size=20, weight=BOLD,
                 t2c={"this country": C_K})
        old = txt("Jakarta", 20, True, GREY_B)
        arr = txt("→", 20, True, GREY_B)
        new = txt("IKN", 20, True, C_V)
        ex = VGroup(flag, q, old, arr, new).arrange(RIGHT, buff=0.2).move_to([0, 2.15, 0])
        strike = Line(old.get_left(), old.get_right(), color=C_K, stroke_width=3.5)
        self.play(FadeIn(ex[:3]))
        self.play(Create(strike), FadeIn(arr), FadeIn(new, shift=LEFT * 0.1))

        # LLaVA's input sequence
        vis = VGroup(*[Square(0.18, stroke_color=C_ICE, stroke_width=1, fill_color=C_ICE, fill_opacity=0.45)
                       for _ in range(6)]).arrange(RIGHT, buff=0.04)
        words = ["The", "capital", "of", "this country", "is", "in?"]
        toks = VGroup(*[chip(txt(w, 12, True, C_K if w == "this country" else GREY_A),
                             C_K if w == "this country" else GREY_D, pad=0.06) for w in words])
        toks.arrange(RIGHT, buff=0.06)
        seq = VGroup(vis, toks).arrange(RIGHT, buff=0.15).move_to([0, 1.35, 0]).align_to([-6.4, 0, 0], LEFT)
        vis_l = txt("visual tokens v₁…v_N", 10, color=C_ICE).next_to(vis, DOWN, buff=0.06)
        txt_l = txt("text tokens e₁…e_M", 10, color=GREY_B).next_to(toks, DOWN, buff=0.06)
        last_mark = Triangle(color=C_K, fill_opacity=1, stroke_width=0).scale(0.06).rotate(PI)
        last_mark.next_to(toks[3], UP, buff=0.04).align_to(toks[3], RIGHT).shift(LEFT * 0.08)
        self.play(FadeIn(seq), FadeIn(vis_l), FadeIn(txt_l))

        # MLP at layer 2
        S, CX, CY = 0.8, -3.4, -0.75

        def P(x, y):
            return np.array([CX + x * S, CY + y * S, 0])

        def trap(xl, xr, hl, hr, color, fop):
            return Polygon(P(xl, hl / 2), P(xr, hr / 2), P(xr, -hr / 2), P(xl, -hl / 2), color=color,
                           stroke_width=2.5, fill_color=color, fill_opacity=fop)

        def vec(x, h, color=GREY_A):
            return VGroup(Line(P(x - 0.1, h / 2), P(x + 0.1, h / 2), color=color, stroke_width=3),
                          DashedLine(P(x, h / 2), P(x, -h / 2), dash_length=0.06, color=color, stroke_width=2),
                          Line(P(x - 0.1, -h / 2), P(x + 0.1, -h / 2), color=color, stroke_width=3))

        def arw(a_, b_):
            return Arrow(P(a_, 0), P(b_, 0), buff=0, color=GREY_B, stroke_width=3, tip_length=0.12)

        h_in, wfc = vec(-2.85, 0.9), trap(-2.4, -1.1, 0.9, 2.3, GREY_A, 0.08)
        sig = Rectangle(width=0.2 * S, height=2.3 * S, color=GREY_A, stroke_width=2).move_to(P(-0.85, 0))
        k_v, wpr, z_v = vec(-0.2, 2.3), trap(0.35, 1.65, 2.3, 0.9, C_W, 0.18), vec(2.3, 0.9)
        arrows = VGroup(arw(-2.7, -2.45), arw(-0.73, -0.35), arw(-0.05, 0.3), arw(1.7, 2.15))
        lbls = VGroup(MathParts([("w", r"W_{fc}")], size=0.3).move_to(wfc),
                      MathParts([("w", r"W_{proj}")], size=0.3, color=C_W).move_to(wpr),
                      lines(["hidden state at", "“this country”"], align=ORIGIN, size=10, buff=0.02,
                            color=C_K).next_to(h_in, DOWN, buff=0.12))
        mlp = VGroup(h_in, wfc, sig, k_v, wpr, z_v, arrows, lbls)
        mlp_t = txt("MLP at layer ℓ = 2 (an early causal layer)", 14, True, C_GOAL).next_to(mlp, UP, buff=0.18)
        self.play(FadeIn(mlp_t), LaggedStart(*[FadeIn(m) for m in mlp], lag_ratio=0.08), run_time=1.2)
        self.next_slide()

        # step cards on the right
        PX = 0.9

        def step_card(n, color, title, formula, note):
            badge = VGroup(Circle(0.16, color=color, fill_color=color, fill_opacity=1), txt(str(n), 14, True, BLACK))
            hdr = VGroup(badge, txt(title, 16, True, color)).arrange(RIGHT, buff=0.15)
            body = VGroup(*([formula] if formula is not None else []),
                          *[txt(t_, 12, color=GREY_A) for t_ in note]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            body.next_to(hdr, DOWN, buff=0.12, aligned_edge=LEFT).shift(RIGHT * 0.47)
            return VGroup(hdr, body)

        f1 = MathParts([("k", r"k_{c,\ell}"), ("e", r"=\mathrm{input\ of\ }W^{\ell}_{proj}\mathrm{\ at\ token\ }c")],
                       size=0.3)
        f1["k"].set_color(C_K)
        f2 = MathParts([("z", r"z^*_{c,\ell}"), ("e", r"=\arg\min_{z}\ -\log P(O^*\,|\,v_1\ldots v_N,\ e_1\ldots e_M)")],
                       size=0.28)
        f2["z"].set_color(C_V)
        s1 = step_card(1, C_K, "Key: read k at the constraint's last token", f1,
                       ["c = last token of “this country”", "one forward pass with image + question"])
        s2 = step_card(2, C_V, "Value: optimize z* so LLaVA says “IKN”", f2,
                       ["z replaces W_proj's output at token c", "standard next-token loss · Adam, lr 0.1"])
        s3 = step_card(3, C_W, "Update W_proj: map k → z*, closed form", None,
                       ["change the weights so the old key now returns z*", "(the math on the next slide)"])
        steps = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        steps.align_to([PX, 0, 0], LEFT).align_to([0, 0.9, 0], UP)

        # ---- step 1 ----
        k_box = Rectangle(width=0.32, height=0.42, color=C_K, stroke_width=3).move_to(P(-0.2, 0))
        k_bg = BackgroundRectangle(k_box, color=BLACK, fill_opacity=1, buff=0)
        k_l = MathParts([("k", "k")], size=0.32, color=C_K).move_to(k_box)
        self.play(FadeIn(last_mark), Indicate(toks[3], color=C_K, scale_factor=1.15))
        self.play(FadeIn(s1[0]), FadeIn(k_bg), Create(k_box), FadeIn(k_l), FadeIn(s1[1]))
        self.next_slide()

        # ---- step 2 ----
        z_box = Rectangle(width=0.32, height=0.42, color=C_V, stroke_width=3).move_to(P(2.3, 0))
        z_bg = BackgroundRectangle(z_box, color=BLACK, fill_opacity=1, buff=0)
        z_l = MathParts([("z", "z^*")], size=0.3, color=C_V).move_to(z_box)
        BX = -3.0
        pj = self.pbar("Jakarta", 0.85, GREY_B, BX, -2.45)
        pi_ = self.pbar("IKN", 0.05, C_V, BX, -2.78)
        p_t = txt("LLaVA's answer", 11, color=GREY_B).next_to(pj[1], UP, buff=0.08).align_to(pj[1], LEFT)
        self.play(FadeIn(s2[0]), FadeIn(z_bg), Create(z_box), FadeIn(z_l))
        self.play(FadeIn(s2[1]), FadeIn(pj), FadeIn(pi_), FadeIn(p_t))
        self.play(Transform(pj, self.pbar("Jakarta", 0.05, GREY_B, BX, -2.45)),
                  Transform(pi_, self.pbar("IKN", 0.85, C_V, BX, -2.78)), Indicate(z_box, color=C_V),
                  run_time=1.3)
        self.next_slide()

        # ---- step 3 ----
        self.play(FadeIn(s3[0]), FadeIn(s3[1]))
        self.play(wpr.animate.set_fill(C_W, 0.55), Flash(wpr, color=C_W, flash_radius=0.9, line_length=0.2),
                  run_time=0.8)
        self.next_slide()

    # ---------- Multimodal: MultEdit's closed-form update ----------
    def mm_math(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("MultEdit: the closed-form update")
        obj = MathParts([("w", r"W^{\ell*}_{proj}=\arg\min_{W}\ "), ("a", r"\Vert Wk-z^*\Vert_2^2"), ("p", r"\ +\ "),
                         ("b", r"\lambda\,\Vert W-W^{\prime}\Vert_2^2")], size=0.5)
        obj["a"].set_color(C_V)
        obj["b"].set_color(C_PRETRAIN)
        obj.move_to([0, 1.7, 0])
        br_a = Brace(obj["a"], DOWN, buff=0.08, color=C_V)
        br_b = Brace(obj["b"], DOWN, buff=0.08, color=C_PRETRAIN)
        lb_a = lines(["store the new fact:", "key of “this country” → value for IKN"], align=ORIGIN, size=13,
                     bold=True, buff=0.04, color=C_V).next_to(br_a, DOWN, buff=0.06)
        lb_b = lines(["stay close to the old weights W′", "→ other answers stay the same"], align=ORIGIN, size=13,
                     bold=True, buff=0.04, color=C_PRETRAIN).next_to(br_b, DOWN, buff=0.06)
        self.play(Write(obj), run_time=1.0)
        self.play(GrowFromCenter(br_a), FadeIn(lb_a))
        self.play(GrowFromCenter(br_b), FadeIn(lb_b))
        self.next_slide()

        # ===== Click: derivation =====
        EX, LXL = -2.4, -6.4
        steps = [("set the gradient to 0", r"2\,(Wk-z^*)\,k^{\top}+2\lambda\,(W-W^{\prime})=0"),
                 ("collect the W terms", r"W\,(kk^{\top}+\lambda I)=z^*k^{\top}+\lambda W^{\prime}"),
                 ("solve for W", r"W^{\ell*}_{proj}=(\lambda W^{\prime}+z^*k^{\top})\,(\lambda I+kk^{\top})^{-1}")]
        ys = [0.0, -0.75, -1.55]
        der = VGroup()
        self.play(FadeOut(VGroup(br_a, br_b, lb_a, lb_b)), run_time=0.4)
        for (lab, tex), y in zip(steps, ys):
            eq = MathParts([("a", tex)], size=0.42).move_to([0, y, 0]).align_to([EX, 0, 0], LEFT)
            l_ = txt(lab, 14, True, GREY_B).move_to([0, y, 0]).align_to([LXL, 0, 0], LEFT)
            der.add(VGroup(l_, eq))
            self.play(FadeIn(l_), Write(eq), run_time=0.9)
        der[2][1].set_color(C_W)
        box = SurroundingRectangle(der[2][1], buff=0.15, corner_radius=0.1, color=C_W, stroke_width=2.5)
        eqn = txt("paper Eq. 7", 11, color=GREY_B).next_to(box, RIGHT, buff=0.2).align_to(box, UP)
        self.play(Create(box), FadeIn(eqn))
        self.next_slide()

        # ===== Click: what λ does =====
        def knob(title, color, desc):
            return VGroup(txt(title, 15, True, color), txt(desc, 12, color=GREY_A)).arrange(DOWN, buff=0.05)

        k0 = knob("λ → 0", C_V, "W k = z* exactly: pure edit, may disturb others")
        k1 = knob("λ = 0.01 (paper)", C_W, "found by grid search")
        k2 = knob("λ → ∞", C_PRETRAIN, "W = W′: no edit at all")
        knobs = VGroup(k0, k1, k2).arrange(RIGHT, buff=1.0).move_to([0, -2.55, 0])
        dial = Arrow(knobs.get_left() + LEFT * 0.2 + UP * 0.45, knobs.get_right() + RIGHT * 0.2 + UP * 0.45,
                     buff=0, color=GREY_C, stroke_width=2, tip_length=0.12)
        self.play(GrowArrow(dial), LaggedStart(*[FadeIn(k_, shift=UP * 0.1) for k_ in knobs], lag_ratio=0.3),
                  run_time=1.2)
        cpu = lines(["one matrix inverse,", "no training loop:", "solvable on a CPU"], size=13, bold=True,
                    buff=0.04, color=C_ACCENT).next_to(eqn, DOWN, buff=0.12, aligned_edge=LEFT)
        self.play(FadeIn(cpu))
        self.next_slide()

    # ---------- Multimodal (3/3): how the edit is evaluated ----------
    def mm_eval(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.rome_clear("How is the edit evaluated?")

        def card(title, color, body, x, y, w, h):
            box = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, stroke_width=2,
                                   fill_color=C_PANEL, fill_opacity=1).move_to([x, y, 0])
            t_ = txt(title, 16, True, color).next_to(box.get_top(), DOWN, buff=0.15)
            body.next_to(t_, DOWN, buff=0.15)
            return VGroup(box, t_, body)

        b1 = lines(["~450 Multimodal-Known questions that", "LLaVA-7B answers wrongly",
                    "(well-known places, people, brands)"], align=ORIGIN, size=13, buff=0.05, color=GREY_A)
        lm = VGroup(*[chip(txt(n, 11, True, GREY_A), GREY_C, pad=0.06) for n in
                     ["Inchcolm", "Giannutri", "Croome Court", "Kandy Lake"]]).arrange(RIGHT, buff=0.1)
        b2 = VGroup(lines(["Encyclopedic-VQA: rare landmarks", "“In which country is this island located?”"],
                          align=ORIGIN, size=13, buff=0.05, color=GREY_A), lm).arrange(DOWN, buff=0.1)
        s1 = card("① Fix errors", C_K, b1, -3.4, 1.15, 6.0, 1.55)
        s2 = card("② Insert long-tailed knowledge", C_V, b2, 3.4, 1.15, 6.0, 1.55)
        base = Text("baselines: fine-tuning (all layers) · fine-tuning with a weight constraint (Zhu et al., 2020)",
                    font=FONT, font_size=13, color=GREY_B).move_to([0, 0.1, 0])
        self.play(FadeIn(s1, shift=UP * 0.1))
        self.play(FadeIn(s2, shift=UP * 0.1))
        self.play(FadeIn(base))
        self.next_slide()

        # ===== Click: the three metrics =====
        recall = txt("our “good edit” checklist, made measurable:", 14, True, GREY_A).move_to([0, -0.4, 0])

        def metric(name, alias, color, lines_, measure, x):
            body = VGroup(lines(lines_, align=ORIGIN, size=12, buff=0.04, color=GREY_A),
                          txt(measure, 13, True, color)).arrange(DOWN, buff=0.12)
            c = card(name, color, body, x, -1.75, 4.15, 2.0)
            al = txt(alias, 11, color=GREY_B).next_to(c[1], DOWN, buff=0.03)
            c[2].next_to(al, DOWN, buff=0.12)
            c.add(al)
            return c

        m1 = metric("Editing efficacy", "≈ reliability", C_V, ["same image + same question", "“The capital of this "
                    "country is in?”"], "P(IKN) ↑", -4.35)
        m2 = metric("Generalization", "≈ generality", C_PRETRAIN, ["same image, question paraphrased by an LM",
                    "“Which city is this country's capital?”"], "P(IKN) ↑", 0)
        m3 = metric("Specificity", "≈ locality", C_W, ["unrelated VQA questions", "(OK-VQA, Movies)"],
                    "VQA accuracy should not drop", 4.35)
        self.play(FadeIn(recall))
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in (m1, m2, m3)], lag_ratio=0.3), run_time=1.4)
        note = txt("paraphrase example illustrative", 10, color=GREY_C).next_to(m2, DOWN, buff=0.06)
        self.play(FadeIn(note))
        self.next_slide()

        # ===== Click: results (paper Fig. 6) =====
        self.rome_clear("MultEdit results (LLaVA-7B)")
        METHODS = [("Unedited model", "#7B8A8B"), ("Finetune", "#E59866"), ("Finetune (constrained)", "#5DADE2"),
                   ("MultEdit", C_GOAL)]
        DATA = {"① Fixing error cases": [[0.07, 0.80, 0.51, 0.82], [0.07, 0.63, 0.29, 0.76], [0.40, 0.31, 0.26, 0.38]],
                "② Long-tailed knowledge": [[0.15, 0.79, 0.49, 0.83], [0.14, 0.62, 0.30, 0.79],
                                             [0.40, 0.29, 0.27, 0.39]]}
        GROUPS = ["Editing efficacy", "Generalization", "Specificity"]
        BB, SC, BW = -1.85, 3.9, 0.27

        legend = VGroup(*[VGroup(Square(0.16, stroke_width=0, fill_color=c, fill_opacity=1), txt(n, 13, True, GREY_A))
                          .arrange(RIGHT, buff=0.1) for n, c in METHODS]).arrange(RIGHT, buff=0.45)
        legend.move_to([0, 2.05, 0])
        self.play(FadeIn(legend))
        panels, mult_bars, mult_vals = VGroup(), VGroup(), VGroup()
        for pi_, (title, vals) in enumerate(DATA.items()):
            x0 = -6.2 if pi_ == 0 else 0.6
            W_ = 5.6
            axis = Line([x0, BB, 0], [x0 + W_, BB, 0], color=GREY_C, stroke_width=1.5)
            grid = VGroup(*[DashedLine([x0, BB + g * SC, 0], [x0 + W_, BB + g * SC, 0], color=GREY_D,
                                       stroke_width=1, dash_length=0.05) for g in (0.2, 0.4, 0.6, 0.8)])
            gl = VGroup(*[txt(f"{g:.1f}", 10, color=GREY_B).next_to([x0, BB + g * SC, 0], LEFT, buff=0.06)
                          for g in (0.2, 0.4, 0.6, 0.8)])
            t_ = txt(title, 16, True).move_to([x0 + W_ / 2, BB + 0.92 * SC, 0])
            bars, glabels = VGroup(), VGroup()
            for gi, g in enumerate(GROUPS):
                gc = x0 + W_ * (gi + 0.5) / 3
                for mi, (n, c) in enumerate(METHODS):
                    v = vals[gi][mi]
                    r = Rectangle(width=BW, height=v * SC, stroke_width=0, fill_color=c, fill_opacity=1)
                    r.move_to([gc + (mi - 1.5) * (BW + 0.03), BB, 0], aligned_edge=DOWN)
                    bars.add(r)
                    if mi == 3:
                        mult_bars.add(r)
                        mult_vals.add(txt(f"{v:.2f}", 11, True, C_GOAL).next_to(r, UP, buff=0.05))
                glabels.add(txt(g, 13, True, GREY_A).next_to([gc, BB, 0], DOWN, buff=0.1))
            panels.add(VGroup(axis, grid, gl, t_, bars, glabels))
        for pnl in panels:
            self.play(Create(pnl[0]), FadeIn(pnl[1]), FadeIn(pnl[2]), FadeIn(pnl[3]), FadeIn(pnl[5]), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(b_, DOWN) for b_ in pnl[4]], lag_ratio=0.04), run_time=1.4)
        self.play(FadeIn(mult_vals))
        src = txt("values read from the paper's Fig. 6", 10, color=GREY_C).move_to([5.0, BB - 0.5, 0])
        self.play(FadeIn(src))
        self.next_slide()

        # ===== Click: what to notice =====
        def hl(pnl, gi, color):
            pi_ = list(panels).index(pnl)
            grp = VGroup(*pnl[4][gi * 4:(gi + 1) * 4], mult_vals[pi_ * 3 + gi])
            return SurroundingRectangle(grp, buff=0.06, color=color, stroke_width=2.5, corner_radius=0.06)

        gen = VGroup(hl(panels[0], 1, C_ACCENT), hl(panels[1], 1, C_ACCENT))
        spe = VGroup(hl(panels[0], 2, C_K), hl(panels[1], 2, C_K))
        notes = VGroup(
            Text("Generalization: fine-tuning drops on paraphrased questions (0.80 → 0.63) – MultEdit holds up (0.76)",
                 font=FONT, font_size=14, color=C_ACCENT),
            Text("Specificity: fine-tuning breaks unrelated answers (0.40 → 0.31) – MultEdit barely moves (0.38)",
                 font=FONT, font_size=14, color=C_K),
        ).arrange(DOWN, buff=0.08).move_to([0, BB - 0.95, 0]).shift(DOWN * 0.05)
        self.play(Create(gen), FadeIn(notes[0]))
        self.play(Create(spe), FadeIn(notes[1]))
        self.next_slide()

    # ---------- M2Edit (Zhou et al., EMNLP 2025) ----------
    def m2_start(self, title):
        rm = self.rm
        head = txt(title, size=40, bold=True).next_to(rm["kicker"], DOWN, buff=0.15, aligned_edge=LEFT)
        cite = make_cite("M2Edit: Locate and Edit Multi-Granularity Knowledge in Multimodal Large Language Model "
                         "(Zhou et al., EMNLP 2025)")
        keep = {rm["kicker"], rm["head"], rm["cite"]}
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], Transform(rm["head"], head),
                  Transform(rm["cite"], cite), run_time=0.8)

    def m2_intro(self):
        self.m2_start("M2Edit: knowledge comes in granularities")

        def card(title, color, triple, q, x):
            box = RoundedRectangle(corner_radius=0.12, width=4.1, height=2.3, stroke_color=color, stroke_width=2,
                                   fill_color=C_PANEL, fill_opacity=1).move_to([x, 0.55, 0])
            t_ = txt(title, 20, True, color).next_to(box.get_top(), DOWN, buff=0.18)
            tr = txt(triple, 14, True).next_to(t_, DOWN, buff=0.2)
            qq = lines(q, align=ORIGIN, size=12, buff=0.04, color=GREY_A).next_to(tr, DOWN, buff=0.2)
            return VGroup(box, t_, tr, qq)

        cards = VGroup(
            card("Entity", C_NEW, "(image, is a, capybara)", ["“What animal is presented", "in the image?”"], -4.4),
            card("Relation", C_ACCENT, "(image, profession, scientist)",
                 ["“What profession does the person", "in the image have?”"], 0),
            card("Action", C_PRETRAIN, "(image, action, running)",
                 ["“What action is the [agent] undertaking", "at [place] that involves moving quickly?”"], 4.4))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards], lag_ratio=0.3), run_time=1.4)
        stats = txt("M2Edit dataset: 35,673 entity · 2,167 relation · 4,557 action samples (examples from the paper)",
                    13, color=GREY_B).move_to([0, -1.05, 0])
        ours = Text("our running example  (Indonesia, capital, Jakarta → IKN)  is relation-level knowledge",
                    font=FONT, font_size=17, weight=BOLD, t2c={"relation-level": C_ACCENT}).move_to([0, -1.9, 0])
        flag = id_flag(0.4).next_to(ours, LEFT, buff=0.25)
        self.play(FadeIn(stats))
        self.play(FadeIn(flag), FadeIn(ours, shift=UP * 0.1), Indicate(cards[1][0], color=C_ACCENT, scale_factor=1.03))
        self.next_slide()

    def m2_metrics(self):
        C_K, C_V = "#EC7063", C_ACCENT
        self.m2_start("Evaluating an edit: four metrics")
        # the edit sample
        xi = chip(Text("x_i: “The capital of this country is in?”", font=FONT, font_size=14, color=WHITE,
                       t2c={"x_i:": GREY_B}), GREY_C, pad=0.08)
        vi = VGroup(txt("v_i:", 14, True, GREY_B), id_flag(0.32)).arrange(RIGHT, buff=0.1)
        yi = chip(Text("y_i: IKN", font=FONT, font_size=14, weight=BOLD, color=C_V, t2c={"y_i:": GREY_B}), C_V,
                  pad=0.08)
        samp = VGroup(txt("one edit sample:", 14, True), xi, vi, yi).arrange(RIGHT, buff=0.25).move_to([0, 2.1, 0])
        nota = txt("I(·) = 1 if the statement is true, else 0   ·   E[·] = average over all edit samples → a score in %",
                   12, color=GREY_B).move_to([0, 1.62, 0])
        ex_note = txt("example images / paraphrase illustrative", 10, color=GREY_C).move_to([5.6, -2.85, 0])
        self.play(FadeIn(samp))
        self.play(FadeIn(nota), FadeIn(ex_note))

        NX, FX, EXX = -6.4, -4.2, 1.45
        ys = [0.85, -0.2, -1.25, -2.3]

        def row(y, name, sub, color, parts, hi, meaning, img, q, ans, extra=None):
            nm = VGroup(txt(name, 15, True, color), txt(sub, 11, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT,
                                                                                       buff=0.04)
            nm.move_to([0, y, 0]).align_to([NX, 0, 0], LEFT)
            f = MathParts(parts, size=0.3)
            for h in hi:
                f[h].set_color(color)
            f.move_to([0, y + 0.13, 0]).align_to([FX, 0, 0], LEFT)
            mn = Text(meaning, font=FONT, font_size=12, color=GREY_A).next_to(f, DOWN, buff=0.1, aligned_edge=LEFT)
            img.move_to([EXX + 0.35, y, 0])
            qq = Text(q, font=FONT, font_size=12, color=WHITE).next_to(img, RIGHT, buff=0.15)
            a_ = VGroup(txt("→", 13, True, GREY_B), chip(txt(ans, 12, True, color), color, pad=0.05),
                        txt("✓", 15, True, C_GOAL)).arrange(RIGHT, buff=0.1).next_to(qq, RIGHT, buff=0.12)
            g = VGroup(nm, f, mn, img, qq, a_)
            if extra is not None:
                extra.next_to(a_, DOWN, buff=0.06).align_to(qq, LEFT)
                g.add(extra)
            return g

        def framed(m, h=0.42):
            m.set_height(h)
            return VGroup(m, SurroundingRectangle(m, buff=0.04, color=GREY_C, stroke_width=1))

        def flag_pole():
            f = id_flag(0.3)
            pole = Line(f.get_corner(UL) + UP * 0.08, f.get_corner(DL) + DOWN * 0.18, color=GREY_B, stroke_width=3)
            pole.shift(LEFT * 0.03)
            g = VGroup(pole, f.apply_function(lambda p: p + np.array([0, 0.05 * np.sin(4 * (p[0] - f.get_left()[0])), 0])))
            return g

        dog = icon("dog", GREY_A, 0.32)
        rows = [
            row(ys[0], "Reliability", "the edit itself", C_V,
                [("a", r"O_{rel}(\hat{\Theta})=\mathbb{E}\,[\,\mathbb{I}(\hat{\Theta}(x_i,v_i)=y_i)\,]")], [],
                "same image, same question: does the edited model output y_i?",
                framed(id_flag(0.32)), "“The capital of this country is in?”", "IKN"),
            row(ys[1], "Visual generality", "other images", C_PRETRAIN,
                [("a", r"O^{v}_{gen}(\hat{\Theta})=\mathbb{E}\,[\,\mathbb{I}(\hat{\Theta}(x_i,"), ("b", r"v_j"),
                 ("c", r")=y_i)\,]\quad"), ("d", r"v_j\sim v_i")], ["b", "d"],
                "swap in a SIMILAR image v_j (another photo of the same thing)",
                framed(flag_pole()), "“The capital of this country is in?”", "IKN"),
            row(ys[2], "Text generality", "other wordings", C_NEW,
                [("a", r"O^{t}_{gen}(\hat{\Theta})=\mathbb{E}\,[\,\mathbb{I}(\hat{\Theta}("), ("b", r"x_j"),
                 ("c", r",v_i)=y_i)\,]\quad"), ("d", r"x_j\sim x_i")], ["b", "d"],
                "swap in a PARAPHRASED question x_j, same image",
                framed(id_flag(0.32)), "“Which city is this country's capital?”", "IKN"),
            row(ys[3], "Locality", "everything else", C_GOAL,
                [("a", r"O_{loc}(\hat{\Theta})=\mathbb{E}\,[\,\mathbb{I}(\hat{\Theta}(x_k,v_k)="),
                 ("b", r"\Theta(x_k,v_k)"), ("c", r")\,]")], ["b"],
                "(x_k, v_k) ⊥ (x_i, v_i): an UNRELATED sample keeps the ORIGINAL answer",
                framed(dog), "“What animal is this?”", "dog",
                extra=txt("same as before the edit", 10, color=GREY_B)),
        ]
        for r_ in rows:
            self.play(FadeIn(r_[0]), Write(r_[1]), run_time=0.8)
            self.play(FadeIn(r_[2]), FadeIn(r_[3:], shift=LEFT * 0.1))
            self.next_slide()

    def m2_arch(self, Y, answer="Jakarta"):
        """BLIP2-style MLLM: visual encoder -> multimodal interface -> LLM, each with a strip of layers."""
        o = {}
        o["flag"] = id_flag(0.6).move_to([-6.3, Y, 0])

        def comp(x, w, n, label, color):
            box = RoundedRectangle(corner_radius=0.12, width=w, height=1.0, stroke_color=color, stroke_width=2,
                                   fill_color=C_PANEL, fill_opacity=1).move_to([x, Y, 0])
            cells = VGroup(*[Square(0.2, stroke_color=GREY_C, stroke_width=1, fill_color=GREY_E, fill_opacity=1)
                             for _ in range(n)]).arrange(RIGHT, buff=0.05).move_to(box)
            lb = lines(label, align=ORIGIN, size=12, bold=True, buff=0.02, color=color).next_to(box, UP, buff=0.08)
            return VGroup(box, cells, lb)

        o["ve"] = comp(-4.35, 2.3, 8, ["visual encoder", "(ViT)"], C_ICE)
        o["mi"] = comp(-1.75, 1.9, 6, ["multimodal interface", "(Q-Former)"], C_S2)
        o["llm"] = comp(1.35, 3.5, 12, ["LLM", "(OPT-6.7B)"], C_PRETRAIN)
        ar = lambda a_, b_: Arrow(a_, b_, buff=0.1, color=GREY_B, stroke_width=2.5, tip_length=0.12)
        o["arrows"] = VGroup(ar(o["flag"].get_right(), o["ve"][0].get_left()),
                             ar(o["ve"][0].get_right(), o["mi"][0].get_left()),
                             ar(o["mi"][0].get_right(), o["llm"][0].get_left()),
                             ar(o["llm"][0].get_right(), o["llm"][0].get_right() + RIGHT * 0.5))
        o["prompt"] = chip(Text("“The capital of this country is in?”", font=FONT, font_size=12, color=WHITE,
                                t2c={"this country": "#EC7063"}), GREY_C, pad=0.06).move_to([1.35, Y - 1.0, 0])
        o["p_arrow"] = Arrow(o["prompt"].get_top(), o["llm"][0].get_bottom(), buff=0.05, color=GREY_B,
                             stroke_width=2, tip_length=0.1)
        o["OX"] = o["arrows"][3].get_end()[0] + 0.1
        o["Y"] = Y
        o["out"] = txt(answer, 17, True, C_GOAL).move_to([0, Y + 0.12, 0]).align_to([o["OX"], 0, 0], LEFT)
        o["all"] = VGroup(o["flag"], o["ve"], o["mi"], o["llm"], o["arrows"], o["prompt"], o["p_arrow"])
        return o

    def m2_trace(self):
        C_COR = "#F0B27A"
        self.m2_start("M2Edit ① where is each kind of knowledge?")
        a = self.m2_arch(1.0)
        Y, OX = a["Y"], a["OX"]

        def pbar(frac, color):
            return self.pbar("", frac, color, OX, Y - 0.25, w=1.1)

        bar = pbar(0.85, C_GOAL)
        b_l = txt("P(Jakarta)", 10, color=GREY_B).next_to(bar[1], DOWN, buff=0.05).align_to(bar[1], LEFT)
        self.play(FadeIn(a["all"]), run_time=1.0)
        self.play(FadeIn(a["out"]), FadeIn(bar), FadeIn(b_l))
        rng = np.random.RandomState(4)
        fl = a["flag"]
        speck = VGroup(*[Dot([fl.get_left()[0] + rng.rand() * fl.width, fl.get_bottom()[1] + rng.rand() * fl.height, 0],
                             radius=0.025, color=C_COR) for _ in range(60)])
        cap = Text("corrupted run: add Gaussian noise to the IMAGE (v_i + ε), not to the text", font=FONT,
                   font_size=15, color=GREY_A, t2c={"IMAGE": C_COR}).move_to([0, -0.75, 0])
        out_q = txt("?", 17, True, C_COR).move_to(a["out"]).align_to(a["out"], LEFT)
        self.play(FadeIn(speck, lag_ratio=0.02), FadeIn(cap), run_time=0.8)
        self.play(Transform(a["out"], out_q), Transform(bar, pbar(0.04, C_COR)))
        self.next_slide()

        # ===== Click: restore states in all three components =====
        cap2 = Text("restore one clean state at a time, in every component → which ones bring Jakarta back?",
                    font=FONT, font_size=15, color=GREY_A).move_to(cap)
        self.play(Transform(cap, cap2))
        IE = {"ve": [0.05, 0.1, 0.15, 0.3, 0.45, 0.35, 0.2, 0.1], "mi": [0.15, 0.4, 0.55, 0.3, 0.2, 0.1],
              "llm": [0.2, 0.45, 0.85, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]}
        scan = Square(0.3, color=WHITE, stroke_width=2.5)
        scan.move_to(a["ve"][1][0])
        self.play(Create(scan), run_time=0.2)
        for key in ("ve", "mi", "llm"):
            for i, v in enumerate(IE[key]):
                cell = a[key][1][i]
                self.play(scan.animate.move_to(cell),
                          cell.animate.set_fill(interpolate_color(ManimColor(GREY_E), ManimColor(C_GOAL), v), 1),
                          Transform(bar, pbar(max(v, 0.04), C_GOAL if v > 0.3 else C_COR)), run_time=0.12)
        self.play(FadeOut(scan), Transform(bar, pbar(0.04, C_COR)), run_time=0.3)
        ie = MathParts([("a", r"\mathrm{IE}=P_{h^{(i,l)}_c\,\mathrm{restored}}(y_i)-P_{cor}(y_i),\quad "
                              r"c\in\{\mathrm{ve},\mathrm{mi},\mathrm{llm}\}")], size=0.34).move_to([0, -1.4, 0])
        self.play(FadeIn(ie))
        self.next_slide()

        # ===== Click: different granularities live in different LLM layers =====
        self.play(FadeOut(VGroup(cap, ie)), run_time=0.4)
        big = VGroup(*[Square(0.42, stroke_color=GREY_C, stroke_width=1.2, fill_color=GREY_E, fill_opacity=1)
                       for _ in range(12)]).arrange(RIGHT, buff=0.06).move_to([-0.6, -1.25, 0])
        b_t = txt("LLM layers, split into 4 parts (paper Fig. 4–5, BLIP2-OPT)", 14, True, C_PRETRAIN)
        b_t.next_to(big, UP, buff=0.5)
        parts = ["Frontmost", "Foremost", "Rearmost", "Last"]
        braces = VGroup(*[Brace(big[3 * i:3 * i + 3], DOWN, buff=0.06, color=GREY_B) for i in range(4)])
        p_l = VGroup(*[txt(parts[i], 12, True, GREY_A).next_to(braces[i], DOWN, buff=0.04) for i in range(4)])
        self.play(TransformFromCopy(a["llm"][1], big), FadeIn(b_t), run_time=0.9)
        self.play(FadeIn(braces), FadeIn(p_l))
        kinds = [("entity", C_NEW, 0), ("relation", C_ACCENT, 1), ("action", C_PRETRAIN, 2)]
        tags = VGroup()
        for name, col, pi_ in kinds:
            seg = big[3 * pi_:3 * pi_ + 3]
            box = SurroundingRectangle(seg, buff=0.05, color=col, stroke_width=3)
            t_ = txt(name, 14, True, col).next_to(box, UP, buff=0.08)
            tags.add(VGroup(box, t_))
            self.play(Create(box), FadeIn(t_), *[c.animate.set_fill(col, 0.6) for c in seg], run_time=0.5)
        ours = Text("our fact (capital of Indonesia) is relation knowledge  →  no single fixed edit layer for all facts",
                    font=FONT, font_size=14, color=GREY_A, t2c={"relation": C_ACCENT}).move_to([0, -2.75, 0])
        self.play(FadeIn(ours))
        self.next_slide()

    def m2_locate(self):
        C_K, C_V = "#EC7063", C_ACCENT
        self.m2_start("M2Edit ② pick the key layers (Edit Score search)")
        es = MathParts([("a", r"\mathrm{Edit\ Score}=\frac{4}{\frac{1}{O_{rel}}+\frac{1}{O^{v}_{gen}}+"
                              r"\frac{1}{O^{t}_{gen}}+\frac{1}{O_{loc}}}")], size=0.44)
        es.move_to([3.6, 1.65, 0])
        es_n = txt("harmonic mean of the four metrics: low if ANY of them is low", 12, True, C_ACCENT)
        es_n.next_to(es, DOWN, buff=0.12)
        self.play(Write(es), run_time=1.0)
        self.play(FadeIn(es_n))

        # scatter of training edits -> K-means knowledge centers
        rng = np.random.RandomState(7)
        centers = [np.array(c) for c in [(-5.3, 0.6), (-3.2, 0.9), (-4.9, -1.5), (-2.6, -1.1)]]
        cols = [C_NEW, C_PRETRAIN, C_GOAL, C_S2]
        dots, stars = VGroup(), VGroup()
        for c, col in zip(centers, cols):
            for _ in range(11):
                p = c + rng.randn(2) * 0.32
                dots.add(Dot([p[0], p[1], 0], radius=0.05, color=GREY_B))
            stars.add(Star(5, outer_radius=0.14, inner_radius=0.06, color=col, fill_opacity=1).move_to([c[0], c[1], 0]))
        sc_t = txt("training edits (MLLM representations)", 13, True, GREY_A).move_to([-4.0, 1.75, 0])
        self.play(FadeIn(sc_t), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.01), run_time=1.0)
        km = txt("K-means → knowledge centers (k = 50 in the paper; 4 shown)", 12, color=GREY_B).move_to([-4.0, -2.45, 0])
        self.play(*[dots[i].animate.set_color(cols[i // 11]) for i in range(len(dots))],
                  LaggedStart(*[GrowFromCenter(s_) for s_ in stars], lag_ratio=0.2), FadeIn(km), run_time=1.2)
        self.next_slide()

        # ===== Click: grid search of layer combinations for one center =====
        def strip(n, label, color, y):
            cells = VGroup(*[Square(0.2, stroke_color=GREY_C, stroke_width=1, fill_color=GREY_E, fill_opacity=1)
                             for _ in range(n)]).arrange(RIGHT, buff=0.05)
            cells.move_to([0, y, 0]).align_to([1.9, 0, 0], LEFT)
            lb = txt(label, 12, True, color).move_to([0, y, 0]).align_to([0.1, 0, 0], LEFT)
            return VGroup(lb, cells)

        sl = strip(12, "LLM layer r", C_PRETRAIN, 0.55)
        sv = strip(8, "vision enc. layer s", C_ICE, 0.1)
        sm = strip(6, "interface layer t", C_S2, -0.35)
        es_bar_bg = Rectangle(width=1.4, height=0.16, stroke_width=0, fill_color=GREY_E, fill_opacity=1)
        es_bar_bg.move_to([5.6, 0.1, 0])
        es_lbl = txt("Edit Score", 11, True, GREY_B).next_to(es_bar_bg, UP, buff=0.06)

        def es_fill(v):
            r = Rectangle(width=max(1.4 * v, 0.02), height=0.16, stroke_width=0,
                          fill_color=C_GOAL if v > 0.8 else GREY_B, fill_opacity=1)
            return r.align_to(es_bar_bg, LEFT).align_to(es_bar_bg, DOWN)

        ring = Circle(0.24, color=WHITE, stroke_width=3).move_to(stars[1])
        self.play(Create(ring), FadeIn(sl), FadeIn(sv), FadeIn(sm), FadeIn(es_bar_bg), FadeIn(es_lbl))
        fill = es_fill(0.3)
        self.add(fill)
        hl = VGroup(*[Square(0.27, color=WHITE, stroke_width=2.5) for _ in range(3)])
        trials = [(9, 1, 4, 0.35), (6, 3, 0, 0.5), (1, 7, 2, 0.62), (3, 5, 1, 0.71), (4, 5, 2, 0.88)]
        for k_, (r, s_, t_, v) in enumerate(trials):
            tgt = VGroup(hl[0].copy().move_to(sl[1][r]), hl[1].copy().move_to(sv[1][s_]), hl[2].copy().move_to(sm[1][t_]))
            if k_ == 0:
                hl.become(tgt)
                self.play(FadeIn(hl), Transform(fill, es_fill(v)), run_time=0.35)
            else:
                self.play(hl.animate.become(tgt), Transform(fill, es_fill(v)), run_time=0.35)
        best = (4, 5, 2)
        self.play(*[c.animate.set_fill(C_V, 1) for c in (sl[1][best[0]], sv[1][best[1]], sm[1][best[2]])],
                  hl.animate.set_color(C_V), run_time=0.5)
        eq6 = MathParts([("a", r"(r_j,s_j,t_j)=\arg\max_{r,s,t}\ \mathrm{EditScore}(\hat{\Theta}_{r,s,t}(c_j))")],
                        size=0.3).move_to([3.4, -1.05, 0])
        e6n = txt("for each center c_j: edit every layer combination, keep the best one", 12, color=GREY_A)
        e6n.next_to(eq6, DOWN, buff=0.1)
        self.play(FadeIn(eq6), FadeIn(e6n))
        tags = VGroup(*[txt(t_, 11, True, c).next_to(s_, UP, buff=0.06) for t_, c, s_ in
                        zip(["(2,7,1)", "(5,6,3)", "(3,2,1)", "(9,4,5)"], cols, stars)])
        self.play(FadeIn(tags), FadeOut(ring))
        self.next_slide()

        # ===== Click: a new edit inherits the key layers of its nearest center =====
        newp = np.array([-3.6, -0.15, 0])
        nd = VGroup(Dot(newp, radius=0.1, color=C_V), id_flag(0.22).next_to(newp, DOWN, buff=0.08))
        nd_l = txt("new edit: Indonesia → IKN", 12, True, C_V).next_to(nd, DOWN, buff=0.05)
        self.play(FadeIn(nd, scale=1.5), FadeIn(nd_l))
        links = VGroup(*[DashedLine(newp, s_.get_center(), dash_length=0.06, color=GREY_B, stroke_width=1.5)
                         for s_ in stars])
        self.play(Create(links), run_time=0.6)
        self.play(links[1].animate.set_color(C_V).set_stroke(width=3.5), *[l_.animate.set_opacity(0.2)
                                                                          for i, l_ in enumerate(links) if i != 1])
        eq7 = MathParts([("a", r"j^*=\arg\max_j\ \frac{M(a_i)\cdot M(c_j)}{|M(a_i)|\,|M(c_j)|}")], size=0.32)
        eq7.move_to([3.4, -2.25, 0])
        e7n = txt("nearest center by cosine similarity → reuse its (r, s, t)", 12, color=GREY_A).next_to(eq7, RIGHT,
                                                                                                    buff=0.2)
        e7n.next_to(eq7, DOWN, buff=0.08)
        inherit = tags[1].copy()
        self.play(FadeIn(eq7), FadeIn(e7n))
        self.play(inherit.animate.next_to(nd[0], LEFT, buff=0.1).set_color(C_V), run_time=0.8)
        self.next_slide()

    def m2_edit(self):
        C_K, C_V, C_W = "#EC7063", C_ACCENT, C_GOAL
        self.m2_start("M2Edit ③ edit the key layers together")
        a = self.m2_arch(1.15)
        self.play(FadeIn(a["all"]), FadeIn(a["out"]), run_time=0.8)
        picks = [("llm", 4, "r"), ("ve", 5, "s"), ("mi", 2, "t")]
        marks = VGroup()
        for n_, (key, idx, lab) in enumerate(picks):
            cell = a[key][1][idx]
            badge = VGroup(Circle(0.13, color=C_V, fill_color=C_V, fill_opacity=1), txt(str(n_ + 1), 11, True, BLACK))
            badge.next_to(cell, DOWN, buff=0.12)
            marks.add(VGroup(SurroundingRectangle(cell, buff=0.03, color=C_V, stroke_width=2.5), badge,
                             txt(lab, 11, True, C_V).next_to(cell, UP, buff=0.06)))
        order = Text("edit order: ① LLM layer r → ② visual-encoder layer s → ③ interface layer t   (layers from step ②)",
                     font=FONT, font_size=14, color=GREY_A).move_to([0, -0.35, 0])
        self.play(LaggedStart(*[AnimationGroup(Create(m[0]), FadeIn(m[1]), FadeIn(m[2]),
                                               a[k][1][i].animate.set_fill(C_V, 1)) for m, (k, i, _) in zip(marks, picks)],
                              lag_ratio=0.4), FadeIn(order), run_time=1.4)
        self.next_slide()

        # ===== Click: least squares on each key layer =====
        self.play(FadeOut(order), run_time=0.3)
        kv = VGroup(*[self.cellvec(v, C_K, cell=0.13) for v in ([0.8, 0.3, 0.6, 0.2], [0.7, 0.4, 0.5, 0.3],
                                                                 [0.9, 0.2, 0.7, 0.25])]).arrange(RIGHT, buff=0.06)
        Wb = Square(0.9, stroke_color=C_W, stroke_width=2.5, fill_color=C_W, fill_opacity=0.25)
        zv = VGroup(*[self.cellvec(v, C_V, cell=0.13) for v in ([0.3, 0.9, 0.5, 0.6], [0.35, 0.85, 0.45, 0.6],
                                                                 [0.3, 0.9, 0.55, 0.65])]).arrange(RIGHT, buff=0.06)
        pic = VGroup(kv, Wb, zv).arrange(RIGHT, buff=0.45).move_to([-4.3, -1.45, 0])
        pa = VGroup(Arrow(kv.get_right(), Wb.get_left(), buff=0.06, color=GREY_B, stroke_width=2.5, tip_length=0.1),
                    Arrow(Wb.get_right(), zv.get_left(), buff=0.06, color=GREY_B, stroke_width=2.5, tip_length=0.1))
        wl = MathParts([("w", "W")], size=0.36, color=C_W).move_to(Wb)
        kl = MathParts([("k", r"k_1\ldots k_N")], size=0.24, color=C_K).next_to(kv, DOWN, buff=0.1)
        zl = MathParts([("z", r"z_1\ldots z_N")], size=0.24, color=C_V).next_to(zv, DOWN, buff=0.1)
        pt = txt("output map W of a key layer", 12, True, C_W).next_to(pic, UP, buff=0.15)
        obj = MathParts([("a", r"\min_W\ \sum_{i=1}^{N}"), ("b", r"\Vert Wk_i-z_i\Vert_2^2"), ("p", r"+"),
                         ("c", r"\lambda\Vert W-W^{\prime}\Vert_2^2")], size=0.36)
        obj["b"].set_color(C_V)
        obj["c"].set_color(C_PRETRAIN)
        obj.move_to([2.6, -0.95, 0])
        sol = MathParts([("a", r"W=(\lambda W^{\prime}+\sum_{i=1}^{N}z_ik_i^{\top})(\lambda I+\sum_{i=1}^{N}k_ik_i^{\top})^{-1}")],
                        size=0.34, color=C_W).move_to([2.6, -2.0, 0])
        notes = VGroup(txt("k_i: this layer's input for a pair a_i = (x_i, v_i)  ·  z_i: the output that gives the target b_i",
                           11, color=GREY_A),
                       txt("same closed form as MultEdit, now summed over N pairs of the same knowledge "
                           "(paper Eq. 8–9)", 11, color=GREY_A)).arrange(DOWN, buff=0.05).next_to(sol, DOWN, buff=0.12)
        self.play(FadeIn(pt), FadeIn(kv), FadeIn(kl), GrowArrow(pa[0]), FadeIn(Wb), FadeIn(wl), GrowArrow(pa[1]),
                  FadeIn(zv), FadeIn(zl), run_time=1.0)
        self.play(Write(obj), run_time=0.9)
        self.play(Write(sol), run_time=0.9)
        self.play(FadeIn(notes))
        self.next_slide()

        # ===== Click: apply the three edits in order -> IKN =====
        for m, (key, idx, _) in zip(marks, picks):
            self.play(Flash(a[key][1][idx], color=C_V, flash_radius=0.3, line_length=0.12),
                      Indicate(m[1], color=C_V), run_time=0.5)
        new_out = txt("IKN ✓", 17, True, C_V).move_to(a["out"]).align_to(a["out"], LEFT)
        self.play(Transform(a["out"], new_out), Flash(a["out"], color=C_V, flash_radius=0.5))
        self.next_slide()

        # ===== Click: results (relation knowledge) =====
        self.m2_start("M2Edit results: relation knowledge (BLIP2-OPT)")
        METHODS = [("ROME", "#5DADE2"), ("MEND", "#E59866"), ("MLE (M2Edit)", C_GOAL)]
        GROUPS = ["Reliability", "Text generality", "Visual generality", "Locality"]
        VALS = [[69.2, 80.4, 88.4], [63.7, 77.4, 82.0], [32.5, 55.3, 64.1], [94.2, 95.3, 94.3]]
        BB, SC, BW, X0, WD = -2.0, 0.036, 0.42, -5.6, 8.4
        legend = VGroup(*[VGroup(Square(0.16, stroke_width=0, fill_color=c, fill_opacity=1), txt(n, 13, True, GREY_A))
                          .arrange(RIGHT, buff=0.1) for n, c in METHODS]).arrange(RIGHT, buff=0.5).move_to([-1.4, 2.05, 0])
        axis = Line([X0, BB, 0], [X0 + WD, BB, 0], color=GREY_C, stroke_width=1.5)
        grid = VGroup(*[DashedLine([X0, BB + g * SC, 0], [X0 + WD, BB + g * SC, 0], color=GREY_D, stroke_width=1,
                                   dash_length=0.05) for g in (25, 50, 75, 100)])
        gl = VGroup(*[txt(str(g), 10, color=GREY_B).next_to([X0, BB + g * SC, 0], LEFT, buff=0.06) for g in (25, 50, 75, 100)])
        bars, vals, glabels = VGroup(), VGroup(), VGroup()
        for gi, g in enumerate(GROUPS):
            gc = X0 + WD * (gi + 0.5) / 4
            for mi, (n, c) in enumerate(METHODS):
                v = VALS[gi][mi]
                r = Rectangle(width=BW, height=v * SC, stroke_width=0, fill_color=c, fill_opacity=1)
                r.move_to([gc + (mi - 1) * (BW + 0.04), BB, 0], aligned_edge=DOWN)
                bars.add(r)
                vals.add(txt(f"{v:.1f}", 10, True, C_GOAL if mi == 2 else GREY_A).next_to(r, UP, buff=0.04))
            glabels.add(txt(g, 13, True, GREY_A).next_to([gc, BB, 0], DOWN, buff=0.1))
        self.play(FadeIn(legend), Create(axis), FadeIn(grid), FadeIn(gl), FadeIn(glabels))
        self.play(LaggedStart(*[GrowFromEdge(b_, DOWN) for b_ in bars], lag_ratio=0.05), run_time=1.4)
        self.play(FadeIn(vals))
        vg = SurroundingRectangle(VGroup(*bars[6:9], *vals[6:9]), buff=0.08, color=C_ACCENT, stroke_width=2.5)
        side = VGroup(txt("biggest gain: visual generality", 16, True, C_ACCENT),
                      txt("editing all three components helps", 13, color=GREY_A),
                      txt("the model generalize to new images", 13, color=GREY_A),
                      txt("(paper: +4.4 to +10.8 over the baselines", 12, color=GREY_B),
                      txt("in different settings)", 12, color=GREY_B),
                      txt("Locality stays ≈ 94 for everyone", 13, True, C_GOAL)).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        side[5].shift(DOWN * 0.2)
        side.move_to([0, 0.0, 0]).align_to([3.3, 0, 0], LEFT)
        self.play(Create(vg), FadeIn(side, shift=LEFT * 0.1))
        src = txt("paper Table 2 (BLIP2-OPT, relation subset)", 10, color=GREY_C).move_to([-1.4, BB - 0.6, 0])
        self.play(FadeIn(src))
        self.next_slide()

    # ---------- Recap ----------
    def recap(self):
        rm = self.rm
        kicker = txt("RECAP", size=18, bold=True, color=C_ACCENT).move_to(rm["kicker"]).align_to(rm["kicker"], LEFT)
        head = txt("Knowledge enrichment in VLMs: what we saw", size=40, bold=True)
        head.next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (rm["kicker"], rm["head"])],
                  Transform(rm["kicker"], kicker), Transform(rm["head"], head), run_time=0.9)

        XS, W, H = [-4.55, 0, 4.55], 4.3, 1.18

        def card(title, color, l1, l2, x, y):
            box = RoundedRectangle(corner_radius=0.1, width=W, height=H, stroke_color=color, stroke_width=1.8,
                                   fill_color=C_PANEL, fill_opacity=1).move_to([x, y, 0])
            t_ = txt(title, 17, True, color)
            b_ = VGroup(txt(l1, 13, color=GREY_A), txt(l2, 13, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            if b_.width > W - 0.35:
                b_.scale_to_fit_width(W - 0.35)
            inner = VGroup(t_, b_).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            inner.move_to(box).align_to(box.get_left() + RIGHT * 0.2, LEFT)
            return VGroup(box, inner)

        def column(x, color, header, sub, cards):
            h = VGroup(txt(header, 18, True, color), txt(sub, 12, color=GREY_B)).arrange(DOWN, buff=0.04)
            h.move_to([x, 1.6, 0])
            cs = VGroup(*[card(t_, color, a_, b_, x, 0.72 - i * (H + 0.12)) for i, (t_, a_, b_) in enumerate(cards)])
            return h, cs

        C_AUG, C_TR, C_ED = C_NEW, C_S2, C_GOAL
        cols = [
            column(XS[0], C_AUG, "Augment", "knowledge the model never had", [
                ("LLaVA-Med", "a new domain: biomedical concept alignment", "→ then instruction tuning"),
                ("Yo'LLaVA", "a new personal concept <bo>", "learnable tokens from a few photos")]),
            column(XS[1], C_TR, "Update · by training", "inject evolving facts", [
                ("MMEvoke", "Full-FT / LoRA inject ≤ 18% CEM", "…and forget old knowledge"),
                ("Replay", "mix old data back in (~942 samples)", "→ less forgetting (iCaRL idea)"),
                ("KORE", "knowledge-tree augmentation (adaptation)", "+ null-space LoRA constraint (retention)")]),
            column(XS[2], C_ED, "Update · by editing", "rewrite a few weights", [
                ("ROME", "causal tracing → mid-layer MLPs (LLMs)", "→ closed-form rank-one edit"),
                ("MultEdit", "MLLMs recall facts in EARLY MLPs", "→ closed-form edit at LLaVA layer 2"),
                ("M2Edit (MLE)", "entity / relation / action in different layers", "→ Edit-Score search, edit 3 components")]),
        ]
        upd = txt("Knowledge Update", 13, True, C_ACCENT).move_to([(XS[1] + XS[2]) / 2, 2.12, 0])
        upd_l = Line([XS[1] - 1.9, 1.97, 0], [XS[2] + 1.9, 1.97, 0], color=C_ACCENT, stroke_width=1.5)
        for i, (h, cs) in enumerate(cols):
            anims = [FadeIn(h, shift=DOWN * 0.1), LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cs], lag_ratio=0.2)]
            if i == 1:
                anims += [FadeIn(upd), Create(upd_l)]
            self.play(*anims, run_time=1.2)
            self.next_slide()

        # spectrum: cost vs scope
        spec = Arrow([-6.4, -2.95, 0], [6.4, -2.95, 0], buff=0, color=GREY_B, stroke_width=3, tip_length=0.15)
        spec_b = Arrow([6.4, -2.95, 0], [-6.4, -2.95, 0], buff=0, color=GREY_B, stroke_width=3, tip_length=0.15)
        l_ = txt("whole domains / concepts · more data & compute", 13, True, C_AUG).next_to([-6.4, -2.95, 0], DOWN,
                                                                                          buff=0.12, aligned_edge=LEFT)
        r_ = txt("single facts · cheap & targeted", 13, True, C_ED).next_to([6.4, -2.95, 0], DOWN,
                                                                                       buff=0.12, aligned_edge=RIGHT)
        l_.align_to([-6.4, 0, 0], LEFT)
        r_.align_to([6.4, 0, 0], RIGHT)
        self.play(GrowArrow(spec), GrowArrow(spec_b), FadeIn(l_), FadeIn(r_))
        self.next_slide()

    # ---------- Conclusion ----------
    def conclusion(self):
        rm = self.rm
        kicker = txt("CONCLUSION", size=18, bold=True, color=C_ACCENT).move_to(rm["kicker"]).align_to(rm["kicker"], LEFT)
        head = txt("Takeaways", size=40, bold=True).next_to(kicker, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (rm["kicker"], rm["head"])],
                  Transform(rm["kicker"], kicker), Transform(rm["head"], head), run_time=0.9)

        def point(n, color, title, sub, y):
            badge = VGroup(Circle(0.22, color=color, fill_color=color, fill_opacity=1), txt(str(n), 18, True, BLACK))
            t_ = VGroup(txt(title, 20, True, color), txt(sub, 14, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT,
                                                                                       buff=0.08)
            g = VGroup(badge, t_).arrange(RIGHT, buff=0.3, aligned_edge=UP)
            return g.move_to([0, y, 0]).align_to([-6.2, 0, 0], LEFT)

        pts = [
            point(1, C_NEW, "Whole new domains or concepts → train, with the right data",
                  "LLaVA-Med's two-stage curriculum, Yo'LLaVA's learnable concept tokens", 1.65),
            point(2, C_S2, "Updating facts by fine-tuning is a trade-off: adaptation vs. retention",
                  "shape the data (KORE augmentation) and protect old knowledge (replay, null-space constraint)", 0.55),
            point(3, C_GOAL, "Knowledge editing is cheap and targeted: locate, then write in closed form",
                  "…but in VLMs knowledge is spread over early layers, granularities and components", -0.55),
        ]
        for p_ in pts:
            self.play(FadeIn(p_, shift=RIGHT * 0.15), run_time=0.8)
            self.next_slide()

        qs = VGroup(
            txt("Open questions", 17, True, C_ACCENT),
            VGroup(
                Text("• Does localization tell us where to edit? Non-causal layers also edit well (Hase et al., 2023)",
                     font=FONT, font_size=14, color=GREY_A),
                Text("• Many edits at once / over time: batch editing still degrades (M2Edit, 500 edits)",
                     font=FONT, font_size=14, color=GREY_A),
                Text("• Visual generality is still the weakest metric (MLE: ≈ 60–79%)", font=FONT, font_size=14,
                     color=GREY_A),
                Text("• Train, edit, or retrieve? Even the best RAG reaches only ≈ 41% CEM on MMEvoke", font=FONT,
                     font_size=14, color=GREY_A),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        qs.move_to([0, -2.15, 0])
        qbox = SurroundingRectangle(qs, buff=0.2, corner_radius=0.12, color=C_ACCENT, stroke_width=1.8)
        self.play(Create(qbox), FadeIn(qs[0]))
        self.play(LaggedStart(*[FadeIn(q, shift=RIGHT * 0.1) for q in qs[1]], lag_ratio=0.25), run_time=1.4)
        self.next_slide()

        # ===== Click: thank you =====
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        ty = txt("Thank you!", size=64, bold=True).move_to(UP * 0.6)
        qq = txt("Questions?", size=30, bold=False, color=GREY_A).next_to(ty, DOWN, buff=0.35)
        name = txt("Muhammad Cendekia Airlangga", size=20, color=GREY_B).next_to(qq, DOWN, buff=0.6)
        self.play(GrowFromCenter(ty), run_time=0.6)
        self.play(ty.animate.set_color(C_ACCENT), run_time=0.3)
        self.play(Indicate(ty, scale_factor=1.15, color=C_ACCENT), FadeIn(qq, shift=UP * 0.1), run_time=0.8)
        self.play(FadeIn(name))
        self.next_slide()
