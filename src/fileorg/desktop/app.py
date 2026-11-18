"""Tkinter desktop window for fileorg, styled after the UI design (sidebar, summary cards, review table).

Flow: Organize (pick a folder) -> Scan and preview -> review / edit the table ->
"Move N files" -> confirm. Every move is logged; History and undo restores a run.
Slow work (reading files, loading models, moving) runs in a background thread and
reports back through a queue, so the window never freezes.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Callable, Optional

from fileorg.desktop.session import (
    CLASSIFIERS, UNREADABLE_CHOICES, UNSURE_CHOICES, ReviewSession, analyse, apply_plan, log_path,
)
from fileorg.organizer import MoveLog, undo_batch

BG, WHITE, BORDER, INK, MUTED = "#F4F6F9", "#FFFFFF", "#E1E5EB", "#1B2230", "#566074"
ACCENT, ACCENT_DARK, ACCENT_OFF, NAV_ON = "#2F6FEB", "#1647B0", "#A9C3F5", "#E8F0FF"
AMBER, AMBER_BG = "#7A4300", "#FFF9EF"
CHECKED, UNCHECKED = "\u2611", "\u2610"
CATEGORY_LABELS = {"cv_resume": "CV / resume", "academic_paper": "Academic paper"}
CATEGORY_COLORS = {"invoice": "#173F99", "receipt": "#0B5B4B", "letter": "#5B2A9A", "report": AMBER,
                   "contract": "#8A1F3D", "cv_resume": "#7A5200", "academic_paper": "#1F5F7A", "other": "#3A4453"}
DECIDE = "Needs your decision"


def _open_file(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


class App(tk.Frame):
    def __init__(self, master: tk.Misc, folder: Optional[str] = None) -> None:
        super().__init__(master, bg=BG)
        self.session: Optional[ReviewSession] = None
        self._outcomes: list = []
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.busy = False
        self._sort_reverse: dict[str, bool] = {}

        self.folder = tk.StringVar(value=folder or "")
        self.dest = tk.StringVar()
        self.recursive = tk.BooleanVar(value=False)
        self.classifier = tk.StringVar(value="rules")
        self.model_file = tk.StringVar()
        self.min_conf = tk.StringVar(value="0.50")
        self.unsure = tk.StringVar(value=next(iter(UNSURE_CHOICES)))
        self.unreadable = tk.StringVar(value=next(iter(UNREADABLE_CHOICES)))
        self.status = tk.StringVar(value="Ready. Nothing is moved until you confirm.")
        self.classifier_note = tk.StringVar()
        self.review_info = tk.StringVar()
        self.k_move, self.k_decide, self.k_left = tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")
        self.hist_status = tk.StringVar()
        self.hist_note = tk.StringVar(value="Select a run to see what undoing it does.")

        self._init_fonts(master)
        self._init_style()
        self._build()
        self._on_classifier()
        self._show_panel("setup")
        self._show("organize")
        self.after(100, self._poll)

    # ----------------------------------------------------------- styling
    def _init_fonts(self, master: tk.Misc) -> None:
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        if "Segoe UI" in set(tkfont.families(master)):
            base = "Segoe UI"
        self.f = tkfont.Font(family=base, size=10)
        self.f_bold = tkfont.Font(family=base, size=10, weight="bold")
        self.f_small = tkfont.Font(family=base, size=9)
        self.f_small_bold = tkfont.Font(family=base, size=9, weight="bold")
        self.f_title = tkfont.Font(family=base, size=17, weight="bold")
        self.f_kpi = tkfont.Font(family=base, size=22, weight="bold")

    def _init_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=34, font=self.f, background=WHITE, fieldbackground=WHITE,
                        foreground=INK, borderwidth=0)
        style.map("Treeview", background=[("selected", NAV_ON)], foreground=[("selected", INK)])
        style.configure("Treeview.Heading", font=self.f_small_bold, background="#FAFBFC", foreground=MUTED,
                        relief="flat", padding=(8, 8), borderwidth=0)
        style.map("Treeview.Heading", background=[("active", "#EEF1F5")])
        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=5)
        style.configure("TCheckbutton", background=WHITE, font=self.f)
        style.configure("Accent.Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT, bordercolor=BORDER,
                        lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
        self.option_add("*TCombobox*Listbox.font", self.f)

    def _btn(self, parent: tk.Misc, text: str, command: Callable[[], None], primary: bool = False,
             big: bool = False) -> tk.Button:
        button = tk.Button(parent, text=text, command=command, font=self.f_bold, relief="flat", bd=0, cursor="hand2",
                           padx=26 if big else 16, pady=10 if big else 7,
                           bg=ACCENT if primary else WHITE, fg=WHITE if primary else INK,
                           activebackground=ACCENT_DARK if primary else "#EEF1F5",
                           activeforeground=WHITE if primary else INK,
                           highlightthickness=1, highlightbackground=ACCENT if primary else "#BFC7D3",
                           disabledforeground="#FFFFFF" if primary else "#9AA3B2")
        button.primary = primary  # type: ignore[attr-defined]
        return button

    @staticmethod
    def _enable(button: tk.Button, on: bool) -> None:
        button.configure(state="normal" if on else "disabled")
        if getattr(button, "primary", False):
            button.configure(bg=ACCENT if on else ACCENT_OFF, highlightbackground=ACCENT if on else ACCENT_OFF)

    @staticmethod
    def _card(parent: tk.Misc, **grid) -> tk.Frame:
        return tk.Frame(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)

    def _text(self, parent: tk.Misc, text="", var=None, font=None, fg=INK, bg=WHITE, **kw) -> tk.Label:
        options = dict(font=font or self.f, fg=fg, bg=bg, anchor="w", justify="left")
        options.update(kw)
        return tk.Label(parent, **({"textvariable": var} if var is not None else {"text": text}), **options)

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        side = tk.Frame(self, bg=WHITE, width=220, highlightthickness=1, highlightbackground=BORDER)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        self._text(side, "fileorg", font=self.f_title).pack(fill="x", padx=18, pady=(18, 0))
        self._text(side, "Document organizer", font=self.f_small, fg=MUTED).pack(fill="x", padx=18, pady=(0, 18))
        self.nav: dict[str, tk.Button] = {}
        for key, text in (("organize", "Organize"), ("history", "History and undo"), ("settings", "Settings")):
            b = tk.Button(side, text=text, anchor="w", font=self.f, relief="flat", bd=0, cursor="hand2", padx=12, pady=9,
                          bg=WHITE, fg="#46515F", activebackground="#EEF1F5", command=lambda k=key: self._show(k))
            b.pack(fill="x", padx=10, pady=2)
            self.nav[key] = b

        content = tk.Frame(self, bg=BG)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.pages = {key: tk.Frame(content, bg=BG, padx=28, pady=24) for key in ("organize", "history", "settings")}
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self._build_organize(self.pages["organize"])
        self._build_history(self.pages["history"])
        self._build_settings(self.pages["settings"])

        bar = tk.Frame(self, bg=BG)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8))
        bar.columnconfigure(1, weight=1)
        self.bar = ttk.Progressbar(bar, length=160, mode="determinate", style="Accent.Horizontal.TProgressbar")
        self.bar.grid(row=0, column=0, padx=(0, 10))
        self._text(bar, var=self.status, font=self.f_small, fg=MUTED, bg=BG).grid(row=0, column=1, sticky="w")

    def _show(self, key: str) -> None:
        for name, button in self.nav.items():
            on = name == key
            button.configure(bg=NAV_ON if on else WHITE, fg=ACCENT_DARK if on else "#46515F",
                             font=self.f_bold if on else self.f)
        self.pages[key].tkraise()
        if key == "history":
            self._history_refresh()

    def _show_panel(self, name: str) -> None:
        (self.review if name == "review" else self.setup).tkraise()

    # --------------------------------------------------------- organize
    def _build_organize(self, page: tk.Frame) -> None:
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        self.setup = tk.Frame(page, bg=BG)
        self.review = tk.Frame(page, bg=BG)
        for panel in (self.setup, self.review):
            panel.grid(row=0, column=0, sticky="nsew")
        self._build_setup(self.setup)
        self._build_review(self.review)

    def _build_setup(self, panel: tk.Frame) -> None:
        panel.columnconfigure(0, weight=1)
        self._text(panel, "Organize a folder", font=self.f_title, bg=BG).grid(row=0, column=0, sticky="w")
        self._text(panel, "Choose a folder. You review every move before anything happens.", fg=MUTED, bg=BG).grid(
            row=1, column=0, sticky="w", pady=(2, 16))
        card = self._card(panel)
        card.grid(row=2, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)
        pad = dict(padx=(18, 8), pady=(16, 0))
        self._text(card, "Folder to organize").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(card, textvariable=self.folder, font=self.f).grid(row=0, column=1, sticky="ew", pady=(16, 0))
        self._btn(card, "Browse...", lambda: self._browse_dir(self.folder)).grid(row=0, column=2, padx=18, pady=(16, 0))
        self._text(card, "Move into (optional)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(card, textvariable=self.dest, font=self.f).grid(row=1, column=1, sticky="ew", pady=(16, 0))
        self._btn(card, "Browse...", lambda: self._browse_dir(self.dest)).grid(row=1, column=2, padx=18, pady=(16, 0))
        ttk.Checkbutton(card, text="Include sub-folders", variable=self.recursive).grid(
            row=2, column=1, sticky="w", pady=(12, 0))
        note = tk.Frame(card, bg=WHITE)
        note.grid(row=3, column=0, columnspan=3, sticky="ew", padx=18, pady=(14, 16))
        self._text(note, var=self.classifier_note, fg=MUTED).pack(side="left")
        link = tk.Button(note, text="Change in Settings", font=self.f, relief="flat", bd=0, bg=WHITE, fg=ACCENT_DARK,
                         activebackground=WHITE, cursor="hand2", command=lambda: self._show("settings"))
        link.pack(side="left", padx=8)
        self.scan_btn = self._btn(panel, "Scan and preview", self._scan, primary=True, big=True)
        self.scan_btn.grid(row=3, column=0, sticky="w", pady=16)

    def _build_review(self, panel: tk.Frame) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        head = tk.Frame(panel, bg=BG)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        titles = tk.Frame(head, bg=BG)
        titles.grid(row=0, column=0, sticky="w")
        self._text(titles, "Review before moving", font=self.f_title, bg=BG).pack(anchor="w")
        self._text(titles, var=self.review_info, font=self.f_small, fg=MUTED, bg=BG).pack(anchor="w")
        self._btn(head, "Change folder", self._cancel).grid(row=0, column=1, padx=(0, 8))
        self.rescan_btn = self._btn(head, "Re-scan", self._scan)
        self.rescan_btn.grid(row=0, column=2)

        cards = tk.Frame(panel, bg=BG)
        cards.grid(row=1, column=0, sticky="ew", pady=14)
        for i, (title, var, color) in enumerate((("Will move", self.k_move, ACCENT_DARK), ("Not sure enough", self.k_decide, AMBER),
                                                 ("Left in place", self.k_left, INK))):
            cards.columnconfigure(i, weight=1, uniform="k")
            card = self._card(cards)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 12, 0))
            self._text(card, title, font=self.f_small, fg=MUTED).pack(fill="x", padx=16, pady=(12, 0))
            self._text(card, var=var, font=self.f_kpi, fg=color).pack(fill="x", padx=16, pady=(0, 10))

        table = self._card(panel)
        table.grid(row=2, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        cols = (("use", "Move", 56, "center"), ("file", "File", 400, "w"), ("category", "Category", 150, "w"),
                ("conf", "Confidence", 150, "w"), ("result", "Result", 280, "w"))
        self.tree = ttk.Treeview(table, columns=[c[0] for c in cols], show="headings", selectmode="extended")
        for key, title, width, anchor in cols:
            self.tree.heading(key, text=title, anchor="w" if anchor == "w" else "center", command=lambda k=key: self._sort(k))
            self.tree.column(key, width=width, anchor=anchor, stretch=key in ("file", "result"))
        for category, color in CATEGORY_COLORS.items():
            self.tree.tag_configure("cat_" + category, foreground=color)
        self.tree.tag_configure("muted", foreground=MUTED)
        self.tree.tag_configure("decide", background=AMBER_BG)
        yscroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<Double-1>", self._on_double)
        self.tree.bind("<space>", lambda e: self._toggle(self._selected()))
        self.tree.bind("<Control-a>", lambda e: self.tree.selection_set(self.tree.get_children()))
        for button in ("<Button-3>", "<Button-2>"):
            self.tree.bind(button, self._on_right_click)

        self.menu = tk.Menu(self, tearoff=0)
        self.cat_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_command(label="Move selected", command=lambda: self._include(True))
        self.menu.add_command(label="Leave selected where they are", command=lambda: self._include(False))
        self.menu.add_cascade(label="Set category", menu=self.cat_menu)
        self.menu.add_command(label="Reset to the classifier's choice", command=self._reset)
        self.menu.add_separator()
        self.menu.add_command(label="Open file", command=self._open_selected)

        foot = tk.Frame(panel, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)
        foot.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        foot.columnconfigure(0, weight=1)
        self._text(foot, "Nothing has been moved yet. Every move is logged and can be undone. "
                         "Click the box to include or leave a file; right-click to change its category.",
                   fg=MUTED, font=self.f_small, wraplength=620).grid(row=0, column=0, sticky="w", padx=16, pady=12)
        self._btn(foot, "Cancel", self._cancel).grid(row=0, column=1, padx=(0, 8), pady=10)
        self.move_btn = self._btn(foot, "Move files", self._move, primary=True)
        self.move_btn.grid(row=0, column=2, padx=(0, 16), pady=10)
        self._enable(self.move_btn, False)

    # ---------------------------------------------------------- settings
    def _build_settings(self, page: tk.Frame) -> None:
        page.columnconfigure(0, weight=1)
        self._text(page, "Settings", font=self.f_title, bg=BG).grid(row=0, column=0, sticky="w")
        self._text(page, "How files are classified, and what happens to files the classifier cannot place.",
                   fg=MUTED, bg=BG).grid(row=1, column=0, sticky="w", pady=(2, 16))
        card = self._card(page)
        card.grid(row=2, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)
        pad = dict(padx=(18, 12), pady=(16, 0))

        self._text(card, "Approach").grid(row=0, column=0, sticky="w", **pad)
        box = ttk.Combobox(card, textvariable=self.classifier, values=CLASSIFIERS, state="readonly", width=14, font=self.f)
        box.grid(row=0, column=1, sticky="w", pady=(16, 0))
        box.bind("<<ComboboxSelected>>", lambda e: self._on_classifier())
        self._text(card, "Model file").grid(row=1, column=0, sticky="w", **pad)
        self.model_entry = ttk.Entry(card, textvariable=self.model_file, font=self.f)
        self.model_entry.grid(row=1, column=1, sticky="ew", pady=(16, 0))
        self.model_btn = self._btn(card, "Browse...", self._browse_model)
        self.model_btn.grid(row=1, column=2, padx=18, pady=(16, 0))
        self._text(card, "Minimum confidence").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(card, from_=0.0, to=1.0, increment=0.05, textvariable=self.min_conf, width=7, format="%.2f",
                    font=self.f).grid(row=2, column=1, sticky="w", pady=(16, 0))
        self._text(card, "Unsure files").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(card, textvariable=self.unsure, values=list(UNSURE_CHOICES), state="readonly", width=28,
                     font=self.f).grid(row=3, column=1, sticky="w", pady=(16, 0))
        self._text(card, "Unreadable files").grid(row=4, column=0, sticky="w", padx=(18, 12), pady=(16, 18))
        ttk.Combobox(card, textvariable=self.unreadable, values=list(UNREADABLE_CHOICES), state="readonly", width=28,
                     font=self.f).grid(row=4, column=1, sticky="w", pady=(16, 18))
        self._text(page, "Rules needs no training. ML and embeddings need a model made with `fileorg train`.",
                   fg=MUTED, font=self.f_small, bg=BG).grid(row=3, column=0, sticky="w", pady=10)

    # ----------------------------------------------------------- history
    def _build_history(self, page: tk.Frame) -> None:
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)
        self._text(page, "History and undo", font=self.f_title, bg=BG).grid(row=0, column=0, sticky="w")
        self._text(page, var=self.hist_status, fg=MUTED, bg=BG).grid(row=1, column=0, sticky="w", pady=(2, 12))
        card = self._card(page)
        card.grid(row=2, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(0, weight=1)
        cols = (("id", "Run", 60), ("created", "When (UTC)", 200), ("classifier", "Classifier", 110), ("done", "Moved", 80),
                ("failed", "Failed", 80), ("skipped", "Skipped", 80), ("status", "Status", 110))
        self.hist = ttk.Treeview(card, columns=[c[0] for c in cols], show="headings", selectmode="browse")
        for key, title, width in cols:
            self.hist.heading(key, text=title, anchor="w")
            self.hist.column(key, width=width, anchor="w")
        self.hist.grid(row=0, column=0, sticky="nsew")
        self.hist.bind("<<TreeviewSelect>>", lambda e: self._history_note())
        note = self._card(page)
        note.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        note.columnconfigure(0, weight=1)
        self._text(note, var=self.hist_note, wraplength=720).grid(row=0, column=0, sticky="w", padx=18, pady=14)
        self.undo_btn = self._btn(note, "Undo run", self._undo, primary=True)
        self.undo_btn.grid(row=0, column=1, padx=18, pady=10)

    # ----------------------------------------------------------- helpers
    def _browse_dir(self, var: tk.StringVar) -> None:
        chosen = filedialog.askdirectory(initialdir=var.get() or str(Path.home()))
        if chosen:
            var.set(chosen)

    def _browse_model(self) -> None:
        chosen = filedialog.askopenfilename(title="Trained model", filetypes=[("Model", "*.joblib"), ("All files", "*.*")])
        if chosen:
            self.model_file.set(chosen)

    def _on_classifier(self) -> None:
        needs_model = self.classifier.get() != "rules"
        self.model_entry.state(["!disabled"] if needs_model else ["disabled"])
        self._enable(self.model_btn, needs_model)
        if needs_model and not self.model_file.get():
            for candidate in (f"models/{self.classifier.get()}.joblib", "models/model.joblib"):
                if Path(candidate).is_file():
                    self.model_file.set(str(Path(candidate).resolve()))
                    break
        self.classifier_note.set(f"Classifier: {self.classifier.get()}")

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        for button in (self.scan_btn, self.rescan_btn, self.undo_btn):
            self._enable(button, not busy)
        self._enable(self.move_btn, not busy and self._plan_size() > 0)
        if busy:
            self.status.set(message)
            self.bar.configure(mode="indeterminate")
            self.bar.start(12)
        else:
            self.bar.stop()
            self.bar.configure(mode="determinate", value=0)

    def _plan_size(self) -> int:
        return sum(1 for o in self._outcomes if o.action == "move") if self.session else 0

    def _run(self, work: Callable[[], object], done: Callable[[object], None], message: str) -> None:
        if self.busy:
            return
        self._set_busy(True, message)

        def target() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - show every failure in the window
                self.jobs.put(("error", exc, traceback.format_exc()))
            else:
                self.jobs.put(("done", done, result))

        threading.Thread(target=target, daemon=True).start()

    def _progress_cb(self, label: str) -> Callable[[int, int], None]:
        def callback(done: int, total: int) -> None:
            if done % 5 == 0 or done == total:
                self.jobs.put(("progress", label, done, total))
        return callback

    def _stage_cb(self, message: str) -> None:
        self.jobs.put(("stage", message))

    def _poll(self) -> None:
        try:
            while True:
                kind, *args = self.jobs.get_nowait()
                if kind == "progress":
                    label, done, total = args
                    self.bar.stop()
                    self.bar.configure(mode="determinate", maximum=max(total, 1), value=done)
                    self.status.set(f"{label} {done}/{total}")
                elif kind == "stage":
                    self.bar.configure(mode="indeterminate")
                    self.bar.start(12)
                    self.status.set(args[0])
                elif kind == "done":
                    self._set_busy(False)
                    callback, result = args
                    callback(result)
                elif kind == "error":
                    self._set_busy(False)
                    self.status.set("Something went wrong.")
                    exc, trace = args
                    print(trace, file=sys.stderr)
                    messagebox.showerror("fileorg", str(exc) or exc.__class__.__name__)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # -------------------------------------------------------------- scan
    def _scan(self) -> None:
        folder = self.folder.get().strip()
        if not folder or not Path(folder).is_dir():
            self._show("organize")
            self._show_panel("setup")
            messagebox.showwarning("fileorg", "Choose an existing folder first.")
            return
        try:
            min_conf = float(self.min_conf.get().replace(",", "."))
            if not 0 <= min_conf <= 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("fileorg", "Minimum confidence (Settings) must be a number between 0 and 1.")
            return
        options = dict(
            dest=self.dest.get().strip() or None, classifier_name=self.classifier.get(),
            model_file=self.model_file.get().strip() or None, recursive=self.recursive.get(),
            min_confidence=min_conf, unsure=UNSURE_CHOICES[self.unsure.get()],
            unreadable=UNREADABLE_CHOICES[self.unreadable.get()])

        def work() -> ReviewSession:
            return analyse(folder, progress=self._progress_cb("Reading files"), on_stage=self._stage_cb, **options)

        self._run(work, self._scanned, "Reading files...")

    def _scanned(self, session: ReviewSession) -> None:
        if not session.rows:
            self.status.set("No PDF, DOCX or TXT files found.")
            messagebox.showinfo("fileorg", "No PDF, DOCX or TXT files were found in this folder.")
            return
        self.session = session
        self._refresh(rebuild=True)
        self._show_panel("review")
        self.status.set("Preview ready. Nothing has been moved yet.")

    def _cancel(self) -> None:
        self.session, self._outcomes = None, []
        self._refresh(rebuild=True)
        self._show_panel("setup")
        self.status.set("Ready. Nothing is moved until you confirm.")

    # ------------------------------------------------------------- table
    def _values(self, row, out) -> tuple:
        mark = CHECKED if out.action == "move" else ("\u2013" if out.reason == "already_in_place" else UNCHECKED)
        if row.overridden:
            conf = "chosen by you"
        elif row.confidence is None:
            conf = "\u2013"
        else:
            filled = round(row.confidence * 5)
            conf = "\u25A0" * filled + "\u25A1" * (5 - filled) + f"  {row.confidence:.2f}"
        if out.action == "move":
            folder = out.destination.rsplit("/", 1)[0]
            result = f"Will move to {folder}/" + (" (renamed)" if "renamed" in out.text else "")
        elif out.action == "excluded":
            result = "Left in place (you excluded it)"
        elif out.reason == "low_confidence":
            result = DECIDE
        elif out.reason == "unreadable":
            result = "No readable text, left in place"
        else:
            result = out.text[:1].upper() + out.text[1:]
        return (mark, row.record.rel_path, _label(out.category), conf, result)

    @staticmethod
    def _tags(out) -> tuple:
        if out.reason == "low_confidence":
            return ("decide",)
        if out.action != "move":
            return ("muted",)
        return ("cat_" + out.category,)

    def _refresh(self, rebuild: bool = False) -> None:
        session = self.session
        if session is None:
            self.tree.delete(*self.tree.get_children())
            self.k_move.set("0"), self.k_decide.set("0"), self.k_left.set("0")
            self._enable(self.move_btn, False)
            return
        plan = session.plan()
        self._outcomes = session.outcomes(plan)
        if rebuild:
            self.tree.delete(*self.tree.get_children())
            for i, (row, out) in enumerate(zip(session.rows, self._outcomes)):
                self.tree.insert("", "end", iid=str(i), values=self._values(row, out), tags=self._tags(out))
        else:
            for i, (row, out) in enumerate(zip(session.rows, self._outcomes)):
                self.tree.item(str(i), values=self._values(row, out), tags=self._tags(out))
        moving = len(plan.moves)
        decide = sum(1 for o in self._outcomes if o.reason == "low_confidence")
        self.k_move.set(str(moving))
        self.k_decide.set(str(decide))
        self.k_left.set(str(len(self._outcomes) - moving - decide))
        dest = f" \u2192 {plan.dest_root}" if plan.dest_root != plan.root else ""
        self.review_info.set(f"{session.root}{dest}  \u00b7  {session.classifier_name} classifier  \u00b7  {len(session.rows)} files")
        self.move_btn.configure(text=f"Move {moving} file{'s' if moving != 1 else ''}")
        self._enable(self.move_btn, moving > 0 and not self.busy)

    def _selected(self) -> list[int]:
        return [int(i) for i in self.tree.selection()]

    def _toggle(self, indices: list[int]) -> None:
        session = self.session
        if not session:
            return
        for i in indices:
            out, row = self._outcomes[i], session.rows[i]
            if out.action == "move":
                session.set_include([i], False)
            elif out.action == "excluded":
                session.set_include([i], True)
            elif out.reason in ("low_confidence", "unreadable"):
                session.set_label([i], row.label)   # the user confirms: move it
        self._refresh()

    def _on_click(self, event: tk.Event) -> Optional[str]:
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) == "#1":
            iid = self.tree.identify_row(event.y)
            if iid:
                self._toggle([int(iid)])
            return "break"
        return None

    def _include(self, include: bool) -> None:
        session = self.session
        if not session:
            return
        for i in self._selected():
            if include:
                session.set_include([i], True)
                if self._outcomes[i].reason in ("low_confidence", "unreadable"):
                    session.set_label([i], session.rows[i].label)
            else:
                session.set_include([i], False)
        self._refresh()

    def _set_category(self, category: str) -> None:
        if self.session and self._selected():
            self.session.set_label(self._selected(), category)
            self._refresh()

    def _reset(self) -> None:
        if self.session and self._selected():
            self.session.reset(self._selected())
            self._refresh()

    def _on_right_click(self, event: tk.Event) -> None:
        if not self.session:
            return
        iid = self.tree.identify_row(event.y)
        if iid and iid not in self.tree.selection():
            self.tree.selection_set(iid)
        if not self.tree.selection():
            return
        self.cat_menu.delete(0, "end")
        for category in self.session.categories():
            self.cat_menu.add_command(label=_label(category), command=lambda c=category: self._set_category(c))
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _open_selected(self) -> None:
        if self.session and self._selected():
            try:
                _open_file(self.session.rows[self._selected()[0]].record.path)
            except OSError as exc:
                messagebox.showerror("fileorg", f"Could not open the file: {exc}")

    def _on_double(self, event: tk.Event) -> None:
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) != "#1":
            self._open_selected()

    def _sort(self, column: str) -> None:
        reverse = self._sort_reverse.get(column, False)
        items = sorted(((self.tree.set(k, column).lower(), k) for k in self.tree.get_children("")), reverse=reverse)
        for position, (_, key) in enumerate(items):
            self.tree.move(key, "", position)
        self._sort_reverse[column] = not reverse

    # -------------------------------------------------------------- move
    def _move(self) -> None:
        session = self.session
        if session is None:
            return
        plan = session.plan()
        if not plan.moves:
            messagebox.showinfo("fileorg", "Nothing to move.")
            return
        target = f" into {plan.dest_root}" if plan.dest_root != plan.root else ""
        if not messagebox.askyesno(
                "Move files", f"Move {len(plan.moves)} file(s){target} now?\n\n"
                "Every move is logged, and you can undo this run under History and undo."):
            return
        self._run(lambda: apply_plan(session, progress=self._progress_cb("Moving files")), self._moved, "Moving files...")

    def _moved(self, result) -> None:
        self._cancel()
        self.status.set(f"Moved {result.done} file(s). Scan again to review the folder.")
        text = f"Moved {result.done} file(s) (run {result.batch_id}); {result.failed} failed, {result.skipped} skipped."
        if result.problems:
            text += "\n\n" + "\n".join(f"{p.name}: {m}" for p, m in result.problems[:8])
            if len(result.problems) > 8:
                text += f"\n... and {len(result.problems) - 8} more"
        (messagebox.showwarning if result.failed else messagebox.showinfo)("fileorg", text)

    # ----------------------------------------------------------- history
    def _history_refresh(self) -> None:
        self.hist.delete(*self.hist.get_children())
        folder = self.folder.get().strip()
        if not folder:
            self.hist_status.set("Choose a folder on the Organize page first.")
            return
        db = log_path(folder)
        if not db.exists():
            self.hist_status.set(f"No runs yet for {folder}.")
            return
        with MoveLog(db) as log:
            batches = log.batches()
        for b in reversed(batches):
            status = "Undone" if b.undone else ("Active" if b.finished else "Unfinished")
            created = b.created.replace("T", " ").replace("+00:00", "")
            self.hist.insert("", "end", iid=str(b.id), values=(b.id, created, b.classifier, b.done, b.failed, b.skipped, status))
        self.hist_status.set(f"Runs for {folder}")
        if batches:
            self.hist.selection_set(str(batches[-1].id))

    def _history_note(self) -> None:
        selection, folder = self.hist.selection(), self.folder.get().strip()
        if not selection or not folder or not log_path(folder).exists():
            return
        with MoveLog(log_path(folder)) as log:
            batch_id = int(selection[0])
            info = log.batch(batch_id)
            pending = len(log.moves(batch_id, statuses=("done", "pending", "undo_failed")))
        if info is None or info.undone or pending == 0:
            self.hist_note.set(f"Run {batch_id} has nothing left to undo.")
        else:
            self.hist_note.set(f"Run {batch_id} will move {pending} file(s) back. A file that was edited or replaced "
                               "since the run is never overwritten; it is reported and you decide.")

    def _undo(self) -> None:
        folder = self.folder.get().strip()
        db = log_path(folder) if folder else None
        if db is None or not db.exists():
            messagebox.showinfo("fileorg", "There is no move log for this folder yet.")
            return
        selection = self.hist.selection()
        with MoveLog(db) as log:
            batch_id = int(selection[0]) if selection else log.latest_undoable()
            if batch_id is None:
                messagebox.showinfo("fileorg", "Nothing to undo.")
                return
            info = log.batch(batch_id)
            if info is None or info.undone:
                messagebox.showinfo("fileorg", f"Run {batch_id} was already undone.")
                return
            pending = len(log.moves(batch_id, statuses=("done", "pending", "undo_failed")))
        if not messagebox.askyesno("Undo", f"Run {batch_id} ({info.created}): move {pending} file(s) back to where they were?"):
            return

        def work():
            with MoveLog(db) as log:
                return undo_batch(log, batch_id)

        self._run(work, self._undone, "Restoring files...")

    def _undone(self, result) -> None:
        self.status.set(f"Restored {result.restored} file(s).")
        text = (f"Restored {result.restored} file(s); {result.failed} could not be restored, {result.skipped} were never moved; "
                f"{result.removed_dirs} empty folder(s) removed.")
        if result.problems:
            text += "\n\n" + "\n".join(f"{p.name}: {m}" for p, m in result.problems[:8])
        (messagebox.showwarning if result.failed else messagebox.showinfo)("fileorg", text)
        self._history_refresh()


def _enable_dpi_awareness() -> None:
    """Sharp text on high-DPI Windows screens."""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def main(folder: Optional[str] = None) -> int:
    _enable_dpi_awareness()
    root = tk.Tk()
    root.title("fileorg - document organizer")
    root.geometry("1280x800")
    root.minsize(1000, 640)
    root.configure(bg=BG)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    App(root, folder).grid(row=0, column=0, sticky="nsew")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())