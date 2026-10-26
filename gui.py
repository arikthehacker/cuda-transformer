"""One screen for training the model on your own writing: python gui.py"""
from __future__ import annotations

import os
import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font

import pipeline
from pipeline import FILE_TYPES, MY_WRITING, PUBLIC, STEPS, Options, Runner, Step

BG = "#0c0c0c"
PANE = "#141414"
FG = "#c8c8c8"
DIM = "#6b6b6b"
AMBER = "#f0a33a"
GREEN = "#6fcf7f"
RED = "#ef6f6f"
SELECT = "#3a2c14"
LEFT_CHARS = 52
BAR = 16
MARKS = {"waiting": (" ", DIM), "running": (">", AMBER), "done": ("✓", GREEN),
         "failed": ("x", RED), "stopped": ("-", DIM)}
STEP_WEIGHTS = {"gather": 0.1, "mix": 0.05, "tokenize": 0.15, "train": 0.7}
MAX_LOG_LINES = 2000


class App:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.runner: Runner | None = None
        self.tok = None
        self.model = None
        self.loading = False
        self.tags: list[str] = []
        self.tag_index = 0
        self.public = True
        self.confirm_remove = False
        self.step_state = {s.key: ("waiting", "") for s in STEPS}
        self.done_weight = 0.0
        self.current = STEPS[0].key

        families = set(font.families())
        mono = next((f for f in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono") if f in families),
                    "TkFixedFont")
        self.font = (mono, 10)
        self.bold = (mono, 10, "bold")

        root.title("cuda-transformer")
        root.configure(bg=BG)
        root.geometry("1200x760")
        root.minsize(960, 560)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        top = tk.Frame(root, bg=BG)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 4))
        tk.Label(top, text="cuda-transformer", bg=BG, fg=AMBER, font=self.bold).pack(side="left")
        tk.Label(top, text="  train a small GPT on your own writing", bg=BG, fg=DIM, font=self.font).pack(side="left")
        self.status = tk.Label(top, text="", bg=BG, fg=DIM, font=self.font)
        self.status.pack(side="right")

        left = tk.Frame(root, bg=BG)
        left.grid(row=1, column=0, sticky="ns", padx=(14, 10), pady=(0, 12))
        right = tk.Frame(root, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 14), pady=(0, 12))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)

        self.make_writing(left)
        self.make_build(left)
        self.make_write(left)

        self.rule(right, "log").grid(row=0, column=0, sticky="ew")
        self.log = self.pane(right)
        self.log.grid(row=1, column=0, sticky="nsew")
        self.log.tag_configure("head", foreground=AMBER)
        self.log.tag_configure("error", foreground=RED)
        self.log.tag_configure("good", foreground=GREEN)
        self.rule(right, "output").grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.output = self.pane(right)
        self.output.grid(row=3, column=0, sticky="nsew")
        self.output.tag_configure("yours", foreground=AMBER)

        self.refresh_sources()
        self.draw_steps()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.pump)
        root.after(200, self.load_model)

    def rule(self, parent: tk.Widget, title: str) -> tk.Label:
        return tk.Label(parent, text=f"── {title} " + "─" * 200, bg=BG, fg=DIM, font=self.font, anchor="w")

    def text(self, parent: tk.Widget, value: str = "", fg: str = FG) -> tk.Label:
        return tk.Label(parent, text=value, bg=BG, fg=fg, font=self.font, anchor="w", justify="left")

    def button(self, parent: tk.Widget, label: str, command) -> tk.Button:
        return tk.Button(parent, text=f"[ {label} ]", command=command, bg=BG, fg=AMBER, activebackground=SELECT,
                         activeforeground=AMBER, disabledforeground=DIM, relief="flat", bd=0,
                         highlightthickness=0, padx=0, pady=0, font=self.font, cursor="hand2")

    def entry(self, parent: tk.Widget, value: str, width: int) -> tk.Entry:
        box = tk.Entry(parent, width=width, bg=PANE, fg=FG, insertbackground=AMBER, relief="flat",
                       highlightthickness=1, highlightbackground=PANE, highlightcolor=AMBER, font=self.font)
        box.insert(0, value)
        return box

    def pane(self, parent: tk.Widget) -> tk.Text:
        return tk.Text(parent, bg=PANE, fg=FG, insertbackground=AMBER, selectbackground=SELECT,
                       selectforeground=AMBER, relief="flat", bd=0, highlightthickness=0, wrap="word",
                       padx=10, pady=8, font=self.font, state="disabled", undo=False)

    def row(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="x", pady=1)
        return frame

    def make_writing(self, parent: tk.Widget) -> None:
        self.rule(parent, "your writing").pack(fill="x")
        self.sources = tk.Listbox(parent, height=6, width=LEFT_CHARS, bg=PANE, fg=FG, selectbackground=SELECT,
                                  selectforeground=AMBER, activestyle="none", relief="flat", bd=0,
                                  highlightthickness=0, font=self.font, exportselection=False)
        self.sources.pack(fill="x", pady=(2, 4))
        self.sources.bind("<<ListboxSelect>>", self.source_picked)

        line = self.row(parent)
        self.text(line, "tag ").pack(side="left")
        self.tag_entry = self.entry(line, "mine", 14)
        self.tag_entry.pack(side="left")
        self.button(line, "+ files", self.add_files).pack(side="left", padx=(10, 0))
        self.button(line, "+ folder", self.add_folder).pack(side="left", padx=(6, 0))
        self.remove_button = self.button(line, "remove", self.remove_selected)
        self.remove_button.pack(side="right")

        line = self.row(parent)
        self.public_toggle = self.button(line, "", self.toggle_public)
        self.public_toggle.pack(side="left")
        self.button(line, "open", self.open_folder).pack(side="right")
        self.draw_public()
        self.text(parent, f"copied into my_writing/<tag>/  ·  {' '.join(FILE_TYPES)}", DIM).pack(fill="x")

    def make_build(self, parent: tk.Widget) -> None:
        self.rule(parent, "build").pack(fill="x", pady=(14, 2))
        self.step_labels = {}
        for step in STEPS:
            label = self.text(parent)
            label.pack(fill="x")
            self.step_labels[step.key] = label
        line = self.row(parent)
        line.pack_configure(pady=(6, 0))
        self.text(line, "max steps ").pack(side="left")
        self.steps_entry = self.entry(line, str(pipeline.DEFAULT_STEPS), 7)
        self.steps_entry.pack(side="left")
        self.stop_button = self.button(line, "stop", self.stop_build)
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(side="right")
        self.build_button = self.button(line, "build", self.start_build)
        self.build_button.pack(side="right", padx=(0, 8))
        self.text(parent, "stops early once the model stops improving", DIM).pack(fill="x")

    def make_write(self, parent: tk.Widget) -> None:
        self.rule(parent, "write").pack(fill="x", pady=(14, 2))
        self.model_line = self.text(parent, "", DIM)
        self.model_line.pack(fill="x")

        line = self.row(parent)
        self.text(line, "style ").pack(side="left")
        self.style_button = self.button(line, "none", self.next_style)
        self.style_button.pack(side="left")
        self.temp_entry = self.entry(line, "0.8", 4)
        self.temp_entry.pack(side="right")
        self.text(line, "  heat ").pack(side="right")
        self.length_entry = self.entry(line, "200", 5)
        self.length_entry.pack(side="right")
        self.text(line, "  tokens ").pack(side="right")

        line = self.row(parent)
        self.text(line, "start ").pack(side="left")
        self.opening = self.entry(line, "", 34)
        self.opening.pack(side="left", fill="x", expand=True)
        self.opening.bind("<Return>", lambda e: self.start_write())

        line = self.row(parent)
        line.pack_configure(pady=(4, 0))
        self.write_button = self.button(line, "write", self.start_write)
        self.write_button.configure(state="disabled")
        self.write_button.pack(side="left")
        self.write_note = self.text(line, "", DIM)
        self.write_note.pack(side="left", padx=(10, 0))

    def later(self, fn, *args) -> None:
        self.events.put((fn, args))

    def pump(self) -> None:
        log, out = [], []
        try:
            while True:
                fn, args = self.events.get_nowait()
                if fn == self.log_line:
                    log.append(args)
                    continue
                if fn == self.append_output:
                    out.append(args[0])
                    continue
                self.flush_log(log)
                log = []
                fn(*args)
        except queue.Empty:
            pass
        self.flush_log(log)
        if out:
            self.append_output("".join(out))
        self.root.after(30, self.pump)

    def log_line(self, line: str, tag: str = "") -> None:
        self.flush_log([(line, tag)])

    def flush_log(self, lines: list[tuple]) -> None:
        if not lines:
            return
        at_end = self.log.yview()[1] > 0.98
        self.log.configure(state="normal")
        for args in lines:
            line, tag = args[0], args[1] if len(args) > 1 else ""
            if not tag and line.startswith(("error", "Traceback")):
                tag = "error"
            self.log.insert("end", line + "\n", tag)
        count = int(self.log.index("end-1c").split(".")[0])
        if count > MAX_LOG_LINES:
            self.log.delete("1.0", f"{count - MAX_LOG_LINES}.0")
        self.log.configure(state="disabled")
        if at_end:
            self.log.see("end")

    def say(self, text: str) -> None:
        self.status.configure(text=text)

    def draw_public(self) -> None:
        mark = "x" if self.public else " "
        self.public_toggle.configure(text=f"[{mark}] also learn from the Bible and Austen")

    def toggle_public(self) -> None:
        self.public = not self.public
        self.draw_public()

    def refresh_sources(self) -> None:
        self.found = pipeline.sources()
        self.sources.delete(0, "end")
        if not self.found:
            self.sources.insert("end", " nothing yet. type a tag, then + files")
            self.sources.itemconfigure(0, foreground=DIM)
            return
        width = max(len(s.tag) for s in self.found) + 3
        for s in self.found:
            size = pipeline.size_text(s.size) if s.size else ""
            self.sources.insert("end", f" <{s.tag}>".ljust(width + 1) + f"{s.files:4d} files {size:>9s}")

    def source_picked(self, event=None) -> None:
        chosen = self.sources.curselection()
        if chosen and self.found:
            self.tag_entry.delete(0, "end")
            self.tag_entry.insert(0, self.found[chosen[0]].tag)
            self.say(self.found[chosen[0]].where)

    def add_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        try:
            name, copied, skipped = pipeline.add(self.tag_entry.get().strip() or "mine", paths)
        except ValueError as err:
            self.say(str(err))
            return
        self.refresh_sources()
        extra = f", skipped {len(skipped)}" if skipped else ""
        self.say(f"added {copied} to <{name}>{extra}")

    def add_files(self) -> None:
        types = [("writing", " ".join(f"*{t}" for t in FILE_TYPES)), ("all files", "*.*")]
        self.add_paths([Path(p) for p in filedialog.askopenfilenames(parent=self.root, filetypes=types)])

    def add_folder(self) -> None:
        chosen = filedialog.askdirectory(parent=self.root)
        self.add_paths([Path(chosen)] if chosen else [])

    def remove_selected(self) -> None:
        chosen = self.sources.curselection()
        if not chosen or not self.found:
            self.say("pick a row first")
            return
        source = self.found[chosen[0]]
        if not source.where.startswith("my_writing"):
            self.say(f"<{source.tag}> is listed in sources.local.txt, edit that file")
            return
        if not self.confirm_remove:
            self.confirm_remove = True
            self.remove_button.configure(text="[ sure? ]", fg=RED)
            self.say(f"click again to delete the copies in {source.where}")
            self.root.after(3000, self.reset_remove)
            return
        self.reset_remove()
        pipeline.remove(source.tag)
        self.refresh_sources()
        self.say(f"removed <{source.tag}>, your originals stay")

    def reset_remove(self) -> None:
        self.confirm_remove = False
        self.remove_button.configure(text="[ remove ]", fg=AMBER)

    def open_folder(self) -> None:
        MY_WRITING.mkdir(exist_ok=True)
        if pipeline.ON_WINDOWS:
            os.startfile(MY_WRITING)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(MY_WRITING)])

    def draw_steps(self) -> None:
        for n, step in enumerate(STEPS, 1):
            state, extra = self.step_state[step.key]
            mark, color = MARKS[state]
            self.step_labels[step.key].configure(text=f"{mark} {n}. {step.title:27s}{extra}", fg=color)

    def set_progress(self, part: float) -> None:
        total = min(self.done_weight + STEP_WEIGHTS[self.current] * part, 1.0)
        filled = round(BAR * part)
        self.step_state[self.current] = ("running", "█" * filled + "░" * (BAR - filled) + f" {100 * part:3.0f}%")
        self.draw_steps()
        self.say(f"building {100 * total:.0f}%")

    def set_step(self, step: Step, state: str, secs: float) -> None:
        if state == "running":
            self.current = step.key
            self.log_line(f"== {step.title}", "head")
            self.set_progress(0.0)
            return
        self.step_state[step.key] = (state, f"{secs:.0f} s" if secs >= 1 else f"{secs:.1f} s")
        if state == "done":
            self.done_weight += STEP_WEIGHTS[step.key]
        self.draw_steps()

    def start_build(self) -> None:
        try:
            steps = max(int(self.steps_entry.get()), 50)
        except ValueError:
            self.say("max steps must be a whole number")
            return
        opts = Options(public=self.public, steps=steps)
        self.build_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.write_button.configure(state="disabled")
        self.model_line.configure(text="building a new model", fg=DIM)
        self.model = None
        self.tok = None
        self.step_state = {s.key: ("waiting", "") for s in STEPS}
        self.done_weight = 0.0
        self.draw_steps()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.say("checking what is installed")
        threading.Thread(target=self.build_worker, args=(opts,), daemon=True).start()

    def build_worker(self, opts: Options) -> None:
        missing = [(label, fix) for ok, label, fix in pipeline.checks(opts) if not ok]
        if missing:
            for label, fix in missing:
                self.later(self.log_line, f"error: missing {label}. fix: {fix}")
            self.later(self.build_finished, False)
            return
        current = [STEPS[0].key]

        def on_line(line: str) -> None:
            self.later(self.log_line, line)
            part = pipeline.progress(current[0], line, opts)
            if part is not None:
                self.later(self.set_progress, part)

        def on_step(step: Step, state: str, secs: float) -> None:
            current[0] = step.key
            self.later(self.set_step, step, state, secs)

        self.runner = Runner(on_line)
        ok = pipeline.build(self.runner, opts, on_step=on_step)
        self.later(self.build_finished, ok)

    def stop_build(self) -> None:
        if self.runner:
            self.runner.stop()
            self.say("stopping")

    def build_finished(self, ok: bool) -> None:
        stopped = self.runner is not None and self.runner.stopped
        self.runner = None
        self.build_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if ok:
            self.log_line("your model is ready", "good")
            self.say("built")
        else:
            self.say("stopped" if stopped else "build failed, see the log")
        self.load_model()

    def load_model(self) -> None:
        if self.model or self.loading or self.runner:
            return
        if not pipeline.CHECKPOINT.exists():
            self.model_line.configure(text="no model yet: build one first")
            return
        self.loading = True
        self.write_button.configure(state="disabled")
        self.model_line.configure(text="loading the model")
        threading.Thread(target=self.load_worker, daemon=True).start()

    def load_worker(self) -> None:
        try:
            from playground import Model, Tokenizer

            tok = Tokenizer()
            model = Model(pipeline.CHECKPOINT, tok)
        except Exception as err:
            self.later(self.model_loaded, None, None, str(err))
            return
        self.later(self.model_loaded, tok, model, "")

    def model_loaded(self, tok, model, error: str) -> None:
        self.loading = False
        if self.runner:
            return
        if error:
            self.model_line.configure(text=error, fg=RED)
            return
        self.tok, self.model = tok, model
        self.tags = [name.strip("<>") for name in tok.tags]
        yours = [t for t in self.tags if t not in PUBLIC]
        self.tag_index = self.tags.index(yours[0]) if yours else 0
        self.style_button.configure(text=f"<{self.tags[self.tag_index]}> ›")
        self.model_line.configure(text=f"step {model.step}, val loss {model.best_val:.2f}, on {model.device}", fg=DIM)
        self.write_button.configure(state="normal")

    def next_style(self) -> None:
        if self.tags:
            self.tag_index = (self.tag_index + 1) % len(self.tags)
            self.style_button.configure(text=f"<{self.tags[self.tag_index]}> ›")

    def start_write(self) -> None:
        if not self.model or str(self.write_button["state"]) == "disabled":
            return
        try:
            length = max(1, min(int(self.length_entry.get()), 2000))
            heat = max(0.05, min(float(self.temp_entry.get()), 3.0))
        except ValueError:
            self.say("tokens and heat must be numbers")
            return
        opening = self.opening.get().strip()
        self.write_button.configure(state="disabled")
        self.write_note.configure(text="writing")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        if opening:
            self.output.insert("end", opening, "yours")
        self.output.configure(state="disabled")
        args = (f"<{self.tags[self.tag_index]}>", opening, length, heat)
        threading.Thread(target=self.write_worker, args=args, daemon=True).start()

    def write_worker(self, tag: str, opening: str, length: int, heat: float) -> None:
        tok, model = self.tok, self.model
        torch = model.torch
        try:
            start = [tok.tags[tag], 10] + (tok.encode(opening) if opening else [])
        except RuntimeError as err:
            self.later(self.write_finished, f"could not tokenize: {err}")
            return
        seed = random.randint(1, 99999)
        gen = torch.Generator(device=model.device)
        gen.manual_seed(seed)
        out: list[int] = []
        shown = ""
        began = time.perf_counter()
        for _ in range(length):
            idx = torch.tensor([(start + out)[-model.cfg.n_ctx:]], dtype=torch.long, device=model.device)
            nxt = model.net.generate(idx, 1, heat, 40, generator=gen)[0, -1].item()
            if nxt in tok.tag_ids:
                break
            out.append(nxt)
            text = tok.decode(out).rstrip("�")
            if len(text) > len(shown):
                self.later(self.append_output, text[len(shown):])
                shown = text
        secs = time.perf_counter() - began
        self.later(self.write_finished, f"{len(out)} tokens, {secs:.1f} s, seed {seed}")

    def append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.configure(state="disabled")
        self.output.see("end")

    def write_finished(self, note: str) -> None:
        self.write_note.configure(text=note)
        self.write_button.configure(state="normal")

    def close(self) -> None:
        if self.runner:
            self.runner.stop()
        self.root.destroy()


def main() -> None:
    os.chdir(pipeline.ROOT)
    if pipeline.ON_WINDOWS:
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
