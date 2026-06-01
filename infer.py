"""
Headline inference interface for the UnifiedModel.

Change MODEL_PATH to point to a different trained checkpoint.
"""

import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/unified/seed42-5ep/best.pt"
CONFIG_PATH = "models/unified/config.yaml"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from models.unified.predict import load_unified_predictor, predict_all

console = Console()

# Emotion color palette — warm/cool split
_EMOTION_COLORS = {
    "anger":        "bold red",
    "anticipation": "bold yellow",
    "disgust":      "dark_orange",
    "fear":         "magenta",
    "joy":          "bold green",
    "love":         "bright_magenta",
    "optimism":     "bright_green",
    "pessimism":    "grey58",
    "sadness":      "blue",
    "surprise":     "cyan",
    "trust":        "bright_cyan",
}

_EPISTEMIC_COLORS = {
    "asserted":    "bold green",
    "hedged":      "bold yellow",
    "speculative": "bold red",
}

_BIAS_COLORS = {
    "not biased": "bold green",
    "biased":     "bold red",
}

_BAR_WIDTH = 20


def _bar(score: float, color: str) -> Text:
    """Render a simple ASCII progress bar."""
    filled = round(score * _BAR_WIDTH)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    t = Text()
    t.append(f"[{bar}]", style=color)
    t.append(f"  {score:.0%}", style="bold white")
    return t


def render_epistemic(result: dict) -> Panel:
    label = result["label_name"]
    color = _EPISTEMIC_COLORS.get(label, "white")

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column()
    table.add_column()

    label_names = ["asserted", "hedged", "speculative"]
    for name, prob in zip(label_names, result["sent_probs"]):
        c = _EPISTEMIC_COLORS[name]
        marker = "◀" if name == label else " "
        table.add_row(
            Text(name.capitalize(), style=c),
            _bar(prob, c),
            Text(marker, style="bold white"),
        )

    uncertainty_bar = _bar(result["uncertainty_score"], "yellow")

    content = Text()
    content.append("Prediction: ", style="dim")
    content.append(label.upper(), style=color)
    content.append("\n\n")
    content.append_text(Text.from_markup("[dim]Class probabilities[/dim]"))
    content.append("\n")

    from rich.console import Group
    body = Group(content, table, Text("\n"), Text.assemble(
        Text("Uncertainty score  ", style="dim"),
        uncertainty_bar,
    ))

    return Panel(body, title="[bold white]Epistemic Certainty[/bold white]",
                 border_style="bright_blue", box=box.ROUNDED)


def render_bias(result: dict) -> Panel:
    prediction = result["prediction"]
    color = _BIAS_COLORS.get(prediction, "white")

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column()
    table.add_column(no_wrap=True)
    for label, prob in result["probabilities"].items():
        c = _BIAS_COLORS.get(label, "white")
        is_predicted = label == prediction
        label_style = c if is_predicted else "dim"
        marker = Text("◀ PREDICTED", style=f"bold {c}") if is_predicted else Text("○", style="dim")
        table.add_row(
            Text(label.capitalize(), style=label_style),
            _bar(prob, c if is_predicted else "grey30"),
            marker,
        )

    content = Text()
    content.append_text(Text.from_markup("[dim]Class probabilities[/dim]"))
    content.append("\n")

    from rich.console import Group
    body = Group(content, table)

    return Panel(body, title="[bold white]Political Bias[/bold white]",
                 border_style="bright_magenta", box=box.ROUNDED)


def render_emotion(result: dict) -> Panel:
    scores = result["scores"]
    active = [lbl for lbl in scores if result[lbl] == 1]

    # Sort by score descending
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    table.add_column()
    table.add_column(no_wrap=True)

    for label, score in sorted_emotions:
        c = _EMOTION_COLORS[label]
        is_active = label in active
        label_style = c if is_active else "dim"
        marker = Text("● DETECTED", style=f"bold {c}") if is_active else Text("○", style="dim")
        table.add_row(
            Text(label.capitalize(), style=label_style),
            _bar(score, c if is_active else "grey30"),
            marker,
        )

    if active:
        chips = Text()
        for i, lbl in enumerate(active):
            c = _EMOTION_COLORS[lbl]
            chips.append(f" {lbl.upper()} ", style=f"bold {c} on grey15")
            if i < len(active) - 1:
                chips.append("  ")
        detected_line = Text.assemble(Text("Detected:  ", style="dim"), chips)
    else:
        detected_line = Text("Detected:  none", style="dim italic")

    content = Text()
    content.append_text(detected_line)
    content.append("\n\n")
    content.append_text(Text.from_markup("[dim]All emotion scores[/dim]"))
    content.append("\n")

    from rich.console import Group
    body = Group(content, table)

    return Panel(body, title="[bold white]Emotion Detection[/bold white]",
                 border_style="bright_yellow", box=box.ROUNDED)


def run_inference(headline: str, predictor) -> None:
    console.print()
    console.rule("[bold white]Analysis")

    results = predict_all(predictor, [headline])
    r = results[0]

    console.print(Panel(
        Text(headline, style="bold white"),
        title="[dim]Headline[/dim]",
        border_style="white",
        box=box.SIMPLE_HEAD,
    ))

    console.print(render_epistemic(r["epistemic"]))
    console.print(render_bias(r["bias"]))
    console.print(render_emotion(r["emotion"]))


def main() -> None:
    console.print(Panel(
        "[bold cyan]Headline Sentiment Analyzer[/bold cyan]\n"
        "[dim]Unified model — epistemic certainty · political bias · emotion[/dim]",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    console.print(f"\n[dim]Loading model from[/dim] [cyan]{MODEL_PATH}[/cyan]", end="")
    try:
        predictor = load_unified_predictor(
            checkpoint=MODEL_PATH,
            config=CONFIG_PATH,
        )
    except Exception as exc:
        console.print(f"\n[bold red]Error loading model:[/bold red] {exc}")
        sys.exit(1)
    console.print("  [green]ready[/green]\n")

    while True:
        try:
            headline = console.input("[bold cyan]Enter headline[/bold cyan] [dim](or Ctrl-C to quit)[/dim]: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not headline:
            continue

        try:
            run_inference(headline, predictor)
        except Exception as exc:
            console.print(f"[bold red]Inference error:[/bold red] {exc}")

        console.print()


if __name__ == "__main__":
    main()
