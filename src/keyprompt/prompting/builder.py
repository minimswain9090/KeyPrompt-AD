"""Prompt assembly.

The prompt has three parts, in this order:

1. A task frame describing the inspection job in general terms.
2. The layout specification, generated automatically from the normality graph.
   This is the piece that makes the method transferable: change the reference
   annotations and the specification rewrites itself.
3. The few-shot block, pairing each reference image with its exact coordinate
   annotation, followed by the query image.

Nothing here is hand-tuned per category beyond a short normality statement,
which keeps the comparison across the five LOCO categories honest.
"""

from __future__ import annotations

from typing import List, Tuple

from ..annotate.schema import ImageAnnotation
from ..config import CategoryConfig
from ..prior.graph import NormalityGraph
from .schema import JSON_SCHEMA_HINT

TASK_FRAME = """
You are inspecting images of a manufactured assembly photographed in a fixed
setup. Your job is to check that every required component is present and sitting
where it belongs, and to report the result as structured data.

Work from geometry, not appearance. A part is defective when a component is
absent, duplicated, or sitting outside its tolerance, or when the spacing
between components departs from the reference layout. A part is NOT defective
because of scratches, stains, smudges, reflections, dust, or minor differences
in lighting or shade. Ignore that kind of surface variation entirely.

All coordinates are normalised: x runs from 0 at the left edge to 1 at the
right edge, y runs from 0 at the top edge to 1 at the bottom edge.
""".strip()


def build_context_prompt(category: CategoryConfig, graph: NormalityGraph) -> str:
    parts: List[str] = [TASK_FRAME, ""]

    parts.append(f"COMPONENT TYPES for '{category.name}':")
    classes = category.component_classes or sorted(graph.classes)
    for c in classes:
        parts.append(f"- {c}")
    parts.append("")

    if category.normality_statement:
        parts.append("WHAT A CORRECT PART LOOKS LIKE:")
        parts.append(category.normality_statement.strip())
        parts.append("")

    if category.grouping_description:
        parts.append("HOW COMPONENTS ARE GROUPED:")
        parts.append(category.grouping_description.strip())
        parts.append("")

    parts.append(
        f"REFERENCE LAYOUT, derived from {graph.n_shots} correct examples. "
        "Treat the tolerances as guidance rather than hard limits:"
    )
    parts.append(graph.describe())
    parts.append("")

    if category.ignored_variation:
        parts.append("DO NOT report any of the following as defects:")
        for v in category.ignored_variation:
            parts.append(f"- {v}")
        parts.append("")

    parts.append(
        "A part is NOT OK if any of the following hold: a component listed above "
        "is absent; an extra component appears where the layout expects none; a "
        "component sits well outside its tolerance; or the spacing between two "
        "components departs clearly from the characteristic value."
    )
    return "\n".join(parts)


def build_shot_block(index: int, annotation: ImageAnnotation) -> str:
    lines = [f"--- Reference example {index} (correct part) ---"]
    by_cls: dict[str, List[Tuple[float, float]]] = {}
    for k in annotation.keypoints:
        by_cls.setdefault(k.cls, []).append((k.x, k.y))
    for cls in sorted(by_cls):
        coords = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in by_cls[cls])
        lines.append(f"{cls} ({len(by_cls[cls])}): {coords}")
    lines.append(f"verdict: OK")
    lines.append(f"--- end reference example {index} ---")
    return "\n".join(lines)


def build_query_prompt() -> str:
    return f"""
Now inspect the final image.

Report every component you can locate, using the same class names and the same
normalised coordinate convention as the reference examples. If a component the
layout requires is absent, list it under "missing" with the position where it
should have been.

Set "confidence" to your probability that the part is defective: near 0 when
the part is clearly correct, near 1 when a component is clearly absent or
badly out of place.

Reply with a single JSON object and nothing else, in exactly this shape:

{JSON_SCHEMA_HINT}
""".strip()
