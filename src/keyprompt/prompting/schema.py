"""Response schema for the VLM.

Free-form text answers are unusable for benchmarking: parsing them is brittle
and the failure modes are silent. Every provider is therefore asked for JSON
matching this schema, and responses that do not parse are recorded as explicit
failures rather than quietly dropped.

Models drift from the requested shape in predictable ways -- code fences, a
0-10 confidence scale, pixel coordinates, "classification" instead of
"verdict". The parser absorbs those variations rather than discarding an
otherwise usable answer, because silently dropping responses would bias the
benchmark toward whichever model happens to be the most obedient formatter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field


@dataclass
class DetectedComponent:
    """One component the model claims to have located."""

    cls: str                      # class name, exactly as given in the layout
    x: float                      # normalised horizontal position in [0,1]
    y: float                      # normalised vertical position in [0,1]
    slot: Optional[str] = None    # matching slot id, when the model identifies one


@dataclass
class InspectionResult:
    verdict: str = "OK"                      # OK or NOT OK
    confidence: float = 0.5                  # 0 = certainly OK, 1 = certainly defective
    detected: List[DetectedComponent] = field(default_factory=list)
    missing: List[DetectedComponent] = field(default_factory=list)
    reasoning: str = ""


JSON_SCHEMA_HINT = """
{
  "verdict": "OK" | "NOT OK",
  "confidence": <float between 0 and 1>,
  "detected": [{"cls": "<class>", "x": <float>, "y": <float>, "slot": "<class>[i]" }],
  "missing":  [{"cls": "<class>", "x": <float>, "y": <float>, "slot": "<class>[i]" }],
  "reasoning": "<one or two sentences>"
}
""".strip()


def parse_response(text: str) -> InspectionResult:
    """Recover an InspectionResult from a model response.

    Providers differ in how faithfully they honour a JSON instruction, so the
    parser tolerates code fences and leading prose before giving up.
    """
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in response: {cleaned[:200]!r}")
        payload = json.loads(cleaned[start : end + 1])

    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    d = _coerce(payload)
    return InspectionResult(
        verdict=d["verdict"],
        confidence=d["confidence"],
        detected=[DetectedComponent(**c) for c in d["detected"]],
        missing=[DetectedComponent(**c) for c in d["missing"]],
        reasoning=d["reasoning"],
    )


def _coerce(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the small variations models produce around field naming."""
    out = dict(d)
    for alias in ("classification", "label", "result", "status"):
        if "verdict" not in out and alias in out:
            out["verdict"] = out[alias]
    out.setdefault("verdict", "OK")
    out["verdict"] = "NOT OK" if "NOT" in str(out["verdict"]).upper() else "OK"

    conf = out.get("confidence", out.get("anomaly_score", 0.5))
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.5
    # Some models answer on a 0-10 scale despite the instruction.
    out["confidence"] = conf / 10.0 if conf > 1.0 else max(0.0, conf)

    for key in ("detected", "missing"):
        items = out.get(key) or []
        fixed: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cls = it.get("cls") or it.get("class") or it.get("type") or "unknown"
            if "x" in it and "y" in it:
                x, y = it["x"], it["y"]
            elif isinstance(it.get("position"), (list, tuple)) and len(it["position"]) >= 2:
                x, y = it["position"][0], it["position"][1]
            else:
                continue
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue
            # Guard against models that answer in pixels or on a 0-1000 grid.
            if x > 1.5 or y > 1.5:
                scale = 1000.0 if max(x, y) <= 1000.0 else max(x, y)
                x, y = x / scale, y / scale
            fixed.append({"cls": str(cls), "x": x, "y": y, "slot": it.get("slot")})
        out[key] = fixed

    out["reasoning"] = str(out.get("reasoning") or out.get("defect_description") or "")[:600]
    return out
