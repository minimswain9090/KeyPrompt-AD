"""Minimal click-to-annotate tool for the reference shots.

Usage is intentionally low-tech: matplotlib window, left-click to drop a point,
number keys to switch the active component class, 'g' to start a new group,
'u' to undo, 'w' to write and advance.

Annotating four images per category takes a few minutes, which is the whole
point of the method: no bounding-box campaign, no labelled defects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from .schema import ImageAnnotation, Keypoint

_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


class KeypointAnnotator:
    def __init__(self, classes: List[str], out_dir: str | Path):
        if not classes:
            raise ValueError("at least one component class is required")
        self.classes = classes
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def annotate(
        self,
        image_path: str | Path,
        image_uid: str,
        category: str,
        use_groups: bool = True,
    ) -> Optional[ImageAnnotation]:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(image_path))
        h, w = img.shape[0], img.shape[1]
        ann = ImageAnnotation(image_uid=image_uid, category=category, width=w, height=h)

        state = {"cls_idx": 0, "group": 0, "done": False, "saved": False}
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(img)
        ax.set_axis_off()
        artists: List = []

        def title() -> None:
            ax.set_title(
                f"[{Path(image_path).name}]  class={self.classes[state['cls_idx']]}  "
                f"group={state['group'] if use_groups else '-'}  n={len(ann.keypoints)}\n"
                "click=add   1..9=class   g=new group   u=undo   w=write+close   q=skip",
                fontsize=9,
            )
            fig.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not ax or event.xdata is None:
                return
            cls = self.classes[state["cls_idx"]]
            kp = Keypoint(
                cls=cls,
                x=round(float(event.xdata) / w, 5),
                y=round(float(event.ydata) / h, 5),
                group=f"g{state['group']}" if use_groups else None,
            )
            ann.keypoints.append(kp)
            color = _PALETTE[state["cls_idx"] % len(_PALETTE)]
            dot = ax.plot(event.xdata, event.ydata, "o", ms=9, mfc=color, mec="white")[0]
            txt = ax.annotate(
                f"{cls[:3]}{state['group'] if use_groups else ''}",
                (event.xdata, event.ydata),
                textcoords="offset points",
                xytext=(8, 6),
                color=color,
                fontsize=8,
            )
            artists.append((dot, txt))
            title()

        def on_key(event) -> None:
            if event.key in [str(i) for i in range(1, 10)]:
                i = int(event.key) - 1
                if i < len(self.classes):
                    state["cls_idx"] = i
            elif event.key == "g":
                state["group"] += 1
            elif event.key == "u" and ann.keypoints:
                ann.keypoints.pop()
                dot, txt = artists.pop()
                dot.remove()
                txt.remove()
            elif event.key == "w":
                state["saved"] = True
                plt.close(fig)
                return
            elif event.key == "q":
                plt.close(fig)
                return
            title()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        title()
        plt.show()

        if not state["saved"] or not ann.keypoints:
            return None
        out = self.out_dir / f"{Path(image_path).stem}.json"
        ann.to_json(out)
        print(f"wrote {out}  ({len(ann.keypoints)} keypoints)")
        return ann
