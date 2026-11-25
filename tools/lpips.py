#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

#! The script has the name of the lpips package, so its own folder must leave the import path before the import.
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]

import lpips
import numpy
import torch
from PIL import Image


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    pixels = numpy.asarray(Image.open(path).convert("RGB"), dtype=numpy.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adds LPIPS (VGG) to the report.json of a DiffRenderer comparison.")
    parser.add_argument("folder", type=Path, help="the output folder of the comparison")
    arguments = parser.parse_args()
    report_path = arguments.folder / "report.json"
    report = json.loads(report_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    references = arguments.folder / report["references"]
    with torch.no_grad():
        for method in report["methods"]:
            renders = arguments.folder / method["renders"]
            values = []
            for view in method["views"]:
                render = load_image(renders / view["file"], device)
                reference = load_image(references / view["file"], device)
                view["lpips"] = model(render, reference, normalize=True).item()
                values.append(view["lpips"])
            method["lpips"] = sum(values) / len(values) if values else None
            print(f"{method['id']}: LPIPS {method['lpips']} over {len(values)} test views")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
