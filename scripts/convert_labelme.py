
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "captured" / "images 256"
JSON_DIR = ROOT / "captured" / "labelme_json"
MASK_DIR = ROOT / "captured" / "masks 256"


def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in {JSON_DIR}")

    for json_file in json_files:
        image_path = IMAGE_DIR / f"{json_file.stem}.png"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not read {image_path}")

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        annotation = json.loads(json_file.read_text())

        object_count = 0

        for shape in annotation["shapes"]:
            if shape["label"] != "object":
                continue

            if shape["shape_type"] != "polygon":
                continue

            points = np.array(
                shape["points"], dtype=np.float32
            )

            points = np.round(points).astype(np.int32)

            if len(points) < 3:
                continue

            cv2.fillPoly(mask, [points], 255)
            object_count += 1

        if object_count == 0:
            raise RuntimeError(
                f"No object polygons found in {json_file.name}"
            )

        output_path = MASK_DIR / f"{json_file.stem}.png"

        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Could not save {output_path}")

        print(
            f"{json_file.stem}: "
            f"{object_count} polygons, "
            f"foreground pixels = {np.count_nonzero(mask)}"
        )

    print("\nAll masks exported successfully.")


if __name__ == "__main__":
    main()
