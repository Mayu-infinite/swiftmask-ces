
"""
SwiftMask captured-photo polygon annotation.

Displays original photographs without aspect-ratio distortion.
Draws polygons at original resolution.
Exports masks at 256x256 using INTER_NEAREST.

Mask convention:
    0   = background
    255 = foreground

Controls:
    Left click  : Add polygon point
    Right click : Undo last point
    Enter       : Finish polygon
    Backspace   : Undo last point
    U           : Undo completed polygon
    R           : Reset current image
    S           : Save and move to next image
    Q / ESC     : Quit
"""

# ============================================================
# 1. ENVIRONMENT
# ============================================================

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ============================================================
# 2. IMPORTS
# ============================================================

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 3. PATHS AND SETTINGS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_DIR = ROOT_DIR / "captured" / "original"
MASK_DIR = ROOT_DIR / "captured" / "masks 256"

IMAGE_SIZE = 256

DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 700
HEADER_HEIGHT = 85

BACKGROUND = 0
FOREGROUND = 255

WINDOW_NAME = "SwiftMask - Natural Aspect Ratio Annotation"


# ============================================================
# 4. ANNOTATOR
# ============================================================

class PolygonAnnotator:

    def __init__(self, image):

        self.image = image

        self.height, self.width = image.shape[:2]

        self.mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        self.polygons = []
        self.current_polygon = []

        # Fit original image inside display canvas.
        # A single scale factor preserves aspect ratio.
        available_width = DISPLAY_WIDTH
        available_height = DISPLAY_HEIGHT - HEADER_HEIGHT

        self.scale = min(
            available_width / self.width,
            available_height / self.height
        )

        self.display_width = max(
            1,
            int(round(self.width * self.scale))
        )

        self.display_height = max(
            1,
            int(round(self.height * self.scale))
        )

        self.offset_x = (
            DISPLAY_WIDTH - self.display_width
        ) // 2

        self.offset_y = (
            HEADER_HEIGHT
            + (available_height - self.display_height) // 2
        )

        self.resized_image = cv2.resize(
            image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Display coordinates -> original image coordinates
    # --------------------------------------------------------

    def to_image_coordinates(self, x, y):

        x = x - self.offset_x
        y = y - self.offset_y

        if not (
            0 <= x < self.display_width
            and 0 <= y < self.display_height
        ):
            return None

        image_x = int(round(x / self.scale))
        image_y = int(round(y / self.scale))

        image_x = np.clip(
            image_x, 0, self.width - 1
        )

        image_y = np.clip(
            image_y, 0, self.height - 1
        )

        return int(image_x), int(image_y)

    # --------------------------------------------------------
    # Mouse events
    # --------------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            point = self.to_image_coordinates(x, y)

            if point is not None:
                self.current_polygon.append(point)

        elif event == cv2.EVENT_RBUTTONDOWN:

            if self.current_polygon:
                self.current_polygon.pop()

    # --------------------------------------------------------
    # Rebuild original-resolution mask
    # --------------------------------------------------------

    def rebuild_mask(self):

        self.mask.fill(BACKGROUND)

        for polygon in self.polygons:

            if len(polygon) >= 3:

                points = np.array(
                    polygon,
                    dtype=np.int32
                )

                cv2.fillPoly(
                    self.mask,
                    [points],
                    FOREGROUND
                )

    # --------------------------------------------------------
    # Finish polygon
    # --------------------------------------------------------

    def finish_polygon(self):

        if len(self.current_polygon) < 3:

            print("A polygon needs at least 3 points.")
            return

        self.polygons.append(
            self.current_polygon.copy()
        )

        self.current_polygon.clear()

        self.rebuild_mask()

        print(
            f"Polygon added. "
            f"Total polygons: {len(self.polygons)}"
        )

    # --------------------------------------------------------
    # Undo polygon
    # --------------------------------------------------------

    def undo_polygon(self):

        if not self.polygons:
            print("No completed polygon to undo.")
            return

        self.polygons.pop()
        self.rebuild_mask()

        print("Last polygon removed.")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.polygons.clear()
        self.current_polygon.clear()
        self.mask.fill(BACKGROUND)

        print("Annotations cleared.")

    # --------------------------------------------------------
    # Render preview without distorting original image
    # --------------------------------------------------------

    def render(self, image_name, index, total):

        canvas = np.zeros(
            (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
            dtype=np.uint8
        )

        canvas[:] = (35, 35, 35)

        # Display image at its natural aspect ratio.
        y1 = self.offset_y
        y2 = y1 + self.display_height

        x1 = self.offset_x
        x2 = x1 + self.display_width

        canvas[y1:y2, x1:x2] = self.resized_image

        # Preview mask at display dimensions.
        preview_mask = cv2.resize(
            self.mask,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Include current polygon in temporary preview.
        if len(self.current_polygon) >= 3:

            temp_mask = self.mask.copy()

            points = np.array(
                self.current_polygon,
                dtype=np.int32
            )

            cv2.fillPoly(
                temp_mask,
                [points],
                FOREGROUND
            )

            preview_mask = cv2.resize(
                temp_mask,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_NEAREST
            )

        region = canvas[y1:y2, x1:x2]

        foreground = preview_mask > 0

        green = np.zeros_like(region)
        green[:] = (0, 255, 0)

        region[foreground] = (
            region[foreground].astype(np.float32) * 0.45
            + np.array([0, 255, 0], dtype=np.float32) * 0.55
        ).clip(0, 255).astype(np.uint8)

        # Draw completed polygons.
        for polygon in self.polygons:

            points = np.array(
                [
                    (
                        int(round(x * self.scale)),
                        int(round(y * self.scale))
                    )
                    for x, y in polygon
                ],
                dtype=np.int32
            )

            points[:, 0] += self.offset_x
            points[:, 1] += self.offset_y

            cv2.polylines(
                canvas,
                [points],
                True,
                (0, 255, 255),
                2
            )

        # Draw current polygon.
        if self.current_polygon:

            points = np.array(
                [
                    (
                        int(round(x * self.scale)) + self.offset_x,
                        int(round(y * self.scale)) + self.offset_y
                    )
                    for x, y in self.current_polygon
                ],
                dtype=np.int32
            )

            if len(points) >= 2:

                cv2.polylines(
                    canvas,
                    [points],
                    False,
                    (0, 0, 255),
                    2
                )

            for point in points:

                cv2.circle(
                    canvas,
                    tuple(point),
                    4,
                    (0, 0, 255),
                    -1
                )

        # Header.
        lines = [
            f"Image {index}/{total}: {image_name}",
            f"Original resolution: {self.width} x {self.height}",
            f"Completed polygons: {len(self.polygons)}",
            "Left click: add | Right click: undo point | ENTER: finish",
            "U: undo polygon | R: reset | S: save | Q: quit",
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                canvas,
                line,
                (10, 18 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas


# ============================================================
# 5. EXPORT MASK TO 256 x 256
# ============================================================

def export_mask(original_mask, output_path):

    if not np.any(original_mask == FOREGROUND):

        raise ValueError(
            "Empty mask. Draw at least one polygon."
        )

    # Apply the same direct-resize geometry as the
    # assignment's 256x256 image preprocessing.
    mask_256 = cv2.resize(
        original_mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Ensure strictly binary output.
    mask_256 = np.where(
        mask_256 >= 128,
        FOREGROUND,
        BACKGROUND
    ).astype(np.uint8)

    if not np.any(mask_256 == FOREGROUND):

        raise ValueError(
            "Foreground disappeared during mask resizing. "
            "Please draw a larger or more detailed polygon."
        )

    success = cv2.imwrite(
        str(output_path),
        mask_256
    )

    if not success:

        raise RuntimeError(
            f"Could not save mask: {output_path}"
        )

    coverage = (
        np.count_nonzero(mask_256 == FOREGROUND)
        / mask_256.size
    ) * 100.0

    print(f"Saved: {output_path}")
    print(f"Output shape: {mask_256.shape}")
    print(f"Foreground coverage: {coverage:.2f}%")


# ============================================================
# 6. ANNOTATION SESSION
# ============================================================

def annotate_dataset(overwrite=False):

    if not ORIGINAL_DIR.is_dir():

        raise FileNotFoundError(
            f"Original directory not found: {ORIGINAL_DIR}"
        )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        ORIGINAL_DIR.glob("*.png"),
        key=lambda p: (
            int(p.stem[3:])
            if p.stem.startswith("img")
            and p.stem[3:].isdigit()
            else p.name
        )
    )

    if not image_paths:

        raise RuntimeError(
            "No original PNG images found."
        )

    print("=" * 65)
    print("SWIFTMASK - NATURAL ASPECT RATIO ANNOTATION")
    print("=" * 65)

    print(f"Original images: {ORIGINAL_DIR}")
    print(f"Output masks: {MASK_DIR}")
    print(f"Number of images: {len(image_paths)}")
    print("Annotation is performed at original resolution.")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        output_path = MASK_DIR / image_path.name

        # Skip previously completed annotations.
        if output_path.exists() and not overwrite:

            existing = cv2.imread(
                str(output_path),
                cv2.IMREAD_GRAYSCALE
            )

            if (
                existing is not None
                and existing.shape == (IMAGE_SIZE, IMAGE_SIZE)
                and np.any(existing > 0)
            ):

                print(
                    f"Skipping {image_path.name}: "
                    "non-empty mask already exists."
                )

                continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"Could not read {image_path.name}. Skipping."
            )

            continue

        annotator = PolygonAnnotator(image)

        cv2.setMouseCallback(
            WINDOW_NAME,
            annotator.mouse_callback
        )

        print("\n" + "-" * 65)
        print(f"Image {index}/{len(image_paths)}: {image_path.name}")
        print(f"Original size: {image.shape[1]} x {image.shape[0]}")

        while True:

            preview = annotator.render(
                image_path.name,
                index,
                len(image_paths)
            )

            cv2.imshow(
                WINDOW_NAME,
                preview
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (10, 13):

                annotator.finish_polygon()

            elif key in (8, 127):

                if annotator.current_polygon:
                    annotator.current_polygon.pop()

            elif key in (ord("u"), ord("U")):

                annotator.undo_polygon()

            elif key in (ord("r"), ord("R")):

                annotator.reset()

            elif key in (ord("s"), ord("S")):

                if annotator.current_polygon:

                    print(
                        "Finish the current polygon with ENTER first."
                    )

                    continue

                try:

                    export_mask(
                        annotator.mask,
                        output_path
                    )

                except ValueError as error:

                    print(error)
                    continue

                break

            elif key in (ord("q"), ord("Q"), 27):

                print("Annotation session stopped.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print("ANNOTATION SESSION COMPLETED")
    print("=" * 65)
    print(f"Output directory: {MASK_DIR}")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Annotate original captured photos "
            "without aspect-ratio distortion."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing non-empty masks."
    )

    args = parser.parse_args()

    annotate_dataset(
        overwrite=args.overwrite
    )

"""
SwiftMask captured-photo polygon annotation.

Displays original photographs without aspect-ratio distortion.
Draws polygons at original resolution.
Exports masks at 256x256 using INTER_NEAREST.

Mask convention:
    0   = background
    255 = foreground

Controls:
    Left click  : Add polygon point
    Right click : Undo last point
    Enter       : Finish polygon
    Backspace   : Undo last point
    U           : Undo completed polygon
    R           : Reset current image
    S           : Save and move to next image
    Q / ESC     : Quit
"""

# ============================================================
# 1. ENVIRONMENT
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ============================================================
# 2. IMPORTS
# ============================================================

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 3. PATHS AND SETTINGS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_DIR = ROOT_DIR / "captured" / "original"
MASK_DIR = ROOT_DIR / "captured" / "masks 256"

IMAGE_SIZE = 256

DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 700
HEADER_HEIGHT = 85

BACKGROUND = 0
FOREGROUND = 255

WINDOW_NAME = "SwiftMask - Natural Aspect Ratio Annotation"


# ============================================================
# 4. ANNOTATOR
# ============================================================

class PolygonAnnotator:

    def __init__(self, image):

        self.image = image

        self.height, self.width = image.shape[:2]

        self.mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        self.polygons = []
        self.current_polygon = []

        # Fit original image inside display canvas.
        # A single scale factor preserves aspect ratio.
        available_width = DISPLAY_WIDTH
        available_height = DISPLAY_HEIGHT - HEADER_HEIGHT

        self.scale = min(
            available_width / self.width,
            available_height / self.height
        )

        self.display_width = max(
            1,
            int(round(self.width * self.scale))
        )

        self.display_height = max(
            1,
            int(round(self.height * self.scale))
        )

        self.offset_x = (
            DISPLAY_WIDTH - self.display_width
        ) // 2

        self.offset_y = (
            HEADER_HEIGHT
            + (available_height - self.display_height) // 2
        )

        self.resized_image = cv2.resize(
            image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Display coordinates -> original image coordinates
    # --------------------------------------------------------

    def to_image_coordinates(self, x, y):

        x = x - self.offset_x
        y = y - self.offset_y

        if not (
            0 <= x < self.display_width
            and 0 <= y < self.display_height
        ):
            return None

        image_x = int(round(x / self.scale))
        image_y = int(round(y / self.scale))

        image_x = np.clip(
            image_x, 0, self.width - 1
        )

        image_y = np.clip(
            image_y, 0, self.height - 1
        )

        return int(image_x), int(image_y)

    # --------------------------------------------------------
    # Mouse events
    # --------------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            point = self.to_image_coordinates(x, y)

            if point is not None:
                self.current_polygon.append(point)

        elif event == cv2.EVENT_RBUTTONDOWN:

            if self.current_polygon:
                self.current_polygon.pop()

    # --------------------------------------------------------
    # Rebuild original-resolution mask
    # --------------------------------------------------------

    def rebuild_mask(self):

        self.mask.fill(BACKGROUND)

        for polygon in self.polygons:

            if len(polygon) >= 3:

                points = np.array(
                    polygon,
                    dtype=np.int32
                )

                cv2.fillPoly(
                    self.mask,
                    [points],
                    FOREGROUND
                )

    # --------------------------------------------------------
    # Finish polygon
    # --------------------------------------------------------

    def finish_polygon(self):

        if len(self.current_polygon) < 3:

            print("A polygon needs at least 3 points.")
            return

        self.polygons.append(
            self.current_polygon.copy()
        )

        self.current_polygon.clear()

        self.rebuild_mask()

        print(
            f"Polygon added. "
            f"Total polygons: {len(self.polygons)}"
        )

    # --------------------------------------------------------
    # Undo polygon
    # --------------------------------------------------------

    def undo_polygon(self):

        if not self.polygons:
            print("No completed polygon to undo.")
            return

        self.polygons.pop()
        self.rebuild_mask()

        print("Last polygon removed.")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.polygons.clear()
        self.current_polygon.clear()
        self.mask.fill(BACKGROUND)

        print("Annotations cleared.")

    # --------------------------------------------------------
    # Render preview without distorting original image
    # --------------------------------------------------------

    def render(self, image_name, index, total):

        canvas = np.zeros(
            (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
            dtype=np.uint8
        )

        canvas[:] = (35, 35, 35)

        # Display image at its natural aspect ratio.
        y1 = self.offset_y
        y2 = y1 + self.display_height

        x1 = self.offset_x
        x2 = x1 + self.display_width

        canvas[y1:y2, x1:x2] = self.resized_image

        # Preview mask at display dimensions.
        preview_mask = cv2.resize(
            self.mask,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Include current polygon in temporary preview.
        if len(self.current_polygon) >= 3:

            temp_mask = self.mask.copy()

            points = np.array(
                self.current_polygon,
                dtype=np.int32
            )

            cv2.fillPoly(
                temp_mask,
                [points],
                FOREGROUND
            )

            preview_mask = cv2.resize(
                temp_mask,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_NEAREST
            )

        region = canvas[y1:y2, x1:x2]

        foreground = preview_mask > 0

        green = np.zeros_like(region)
        green[:] = (0, 255, 0)

        region[foreground] = cv2.addWeighted(
            region[foreground],
            0.45,
            green[foreground],
            0.55,
            0
        )

        # Draw completed polygons.
        for polygon in self.polygons:

            points = np.array(
                [
                    (
                        int(round(x * self.scale)),
                        int(round(y * self.scale))
                    )
                    for x, y in polygon
                ],
                dtype=np.int32
            )

            points[:, 0] += self.offset_x
            points[:, 1] += self.offset_y

            cv2.polylines(
                canvas,
                [points],
                True,
                (0, 255, 255),
                2
            )

        # Draw current polygon.
        if self.current_polygon:

            points = np.array(
                [
                    (
                        int(round(x * self.scale)) + self.offset_x,
                        int(round(y * self.scale)) + self.offset_y
                    )
                    for x, y in self.current_polygon
                ],
                dtype=np.int32
            )

            if len(points) >= 2:

                cv2.polylines(
                    canvas,
                    [points],
                    False,
                    (0, 0, 255),
                    2
                )

            for point in points:

                cv2.circle(
                    canvas,
                    tuple(point),
                    4,
                    (0, 0, 255),
                    -1
                )

        # Header.
        lines = [
            f"Image {index}/{total}: {image_name}",
            f"Original resolution: {self.width} x {self.height}",
            f"Completed polygons: {len(self.polygons)}",
            "Left click: add | Right click: undo point | ENTER: finish",
            "U: undo polygon | R: reset | S: save | Q: quit",
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                canvas,
                line,
                (10, 18 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas


# ============================================================
# 5. EXPORT MASK TO 256 x 256
# ============================================================

def export_mask(original_mask, output_path):

    if not np.any(original_mask == FOREGROUND):

        raise ValueError(
            "Empty mask. Draw at least one polygon."
        )

    # Apply the same direct-resize geometry as the
    # assignment's 256x256 image preprocessing.
    mask_256 = cv2.resize(
        original_mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Ensure strictly binary output.
    mask_256 = np.where(
        mask_256 >= 128,
        FOREGROUND,
        BACKGROUND
    ).astype(np.uint8)

    if not np.any(mask_256 == FOREGROUND):

        raise ValueError(
            "Foreground disappeared during mask resizing. "
            "Please draw a larger or more detailed polygon."
        )

    success = cv2.imwrite(
        str(output_path),
        mask_256
    )

    if not success:

        raise RuntimeError(
            f"Could not save mask: {output_path}"
        )

    coverage = (
        np.count_nonzero(mask_256 == FOREGROUND)
        / mask_256.size
    ) * 100.0

    print(f"Saved: {output_path}")
    print(f"Output shape: {mask_256.shape}")
    print(f"Foreground coverage: {coverage:.2f}%")


# ============================================================
# 6. ANNOTATION SESSION
# ============================================================

def annotate_dataset(overwrite=False):

    if not ORIGINAL_DIR.is_dir():

        raise FileNotFoundError(
            f"Original directory not found: {ORIGINAL_DIR}"
        )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        ORIGINAL_DIR.glob("*.png"),
        key=lambda p: (
            int(p.stem[3:])
            if p.stem.startswith("img")
            and p.stem[3:].isdigit()
            else p.name
        )
    )

    if not image_paths:

        raise RuntimeError(
            "No original PNG images found."
        )

    print("=" * 65)
    print("SWIFTMASK - NATURAL ASPECT RATIO ANNOTATION")
    print("=" * 65)

    print(f"Original images: {ORIGINAL_DIR}")
    print(f"Output masks: {MASK_DIR}")
    print(f"Number of images: {len(image_paths)}")
    print("Annotation is performed at original resolution.")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        output_path = MASK_DIR / image_path.name

        # Skip previously completed annotations.
        if output_path.exists() and not overwrite:

            existing = cv2.imread(
                str(output_path),
                cv2.IMREAD_GRAYSCALE
            )

            if (
                existing is not None
                and existing.shape == (IMAGE_SIZE, IMAGE_SIZE)
                and np.any(existing > 0)
            ):

                print(
                    f"Skipping {image_path.name}: "
                    "non-empty mask already exists."
                )

                continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"Could not read {image_path.name}. Skipping."
            )

            continue

        annotator = PolygonAnnotator(image)

        cv2.setMouseCallback(
            WINDOW_NAME,
            annotator.mouse_callback
        )

        print("\n" + "-" * 65)
        print(f"Image {index}/{len(image_paths)}: {image_path.name}")
        print(f"Original size: {image.shape[1]} x {image.shape[0]}")

        while True:

            preview = annotator.render(
                image_path.name,
                index,
                len(image_paths)
            )

            cv2.imshow(
                WINDOW_NAME,
                preview
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (10, 13):

                annotator.finish_polygon()

            elif key in (8, 127):

                if annotator.current_polygon:
                    annotator.current_polygon.pop()

            elif key in (ord("u"), ord("U")):

                annotator.undo_polygon()

            elif key in (ord("r"), ord("R")):

                annotator.reset()

            elif key in (ord("s"), ord("S")):

                if annotator.current_polygon:

                    print(
                        "Finish the current polygon with ENTER first."
                    )

                    continue

                try:

                    export_mask(
                        annotator.mask,
                        output_path
                    )

                except ValueError as error:

                    print(error)
                    continue

                break

            elif key in (ord("q"), ord("Q"), 27):

                print("Annotation session stopped.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print("ANNOTATION SESSION COMPLETED")
    print("=" * 65)
    print(f"Output directory: {MASK_DIR}")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Annotate original captured photos "
            "without aspect-ratio distortion."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing non-empty masks."
    )

    args = parser.parse_args()

    annotate_dataset(
        overwrite=args.overwrite
    )
