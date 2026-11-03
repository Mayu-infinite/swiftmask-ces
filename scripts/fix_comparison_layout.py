from pathlib import Path


def main() -> None:
    """
    Change captured comparison visualizations from a 2x2 grid
    into one horizontal row containing four panels.

    The original evaluation script is backed up before modification.
    """

    script_path = Path("scripts/evaluate_captured.py")

    if not script_path.is_file():
        raise FileNotFoundError(
            "Run this script from the swiftmask-ces repository root. "
            "scripts/evaluate_captured.py was not found."
        )

    source = script_path.read_text(encoding="utf-8")

    replacements = {
        # Update the output description.
        "comparisons/        2x2 visual comparison images":
            "comparisons/        1x4 horizontal comparison strips",

        # Update the function documentation.
        '    """Create a 2x2 comparison panel."""':
            (
                '    """Create one horizontal row: original, '
                'ground truth, prediction, overlay."""'
            ),

        # Replace the 2x2 grid arrangement with one horizontal row.
        """    top = cv2.hconcat(panels[:2])
    bottom = cv2.hconcat(panels[2:])

    return cv2.vconcat([top, bottom])""":
            """    return cv2.hconcat(panels)""",

        # Update the evaluation-loop comment.
        "        # Save 2x2 comparison.":
            "        # Save a 1x4 horizontal comparison strip.",
    }

    updated = source

    # Verify all expected source sections before writing anything.
    for old, new in replacements.items():
        count = updated.count(old)

        if count != 1:
            raise RuntimeError(
                f"Expected exactly one occurrence of:\n{old!r}\n"
                f"Found: {count}\n\n"
                "No changes were written. Inspect the source file."
            )

        updated = updated.replace(old, new, 1)

    # Do not overwrite an existing backup.
    backup_path = script_path.with_suffix(
        script_path.suffix + ".bak"
    )

    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_path}\n"
            "Rename the backup before running this script again."
        )

    # Save the backup first, then update the original script.
    backup_path.write_text(source, encoding="utf-8")
    script_path.write_text(updated, encoding="utf-8")

    print("=" * 65)
    print("CAPTURED COMPARISON LAYOUT UPDATED")
    print("=" * 65)

    print("Old layout: 2 x 2")
    print("New layout: 1 x 4 horizontal")
    print()
    print("Panel order:")
    print("1. Original Image")
    print("2. Ground Truth")
    print("3. Prediction")
    print("4. Overlay")
    print()
    print(f"Original backup: {backup_path}")
    print(f"Updated script:  {script_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()