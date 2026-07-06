import base64
from pathlib import Path

import nbformat

DATABRICKS_DIR = Path(__file__).resolve().parent.parent / "databricks"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATABRICKS_DIR.glob("0*.ipynb")):
        nb = nbformat.read(f, as_version=4)
        stem = f.stem
        fig_index = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                png_b64 = data.get("image/png")
                if not png_b64:
                    continue
                fig_index += 1
                out_path = ASSETS_DIR / f"{stem}_fig{fig_index}.png"
                out_path.write_bytes(base64.b64decode(png_b64))
                print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
