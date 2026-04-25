import shutil
from pathlib import Path


def main():
    root = Path("d:/ultra-hdr-converter")
    src_dir = root / "src" / "ultra_hdr_converter"
    core_dir = src_dir / "core"
    ui_dir = src_dir / "ui"

    core_dir.mkdir(exist_ok=True)
    ui_dir.mkdir(exist_ok=True)
    
    (core_dir / "__init__.py").touch()
    (ui_dir / "__init__.py").touch()

    # Move files
    core_files = ["converter.py", "gain_map.py", "ultrahdr_encoder.py", "jpeg_io.py", "color.py", "color_cms.py"]
    ui_files = ["cli.py", "gui.py", "_gui_style.py", "assets"]

    for f in core_files:
        p = src_dir / f
        if p.exists():
            shutil.move(str(p), str(core_dir / f))

    for f in ui_files:
        p = src_dir / f
        if p.exists():
            shutil.move(str(p), str(ui_dir / f))

    # Update imports across all py files in src and tests
    replacements = {
        "ultra_hdr_converter.converter": "ultra_hdr_converter.core.converter",
        "ultra_hdr_converter.gain_map": "ultra_hdr_converter.core.gain_map",
        "ultra_hdr_converter.ultrahdr_encoder": "ultra_hdr_converter.core.ultrahdr_encoder",
        "ultra_hdr_converter.jpeg_io": "ultra_hdr_converter.core.jpeg_io",
        "ultra_hdr_converter.color ": "ultra_hdr_converter.core.color ",
        "ultra_hdr_converter.color\n": "ultra_hdr_converter.core.color\n",
        "ultra_hdr_converter.color_cms": "ultra_hdr_converter.core.color_cms",
        "ultra_hdr_converter.cli": "ultra_hdr_converter.ui.cli",
        "ultra_hdr_converter.gui": "ultra_hdr_converter.ui.gui",
        "ultra_hdr_converter._gui_style": "ultra_hdr_converter.ui._gui_style",
        "ultra_hdr_converter.assets": "ultra_hdr_converter.ui.assets",
        "from .color ": "from .core.color ",
        "from .color_cms ": "from .core.color_cms ",
        "from .converter ": "from .core.converter ",
        "from .gain_map ": "from .core.gain_map ",
    }

    def process_dir(directory):
        for path in Path(directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            orig = text
            for old, new in replacements.items():
                text = text.replace(old, new)
            if orig != text:
                path.write_text(text, encoding="utf-8")
                print(f"Updated imports in {path.relative_to(root)}")

    process_dir(src_dir)
    process_dir(root / "tests")

if __name__ == "__main__":
    main()
