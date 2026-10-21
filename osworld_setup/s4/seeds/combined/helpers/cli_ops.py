"""CLI operation primitives."""
import subprocess
import xml.etree.ElementTree as ET
import zipfile


def gsettings_get(schema, key):
    """Read a GNOME setting via `gsettings get`. Returns the trimmed stdout
    (typically a quoted string, or 'true'/'false' for booleans)."""
    r = subprocess.run(
        ["gsettings", "get", schema, key],
        capture_output=True, text=True, timeout=10,
    )
    return r.stdout.strip()


def gsettings_set(schema, key, value):
    """Write a GNOME setting via `gsettings set`. Returns
    {'returncode', 'stderr'} — non-zero rc means the value didn't take."""
    r = subprocess.run(
        ["gsettings", "set", schema, key, str(value)],
        capture_output=True, text=True, timeout=10,
    )
    return {"returncode": r.returncode, "stderr": r.stderr}


def get_ooxml_structure(filepath):
    """Map every .xml part in an OOXML zip (.docx/.pptx/.xlsx) to its root tag
    and child count. Returns {part_path: {"tag": ..., "children": N} | {"error": ...}}."""
    parts = {}
    with zipfile.ZipFile(filepath) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                try:
                    root = ET.parse(z.open(name)).getroot()
                    parts[name] = {"tag": root.tag, "children": len(root)}
                except ET.ParseError:
                    parts[name] = {"error": "parse failed"}
    return parts


def get_ooxml_text(filepath):
    """Extract every non-empty text node from every .xml part of an OOXML zip.
    Returns a list of stripped strings, in document order."""
    texts = []
    with zipfile.ZipFile(filepath) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                try:
                    for elem in ET.parse(z.open(name)).iter():
                        if elem.text and elem.text.strip():
                            texts.append(elem.text.strip())
                except ET.ParseError:
                    pass
    return texts
