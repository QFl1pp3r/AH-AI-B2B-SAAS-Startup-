"""Wire format between the CLI and the GPU server: a zip of result.json plus output files."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from ah.contract import RESULT_FILE, ToolResult

MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024


def pack_result(result: ToolResult) -> bytes:
    """Zip outputs under their basenames and rewrite paths to match."""
    buffer = io.BytesIO()
    packed = result.model_copy(deep=True)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, output in enumerate(packed.outputs):
            name = f"{index:02d}_{Path(output.path).name}"
            archive.write(output.path, arcname=name)
            output.path = name
        archive.writestr(RESULT_FILE, packed.model_dump_json())
    return buffer.getvalue()


def unpack_result(payload: bytes, out_dir: Path) -> ToolResult:
    """Inverse of pack_result. Rejects absolute paths, traversal and zip bombs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_UNPACKED_BYTES:
            raise ValueError("server response unpacks to more than 2 GB")
        result = ToolResult.model_validate_json(archive.read(RESULT_FILE))
        allowed = {info.filename for info in archive.infolist()} - {RESULT_FILE}
        for output in result.outputs:
            name = output.path
            if name not in allowed or Path(name).name != name or name.startswith("."):
                raise ValueError(f"server returned an unsafe output path: {name!r}")
            target = out_dir / name
            target.write_bytes(archive.read(name))
            output.path = str(target)
    return result
