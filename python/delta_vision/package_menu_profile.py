"""把已校验菜单 Profile 的全部运行依赖打包为可移植目录。"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import sys
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .menu_profile import load_menu_profile


@dataclass(frozen=True, slots=True)
class MenuProfileBundleResult:
    profile_path: Path
    source_run_count: int
    template_asset_count: int
    total_bytes: int
    archive_path: Path | None


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是对象")
    return value


def _relative_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{field} 必须是非空相对路径")
    posix_value = PurePosixPath(value)
    windows_value = PureWindowsPath(value)
    if (
        posix_value.is_absolute()
        or windows_value.is_absolute()
        or bool(windows_value.drive)
        or ".." in posix_value.parts
        or ".." in windows_value.parts
    ):
        raise ValueError(f"{field} 必须位于 Profile 目录内")
    candidate = (root / value.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{field} 不能逃逸 Profile 目录")
    return candidate


def _copy_file(*, source_root: Path, staging: Path, reference: object, field: str) -> Path:
    source = _relative_path(source_root, reference, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} 必须是普通文件")
    relative = source.relative_to(source_root)
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return relative


def _copy_directory(
    *,
    source_root: Path,
    staging: Path,
    reference: object,
    field: str,
) -> Path:
    source = _relative_path(source_root, reference, field=field)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"{field} 必须是普通目录")
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"{field} 不能包含符号链接: {candidate}")
    relative = source.relative_to(source_root)
    destination = staging / relative
    shutil.copytree(source, destination)
    return relative


def _total_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _publish_archive_no_replace(staging: Path, destination: Path) -> None:
    """用同文件系统硬链接发布；目标已存在时由操作系统原子拒绝。"""

    os.link(staging, destination)


def _publish_directory_no_replace(staging: Path, destination: Path) -> None:
    """使用平台原生原子重命名，并拒绝替换并发出现的目标目录。"""

    if os.name == "nt":
        os.rename(staging, destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "当前 Linux libc 不支持原子 no-replace 目录发布",
                destination,
            ) from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise OSError(
            errno.ENOTSUP,
            "当前平台不支持原子 no-replace 目录发布",
            destination,
        )

    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination,
        )


def _remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def materialize_menu_profile(
    profile: str | Path,
    output: str | Path,
    *,
    archive_path: str | Path | None = None,
) -> MenuProfileBundleResult:
    """复制所有被引用的完整来源运行和模板，并在发布前重新校验。"""

    profile_path = Path(profile).resolve()
    output_path = Path(output).resolve()
    archive = None if archive_path is None else Path(archive_path).resolve()
    if output_path.exists():
        raise ValueError(f"输出目录已经存在: {output_path}")
    if archive is not None:
        if archive.suffix.lower() != ".zip":
            raise ValueError("archive_path 必须使用 .zip 后缀")
        if archive.is_relative_to(output_path):
            raise ValueError("归档文件不能位于输出目录内")
        if archive.exists():
            raise ValueError(f"归档文件已经存在: {archive}")

    # 先对原始 Profile、来源运行、模板和哈希做完整校验，失败时不创建输出。
    load_menu_profile(profile_path)
    source_root = profile_path.parent.resolve()
    raw = _mapping(json.loads(profile_path.read_text(encoding="utf-8")), field="Profile")
    source_runs = raw.get("source_runs")
    templates = raw.get("templates")
    if not isinstance(source_runs, list) or not isinstance(templates, list):
        raise ValueError("Profile 缺少 source_runs 或 templates")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    archive_staging = (
        None
        if archive is None
        else archive.with_name(f".{archive.stem}.tmp-{uuid.uuid4().hex}.zip")
    )
    staging.mkdir()
    output_published = False
    archive_published = False
    try:
        copied_run_directories: set[Path] = set()
        for index, item in enumerate(source_runs):
            run = _mapping(item, field=f"source_runs[{index}]")
            relative = _copy_directory(
                source_root=source_root,
                staging=staging,
                reference=run.get("directory"),
                field=f"source_runs[{index}].directory",
            )
            if relative in copied_run_directories:
                raise ValueError("source_runs 不能重复引用同一来源目录")
            copied_run_directories.add(relative)

        copied_assets: set[Path] = set()
        for index, item in enumerate(templates):
            template = _mapping(item, field=f"templates[{index}]")
            for detector_name in ("page", "action"):
                detector_value = template.get(detector_name)
                if detector_value is None:
                    continue
                detector = _mapping(
                    detector_value,
                    field=f"templates[{index}].{detector_name}",
                )
                copied_assets.add(
                    _copy_file(
                        source_root=source_root,
                        staging=staging,
                        reference=detector.get("image"),
                        field=f"templates[{index}].{detector_name}.image",
                    )
                )

        bundled_profile = staging / "menu.json"
        bundled_profile.write_bytes(profile_path.read_bytes())
        load_menu_profile(bundled_profile)
        total_bytes = _total_file_bytes(staging)

        if archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            assert archive_staging is not None
            created = Path(
                shutil.make_archive(
                    str(archive_staging.with_suffix("")),
                    "zip",
                    root_dir=staging,
                )
            ).resolve()
            if created != archive_staging:
                raise RuntimeError(f"归档路径不符合预期: {created}")
            with zipfile.ZipFile(archive_staging) as bundle:
                corrupted_entry = bundle.testzip()
            if corrupted_entry is not None:
                raise ValueError(f"归档校验失败: {corrupted_entry}")

        _publish_directory_no_replace(staging, output_path)
        output_published = True
        if archive is not None:
            assert archive_staging is not None
            _publish_archive_no_replace(archive_staging, archive)
            archive_published = True
            archive_staging.unlink()
    except BaseException:
        error = sys.exception()
        assert error is not None
        cleanup_errors: list[tuple[str, BaseException]] = []

        def cleanup(label: str, action) -> None:
            try:
                action()
            except BaseException as cleanup_error:
                cleanup_errors.append((label, cleanup_error))

        if archive_published and archive is not None:
            cleanup("正式归档", lambda: _remove_file_if_exists(archive))
        if output_published:
            cleanup("正式输出目录", lambda: _remove_tree_if_exists(output_path))
        if archive_staging is not None:
            cleanup("临时归档", lambda: _remove_file_if_exists(archive_staging))
        cleanup("临时输出目录", lambda: _remove_tree_if_exists(staging))

        for label, cleanup_error in cleanup_errors:
            error.add_note(
                f"{label}清理失败: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise

    return MenuProfileBundleResult(
        profile_path=output_path / "menu.json",
        source_run_count=len(source_runs),
        template_asset_count=len(copied_assets),
        total_bytes=total_bytes,
        archive_path=archive,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成可移植菜单 Profile bundle")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    try:
        result = materialize_menu_profile(
            args.profile,
            args.output,
            archive_path=args.archive,
        )
    except Exception as error:
        print(f"菜单 Profile 打包失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "complete",
                "profile": str(result.profile_path),
                "source_run_count": result.source_run_count,
                "template_asset_count": result.template_asset_count,
                "total_bytes": result.total_bytes,
                "archive": None if result.archive_path is None else str(result.archive_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
