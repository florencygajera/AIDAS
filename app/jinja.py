from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates
from jinja2 import FileSystemLoader


def build_templates(template_dir: Path) -> Jinja2Templates:
    resolved_dir = template_dir.resolve(strict=True)
    if not resolved_dir.is_dir():
        raise RuntimeError(f"Template directory is invalid: {resolved_dir}")

    templates = Jinja2Templates(directory=str(resolved_dir))
    if not isinstance(templates.env.loader, FileSystemLoader):
        raise RuntimeError(
            "Unexpected Jinja loader type: "
            f"{type(templates.env.loader).__name__}"
        )
    return templates


def validate_templates(templates: Jinja2Templates, sample_template: str = "index.html") -> None:
    if not isinstance(sample_template, str):
        raise TypeError(f"Template name must be str, got {type(sample_template).__name__}")

    env = templates.env
    if not isinstance(env.loader, FileSystemLoader):
        raise RuntimeError(f"Invalid Jinja loader: {type(env.loader).__name__}")

    template = templates.get_template(sample_template)
    if getattr(template, "name", None) != sample_template:
        raise RuntimeError(
            f"Unexpected template resolution: expected {sample_template!r}, got {getattr(template, 'name', None)!r}"
        )

