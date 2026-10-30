"""Select pre-generated robot display meshes without altering physical geometry."""
import json
from pathlib import Path


class RobotVisuals:
    def __init__(self, assets_path: Path, mode: str = 'simple'):
        if mode not in {'simple', 'cad'}:
            raise ValueError(f'Unknown robot visual mode {mode!r}; expected simple or cad')
        self.assets_path = Path(assets_path)
        self.mode = mode
        self.metadata = (json.loads((self.assets_path / 'simple_visuals.json').read_text())
                         if mode == 'simple' else {})

    def mesh_path(self, definition_key: str, original: str) -> Path | None:
        replacement = self.metadata.get('definitions', {}).get(definition_key)
        if replacement is None:
            return self.assets_path / original
        mesh = replacement['mesh']
        return self.assets_path / mesh if mesh else None
