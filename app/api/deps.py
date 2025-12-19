from functools import lru_cache

from ml.inference.predict import load_artifacts_from_registry, LoadedArtifacts


@lru_cache(maxsize=1)
def get_artifacts() -> LoadedArtifacts:
    return load_artifacts_from_registry("ml/models/registry.md")
