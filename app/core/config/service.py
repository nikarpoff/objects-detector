"""
Settings of this service on top of the shared ones.

The shared Settings are the same in the three AI services byte for byte; the
variables only the detector reads live here. They come from the same environment
and .env.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config.config import PROJECT_ROOT

BYTES_IN_MB = 1024 * 1024


class ServiceSettings(BaseSettings):
    # 1. Detection
    VOCABULARY_PATH: Path = Field(Path("configs/vocabulary.yaml"), description="Class vocabulary (DET-2); a relative path is taken from the service root")
    MAX_IMAGE_MB: float = Field(5, gt=0, description="Largest snapshot /v1/detect takes, MB; a larger one gets 413")

    # An empty variable keeps the default
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    @property
    def vocabulary_path(self) -> Path:
        return self.VOCABULARY_PATH if self.VOCABULARY_PATH.is_absolute() else PROJECT_ROOT / self.VOCABULARY_PATH

    @property
    def max_image_bytes(self) -> int:
        return int(self.MAX_IMAGE_MB * BYTES_IN_MB)


@lru_cache
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()
