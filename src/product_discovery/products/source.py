"""Resolve configured catalogue sources into the validated parquet repository."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import boto3

from product_discovery.products.repository import ParquetProductRepository
from product_discovery.schemas import Product

logger = logging.getLogger(__name__)


class S3DownloadClient(Protocol):
    """Subset of the S3 client used to retrieve a catalogue."""

    def download_file(self, bucket: str, key: str, filename: str) -> object:
        """Download one object to a local filename."""


class ConfiguredProductRepository:
    """Eagerly resolve a local or S3 catalogue and delegate parquet validation."""

    def __init__(
        self,
        catalogue_path: Path,
        catalogue_uri: str | None = None,
        *,
        s3_client: S3DownloadClient | None = None,
    ) -> None:
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        resolved_path = catalogue_path
        source = str(catalogue_path)

        if catalogue_uri:
            bucket, key = _parse_s3_uri(catalogue_uri)
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="product-discovery-catalogue-"
            )
            resolved_path = Path(self._temporary_directory.name) / "catalogue.parquet"
            source = f"s3://{bucket}/{key}"
            try:
                client = s3_client if s3_client is not None else boto3.client("s3")
                client.download_file(bucket, key, str(resolved_path))
            except Exception as exc:
                self.close()
                raise RuntimeError(f"failed to download catalogue from {catalogue_uri}") from exc

        try:
            repository = ParquetProductRepository(resolved_path)
            self._products = repository.list_products()
            checksum = _sha256(resolved_path)
        except Exception:
            self.close()
            raise

        logger.info(
            "catalogue loaded source=%s rows=%d sha256=%s",
            source,
            len(self._products),
            checksum,
        )

    def list_products(self) -> Sequence[Product]:
        """Return the products validated by the parquet repository."""

        return self._products

    def close(self) -> None:
        """Remove a downloaded catalogue, if this repository owns one."""

        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = unquote(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or "@" in parsed.netloc
        or ":" in parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("catalogue URI must be an s3:// URI with a bucket and object key")
    return parsed.netloc, key


def _sha256(path: Path) -> str:
    with path.open("rb") as catalogue_file:
        return hashlib.file_digest(catalogue_file, "sha256").hexdigest()
