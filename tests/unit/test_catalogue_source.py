import hashlib
import logging
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from product_discovery.products.source import ConfiguredProductRepository


def _write_catalogue(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "product_id": "one",
                "title": "Laptop One",
                "brand": "Dell",
                "price": 1000.0,
                "ram_gb": 16.0,
                "storage_gb": 512.0,
                "screen_in": 14.0,
                "gpu": "Integrated",
                "weight_kg": 1.4,
            }
        ]
    ).to_parquet(path, index=False)


def _checksum(path: Path) -> str:
    with path.open("rb") as catalogue_file:
        return hashlib.file_digest(catalogue_file, "sha256").hexdigest()


def test_local_source_uses_existing_parquet_workflow_and_logs_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "catalogue.parquet"
    _write_catalogue(path)
    unused_s3_client = Mock()

    with caplog.at_level(logging.INFO, logger="product_discovery.products.source"):
        repository = ConfiguredProductRepository(path, s3_client=unused_s3_client)

    assert [product.product_id for product in repository.list_products()] == ["one"]
    unused_s3_client.download_file.assert_not_called()
    assert f"source={path}" in caplog.text
    assert "rows=1" in caplog.text
    assert f"sha256={_checksum(path)}" in caplog.text


def test_s3_source_downloads_once_with_default_credential_chain(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "source.parquet"
    _write_catalogue(source)
    s3_client = Mock()
    s3_client.download_file.side_effect = lambda bucket, key, filename: shutil.copyfile(
        source, filename
    )
    uri = "s3://catalogue-bucket/catalogues/laptop%20database.parquet"

    with (
        patch("product_discovery.products.source.boto3.client", return_value=s3_client) as factory,
        caplog.at_level(logging.INFO, logger="product_discovery.products.source"),
    ):
        repository = ConfiguredProductRepository(tmp_path / "unused.parquet", uri)

    first = repository.list_products()
    second = repository.list_products()
    downloaded_path = Path(s3_client.download_file.call_args.args[2])

    factory.assert_called_once_with("s3")
    s3_client.download_file.assert_called_once_with(
        "catalogue-bucket",
        "catalogues/laptop database.parquet",
        str(downloaded_path),
    )
    assert downloaded_path.is_file()
    assert first is second
    assert [product.product_id for product in first] == ["one"]
    assert "source=s3://catalogue-bucket/catalogues/laptop database.parquet" in caplog.text
    assert "rows=1" in caplog.text
    assert f"sha256={_checksum(source)}" in caplog.text

    repository.close()
    assert not downloaded_path.exists()


@pytest.mark.parametrize(
    "uri",
    [
        "https://catalogue-bucket/catalogue.parquet",
        "s3:///catalogue.parquet",
        "s3://catalogue-bucket",
        "s3://catalogue-bucket/catalogue.parquet?versionId=secret",
        "s3://access-key:secret@catalogue-bucket/catalogue.parquet",
    ],
)
def test_s3_source_rejects_invalid_uri(tmp_path: Path, uri: str) -> None:
    with pytest.raises(ValueError, match="must be an s3:// URI"):
        ConfiguredProductRepository(tmp_path / "unused.parquet", uri)
