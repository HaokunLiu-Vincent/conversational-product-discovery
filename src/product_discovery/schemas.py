"""Pydantic contracts shared across deterministic application boundaries."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PreferenceField = Literal[
    "budget_max",
    "brand",
    "ram_min_gb",
    "storage_min_gb",
    "screen_size_max_in",
    "gpu_required",
    "weight_max_kg",
    "usage",
    "preference_notes",
]
UpdateOperation = Literal["set", "remove"]
type UpdateValue = str | int | float | None

CANONICAL_BRANDS = frozenset(
    {
        "Acer",
        "Apple",
        "ASUS",
        "Dell",
        "Gateway",
        "Gigabyte",
        "HP",
        "Huawei",
        "Lenovo",
        "LG",
        "Microsoft",
        "MSI",
        "Panasonic",
        "Razer",
        "Samsung",
        "Toshiba",
    }
)
CANONICAL_GPU_VALUES = frozenset(
    {
        "Integrated",
        "Intel Iris Xe",
        "AMD Radeon",
        "RTX 3050",
        "RTX 3050 Ti",
        "RTX 3060",
        "RTX 4050",
        "RTX 4060",
        "RTX 4070",
    }
)
USAGE_VALUES = frozenset({"business", "study", "gaming", "all_round"})


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class PreferenceState(BaseModel):
    """Application-owned session preferences."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    budget_max: float | None = Field(default=None, gt=0, le=100_000)
    brand: str | None = None
    ram_min_gb: int | None = Field(default=None, ge=1, le=512)
    storage_min_gb: int | None = Field(default=None, ge=1, le=32_768)
    screen_size_max_in: float | None = Field(default=None, ge=8, le=25)
    gpu_required: str | None = None
    weight_max_kg: float | None = Field(default=None, gt=0, le=20)
    usage: str | None = None
    preference_notes: str | None = None

    @field_validator("brand", "gpu_required", "usage", "preference_notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        return _normalize_optional_text(value)


class PreferenceUpdateItem(BaseModel):
    """One proposed state mutation."""

    model_config = ConfigDict(extra="forbid")

    field: PreferenceField
    operation: UpdateOperation
    value: UpdateValue = None

    @model_validator(mode="after")
    def operation_matches_value(self) -> "PreferenceUpdateItem":
        if self.operation == "set" and self.value is None:
            raise ValueError("set operation requires a value")
        if self.operation == "remove" and self.value is not None:
            raise ValueError("remove operation requires a null value")
        return self


class PreferenceUpdate(BaseModel):
    """Structured proposal that deterministic code must validate before applying."""

    model_config = ConfigDict(extra="forbid")

    updates: list[PreferenceUpdateItem] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: str | None = None

    @field_validator("clarification_question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> str | None:
        return _normalize_optional_text(value)

    @model_validator(mode="after")
    def update_is_unambiguous(self) -> "PreferenceUpdate":
        fields = [item.field for item in self.updates]
        if len(fields) != len(set(fields)):
            raise ValueError("an update may mutate each field at most once")
        if self.clarification_needed and self.clarification_question is None:
            raise ValueError("clarification_needed=true requires a clarification_question")
        if not self.clarification_needed and self.clarification_question is not None:
            raise ValueError("clarification_question requires clarification_needed=true")
        return self


class Product(BaseModel):
    """Normalized catalogue record used by deterministic retrieval."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    product_id: str
    title: str
    brand: str | None = None
    price: float | None = Field(default=None, ge=0)
    ram_gb: float | None = Field(default=None, ge=0)
    storage_gb: float | None = Field(default=None, ge=0)
    screen_in: float | None = Field(default=None, ge=0)
    gpu: str | None = None
    weight_kg: float | None = Field(default=None, ge=0)
    cpu: str | None = None
    os: str | None = None
    description: str | None = None
    average_rating: float | None = None
    rating_number: int | None = None
    quality_score: int | None = None


class RetrievalResult(BaseModel):
    """API-facing summary derived from the complete qualified candidate set."""

    qualified_candidate_count: int = Field(ge=0)
    representative_products: list[Product]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    preferences: PreferenceState
    clarification_needed: bool
    clarification_question: str | None
    qualified_candidate_count: int = Field(ge=0)
    representative_products: list[Product]
