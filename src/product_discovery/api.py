"""FastAPI application and local dependency assembly."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import ValidationError

from product_discovery.configuration import get_settings
from product_discovery.llm.adapter import LLMAdapterError
from product_discovery.llm.openai_adapter import OpenAIAdapter
from product_discovery.preferences.manager import PreferenceStateManager
from product_discovery.products.retrieval import RetrievalEngine
from product_discovery.products.search import ProductSearchService
from product_discovery.products.selection import RepresentativeProductSelector
from product_discovery.products.source import ConfiguredProductRepository
from product_discovery.responses.generator import GroundedResponseGenerator
from product_discovery.schemas import ChatRequest, ChatResponse
from product_discovery.services.chat import ChatService

app = FastAPI(title="Conversational Product Discovery", version="0.1.0")


@lru_cache
def build_chat_service() -> ChatService:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("PRODUCT_DISCOVERY_OPENAI_API_KEY is not configured")

    llm = OpenAIAdapter(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )
    repository = ConfiguredProductRepository(
        catalogue_path=settings.catalogue_path,
        catalogue_uri=settings.catalogue_uri,
    )
    product_search = ProductSearchService(
        retrieval=RetrievalEngine(repository),
        selector=RepresentativeProductSelector(),
        max_candidates=settings.max_candidates,
    )
    return ChatService(
        llm=llm,
        state_manager=PreferenceStateManager(),
        product_search=product_search,
        response_generator=GroundedResponseGenerator(llm),
    )


def get_chat_service() -> ChatService:
    try:
        return build_chat_service()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    try:
        return service.chat(request)
    except LLMAdapterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="LLM returned an invalid update") from exc
