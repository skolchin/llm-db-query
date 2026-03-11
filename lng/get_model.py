# Model support functions

import os
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast, Literal

load_dotenv()

_logger = logging.getLogger(__name__)

ModelType = Literal["ollama", "openai", "deepseek", "yandex"]
""" Allowed model types """

def get_model_qualified_name(model_type: ModelType, model_name: str | None = None) -> str:
    """ Return model of given type as a fully-qualified name in Langchain format (provider:model)"""

    match model_type:
        case "ollama":
            # Local Ollama instance provider
            model_name = model_name or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b"
            return f"ollama:{model_name}"

        case "openai" if "OPENAI_API_KEY" in os.environ:
            # OpenAI provider
            assert "OPENAI_API_KEY" in os.environ
            model_name = model_name or os.environ.get("OPENAI_MODEL") or "o3-mini"
            return f"openai:{model_name}"

        case "deepseek" if "DEEPSEEK_API_KEY" in os.environ:
            # Deepseek provider
            assert "DEEPSEEK_API_KEY" in os.environ
            model_name = model_name or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
            return f"deepseek:{model_name}"

        case "yandex":
            # Yandex provider - not supported
            raise ValueError("Yandex is not supported as generic model provider, use `get_model`")

        case _:
            raise ValueError(f"Unknown or unconfigured model {model_type}")

def get_model(model_type: ModelType, model_name: str | None = None) -> BaseChatModel:
    """ Return model of given type as Langchain ChatModel instance """

    match model_type:
        case "ollama":
            # Local Ollama instance provider
            model_name = model_name or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b"
            _logger.debug(f"Ollama model {model_name} is been created")
            return ChatOllama(
                model=model_name,
                temperature=0.,
            )

        case "openai" if "OPENAI_API_KEY" in os.environ:
            # OpenAI provider
            assert "OPENAI_API_KEY" in os.environ
            model_name = model_name or os.environ.get("OPENAI_MODEL") or "o3-mini"
            _logger.debug(f"Generic OpenAI model {model_name} is been created, base url is {os.environ.get('OPENAI_BASE_URL')}")
            return ChatOpenAI(
                model=model_name,
                api_key=os.environ["OPENAI_API_KEY"], # type:ignore
                base_url=os.environ.get("OPENAI_BASE_URL"), # type:ignore
                temperature=0.,
            )

        case "deepseek" if "DEEPSEEK_API_KEY" in os.environ:
            # Deepseek provider
            model_name = model_name or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
            _logger.debug(f"DeepSeek model {model_name} is been created")
            return ChatDeepSeek(
                model=model_name,
                api_key=os.environ["DEEPSEEK_API_KEY"], # type:ignore
                temperature=0.,
            )

        case "yandex" if "YANDEX_API_KEY" in os.environ and "YANDEX_FOLDER_ID" in os.environ:
            # Yandex provider
            from yandex_ai_studio_sdk import AIStudio
            from yandex_ai_studio_sdk.auth import APIKeyAuth

            model_name = model_name or os.environ.get("YANDEX_MODEL") or "yandex-gpt"
            sdk = AIStudio(folder_id=os.environ["YANDEX_FOLDER_ID"], auth=APIKeyAuth(os.environ["YANDEX_API_KEY"]))
            model = sdk.models.completions(model_name).configure(temperature=0).langchain()
            return cast(BaseChatModel, model)

        case _:
            raise ValueError(f"Unknown or unconfigured model {model_type}")
