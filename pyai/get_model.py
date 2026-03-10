# Model support functions

import os
import dotenv
from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from typing import Literal, Dict, get_args

ModelType = Literal['ollama', 'deepseek', 'yandex']
""" Allowed model types  """

# Load environment from .env
dotenv.load_dotenv()

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,
    timeout=60,
)
""" Model settings (fixed) """

def get_model(model_type: ModelType, model_name: str | None = None) -> OpenAIChatModel | None:
    """ Return LLM instance """

    match model_type:
        case 'ollama':
            # Local Ollama instance provider
            model_name = model_name or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b"
            return OpenAIChatModel(
                model_name=model_name,
                provider=OllamaProvider('http://localhost:11434/v1'),
                settings=MODEL_SETTINGS,
            )

        case 'deepseek':
            # Deepseek provider (does not support Responses API)
            if not 'DEEPSEEK_API_KEY' in os.environ:
                return None
            
            model_name = model_name or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
            return OpenAIChatModel(
                model_name=model_name,
                provider=DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY']),
                settings=MODEL_SETTINGS,
            )

        case "openai" if "OPENAI_API_KEY" in os.environ:
            # OpenAI provider

            model_name = model_name or os.environ.get("OPENAI_MODEL") or "o3-mini"
            client = AsyncOpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                base_url=os.environ.get("OPENAI_BASE_URL"),
                project=os.environ.get("OPENAI_FOLDER_ID"),
            )
            return OpenAIChatModel(
                model=model_name,
                provider=OpenAIProvider(openai_client=client),
                settings=MODEL_SETTINGS,
            )

        case 'yandex':
            # OpenAI-compatible Yandex provider
            if not 'YANDEX_API_KEY' in os.environ or not 'YANDEX_FOLDER_ID' in os.environ:
                return None

            model_name = model_name or os.environ.get("YANDEX_MODEL") or "yandex-gpt"
            client = AsyncOpenAI(
                api_key=os.environ['YANDEX_API_KEY'],
                base_url='https://ai.api.cloud.yandex.net/v1',
                project=os.environ['YANDEX_FOLDER_ID'],
            )
            return OpenAIChatModel(
                model_name=f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{model_name}/latest",
                provider=OpenAIProvider(openai_client=client),
                settings=MODEL_SETTINGS,
            )

        case _:
            raise ValueError(f'Unknown model type {model_type}')

def get_models() -> Dict[str, OpenAIChatModel]:
    """ Return all models available """
    all_models = {k: get_model(k) for k in get_args(ModelType)}
    return {k: m for k, m in all_models.items() if m is not None}
