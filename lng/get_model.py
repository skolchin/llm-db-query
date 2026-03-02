# Model support functions

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast, Literal

load_dotenv()

ModelType = Literal['ollama', 'deepseek', 'yandex']
""" Allowed model types """

def get_model_name(model_type: ModelType) -> str:
    """ Return model of given type as model name in Langchain format """

    match model_type:
        case 'ollama':
            # Local Ollama instance provider
            return f"ollama:{os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b')}"

        case 'deepseek' if 'DEEPSEEK_API_KEY' in os.environ:
            # Deepseek provider
            assert 'DEEPSEEK_API_KEY' in os.environ
            return f"deepseek:{os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')}"

        case 'yandex' if 'YANDEX_API_KEY' in os.environ and 'YANDEX_FOLDER_ID' in os.environ:
            # YandexGPT provider
            assert 'YANDEX_API_KEY' in os.environ
            assert 'YANDEX_FOLDER_ID' in os.environ
            return f"yandex:{os.environ.get('YANDEX_MODEL', 'yandexgpt')}"

        case _:
            raise ValueError(f'Unknown or unconfigured model {model_type}')

def get_model(model_type: ModelType) -> BaseChatModel:
    """ Return model of given type as Langchain ChatModel instance """

    match model_type:
        case 'ollama':
            # Local Ollama instance provider
            return ChatOllama(model=os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b'))

        case 'deepseek' if 'DEEPSEEK_API_KEY' in os.environ:
            # Deepseek provider
            return ChatDeepSeek(
                model=os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
                api_key=os.environ['DEEPSEEK_API_KEY'], # type:ignore
            )

        case 'yandex' if 'YANDEX_API_KEY' in os.environ and 'YANDEX_FOLDER_ID' in os.environ:
            # YandexGPT provider
            from yandex_ai_studio_sdk import AIStudio
            from yandex_ai_studio_sdk.auth import APIKeyAuth

            sdk = AIStudio(folder_id=os.environ['YANDEX_FOLDER_ID'], auth=APIKeyAuth(os.environ['YANDEX_API_KEY']))
            model = sdk.models.completions(os.environ.get('YANDEX_MODEL', 'yandexgpt')).langchain()
            return cast(BaseChatModel, model)

        case _:
            raise ValueError(f'Unknown or unconfigured model {model_type}')
