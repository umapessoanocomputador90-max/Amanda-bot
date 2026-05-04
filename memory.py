import json
import os
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20  # Máximo de mensagens antes de sumarizar
SUMMARY_TRIGGER = 16        # Quando sumarizar (mantém as últimas 8 mensagens recentes)
DATA_DIR = os.environ.get("DATA_DIR", "./data")


class MemoryManager:
    """
    Gerencia memória por usuário:
    - Histórico de mensagens recente (janela deslizante)
    - Sumário comprimido das sessões anteriores
    - Metadados do usuário (nome, primeira interação)
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def _user_file(self, user_id: str) -> str:
        return os.path.join(DATA_DIR, f"{user_id}.json")

    def _load(self, user_id: str) -> dict:
        path = self._user_file(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load memory for {user_id}, starting fresh.")
        return {}

    def _save(self, user_id: str, data: dict) -> None:
        path = self._user_file(user_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Could not save memory for {user_id}: {e}")

    def init_user(self, user_id: str, name: str) -> None:
        data = self._load(user_id)
        if not data:
            data = {
                "name": name,
                "history": [],
                "summary": "",
                "session_count": 0,
            }
            self._save(user_id, data)
        elif data.get("name") != name:
            data["name"] = name
            self._save(user_id, data)

    def add_message(self, user_id: str, role: str, content: str) -> None:
        data = self._load(user_id)
        if "history" not in data:
            data["history"] = []
        data["history"].append({"role": role, "content": content})
        self._save(user_id, data)

    def get_history(self, user_id: str) -> list:
        data = self._load(user_id)
        return data.get("history", [])

    def get_summary(self, user_id: str) -> str:
        data = self._load(user_id)
        return data.get("summary", "")

    def clear_history(self, user_id: str) -> None:
        data = self._load(user_id)
        data["history"] = []
        data["summary"] = ""
        data["session_count"] = 0
        self._save(user_id, data)

    def maybe_summarize(self, user_id: str, client: OpenAI) -> None:
        """
        Se o histórico crescer demais, comprime as mensagens antigas
        em um sumário e mantém apenas as mais recentes.
        """
        data = self._load(user_id)
        history = data.get("history", [])

        if len(history) < SUMMARY_TRIGGER:
            return

        # Mensagens antigas para sumarizar
        old_messages = history[:-8]  # Tudo menos as últimas 8
        recent_messages = history[-8:]  # Últimas 8 a manter

        existing_summary = data.get("summary", "")

        conversation_text = "\n".join(
            [f"{'Paciente' if m['role'] == 'user' else 'Amanda'}: {m['content']}"
             for m in old_messages]
        )

        summary_prompt = f"""Você é uma psicóloga fazendo anotações clínicas sobre uma sessão.
        
Sumário anterior (se houver):
{existing_summary if existing_summary else 'Nenhum ainda.'}

Trecho da conversa recente a ser incorporado ao sumário:
{conversation_text}

Crie um sumário clínico conciso (máximo 300 palavras) que capture:
- Principais temas e conflitos mencionados
- Estado emocional observado
- Progressos ou insights importantes
- Pontos de atenção ou padrões recorrentes
- Informações pessoais relevantes que a pessoa compartilhou

Escreva em terceira pessoa, como notas clínicas profissionais."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            new_summary = response.choices[0].message.content
            data["summary"] = new_summary
            data["history"] = recent_messages
            data["session_count"] = data.get("session_count", 0) + 1
            self._save(user_id, data)
            logger.info(f"Memory summarized for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to summarize memory for {user_id}: {e}")
