from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockLLM:
    """Mock que simula geração. Substitua por OpenAI/Anthropic/Llama."""

    def generate(self, prompt: str) -> str:
        # Extrai pergunta e contexto do prompt para resposta fake mas convincente
        # Prompt esperado: "Contexto: ...\nPergunta: ..."
        if "Contexto:" in prompt and "Pergunta:" in prompt:
            ctx_part = prompt.split("Pergunta:")[0].replace("Contexto:", "").strip()
            question = prompt.split("Pergunta:")[-1].strip()
            # pega primeiro trecho de contexto
            ctx_preview = ctx_part[:400].replace("\n", " ").strip()
            if not ctx_preview:
                return f"[MOCK LLM] Não encontrei contexto relevante para: '{question}'. Resposta genérica: RAG combina recuperação + geração."
            return (
                f"[MOCK LLM] Baseado no contexto recuperado:\n\"{ctx_preview}...\"\n\n"
                f"Resposta para '{question}': De acordo com os documentos indexados, a informação relevante está no contexto acima. "
                f"(troque MockLLM por OpenAI para geração real)"
            )
        return f"[MOCK LLM] Resposta para prompt de {len(prompt)} chars: {prompt[:200]}..."


def get_llm(kind: str = "mock", api_key: str | None = None, model: str = "gpt-4o-mini") -> LLM:
    if kind == "mock":
        return MockLLM()
    # Plugável:
    # if kind == "openai":
    #     from .llm_openai import OpenAILLM
    #     return OpenAILLM(api_key, model)
    return MockLLM()
