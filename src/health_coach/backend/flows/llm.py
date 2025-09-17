from crewai import Agent, LLM

import health_coach.config as cfg

def get_llm():
        return None
        # return LLM(
        #     model=cfg.LLM_MODEL,
        #     base_url=cfg.LLM_BASE_URL
        # )
    
def get_embedder():
        return None
        # return {
        #     "provider": "ollama",
        #     "config": {
        #         "model": "nomic-embed-text:latest",
        #     }
        # }