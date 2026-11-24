from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# 1. ESTADO DO AGENTE (Memória Compartilhada / State)
# ==========================================
# Definimos as rotas aceitas para o controle de tráfego
Rota = Literal["tabular", "imagem", "combinado", "fora_escopo"]

class EstadoAgente(TypedDict, total=False):
    pergunta: str                      # O texto que o usuário digitou
    dados_tabulares: Optional[dict]    # Dados numéricos/categóricos do Forest Cover
    caminho_imagem: Optional[str]      # Caminho do arquivo da foto (Pedra-Papel-Tesoura)
    rota: Rota                         # Caminho decidido pelo nó roteador
    erros: list[str]                   # Log de falhas (validação, confiança baixa, etc)
    pred_tabular: Optional[dict]       # Resultado cru do modelo tabular (classe e probabilidade)
    pred_imagem: Optional[dict]        # Resultado cru da CNN (top 3 classes)
    resposta: str                      # O texto final gerado pelo LLM para o usuário

# ==========================================
# 2. GUARDRAILS (Firewall de Validação)
# ==========================================
class ForestCoverInput(BaseModel):
    """Validador estrito para a entrada do modelo tabular (Forest Cover Type)."""
    Elevation: int = Field(ge=0, description="Elevacao em metros")
    Aspect: int = Field(ge=0, le=360, description="Aspecto em graus de azimute")
    Slope: int = Field(ge=0, le=90, description="Inclinacao em graus")
    Horizontal_Distance_To_Hydrology: int = Field(ge=0)
    Vertical_Distance_To_Hydrology: int
    Horizontal_Distance_To_Roadways: int = Field(ge=0)
    Hillshade_9am: int = Field(ge=0, le=255)
    Hillshade_Noon: int = Field(ge=0, le=255)
    Hillshade_3pm: int = Field(ge=0, le=255)
    Horizontal_Distance_To_Fire_Points: int = Field(ge=0)
    
    # Permitir chaves extras para caso o joblib peça as colunas dummies
    model_config = {"extra": "allow"}

class Decisao(BaseModel):
    """Estrutura de saída estruturada obrigatória para o LLM Roteador."""
    rota: Rota
    motivo: str