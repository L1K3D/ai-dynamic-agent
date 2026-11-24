import joblib
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from functions.schemas import EstadoAgente, ForestCoverInput

class RPSConvNet(nn.Module):
    def __init__(self, use_dropout=False, use_batch_norm=False, extra_block=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        def make_block(in_channels, out_channels):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
            return nn.Sequential(*layers)

        self.block1 = make_block(3, 16)
        self.block2 = make_block(16, 32)
        self.block3 = make_block(32, 64) if extra_block else None
        final_channels = 64 if extra_block else 32
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * 4 * 4, 128),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(128, 3),
        )

    def forward(self, inputs):
        features = self.block1(inputs)
        features = self.block2(features)
        if self.block3 is not None:
            features = self.block3(features)
        features = self.adaptive_pool(features)
        return self.classifier(features)

print("[⚡ Jhonny-Log] Iniciando matriz de ferramentas...")
pipe_tabular = joblib.load("C:/projects/ai-dynamic-agent/python_scripts/modelo_forest_cover.joblib")

with open("C:/projects/ai-dynamic-agent/python_scripts/cnn_classes.json", "r") as f:
    cnn_classes = json.load(f)

model_cnn = RPSConvNet()
model_cnn.load_state_dict(torch.load("C:/projects/ai-dynamic-agent/python_scripts/cnn_rock_paper_scissors.pt", weights_only=True))
model_cnn.eval()

def no_tabular(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Rota tabular acionada.")
    try:
        cliente_valido = ForestCoverInput(**estado.get("dados_tabulares"))
        dados_input = cliente_valido.model_dump()
        
        # Garante as 54 features obrigatórias do modelo na ordem exata do treinamento
        features_esperadas = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area1"
        ]
        for i in range(1, 41):
            features_esperadas.append(f"Soil_Type{i}")
        features_esperadas.extend(["Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"])
        
        dados_alinhados = {coluna: 0 for coluna in features_esperadas}
        for chave, valor in dados_input.items():
            if chave in dados_alinhados:
                dados_alinhados[chave] = valor
                
        df_input = pd.DataFrame([dados_alinhados])
        df_input = df_input[features_esperadas]
        
        probabilidades = pipe_tabular.predict_proba(df_input)[0]
        indice_max = int(probabilidades.argmax())
        resultado = {"classe": str(pipe_tabular.classes_[indice_max]), "prob": float(probabilidades[indice_max])}
        
        return {"pred_tabular": resultado}
    except Exception as e:
        return {"erros": [f"Erro na validação tabular: {str(e)}"]}

def no_imagem(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Rota imagem acionada.")
    try:
        image_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        imagem = Image.open(estado.get("caminho_imagem")).convert("RGB")
        tensor_imagem = image_transform(imagem).unsqueeze(0)
        
        with torch.no_grad():
            logits = model_cnn(tensor_imagem)
            probabilidades = F.softmax(logits, dim=1).squeeze().numpy()
        
        top_indices = probabilidades.argsort()[::-1][:3]
        top_3 = [{"classe": cnn_classes[i], "probabilidade": float(probabilidades[i])} for i in top_indices]
        
        return {"pred_imagem": {"top3": top_3}}
    except Exception as e:
        return {"erros": [f"Erro ao processar imagem: {str(e)}"]}

def no_combinado(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Rota combinada acionada. Rodando ambos...")
    res_tab = no_tabular(estado)
    res_img = no_imagem(estado)
    
    update_dict = {}
    if "pred_tabular" in res_tab: update_dict["pred_tabular"] = res_tab["pred_tabular"]
    if "pred_imagem" in res_img: update_dict["pred_imagem"] = res_img["pred_imagem"]
    if "erros" in res_tab or "erros" in res_img:
        update_dict["erros"] = res_tab.get("erros", []) + res_img.get("erros", [])
        
    return update_dict