# 3D Gaussian Splatting no Colab

Notebook baseado no repositório oficial: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

Este projeto reconstrói uma cena 3D a partir de fotos (via COLMAP + 3D Gaussian Splatting), treina o modelo, renderiza novas vistas e calcula métricas de qualidade (PSNR, SSIM, LPIPS).

---

## 1. Requisitos

### 1.1 Conta e ambiente
- Conta Google (para usar **Google Colab** e **Google Drive** — o Drive é usado para salvar os checkpoints e resultados do treino).
- Runtime do Colab com **GPU** habilitada (`Ambiente de execução > Alterar tipo de ambiente de execução > GPU`). Funciona com T4 (plano free), mas A100/V100 (Colab Pro) rendem um treino mais rápido e com menos risco de erro de memória.
- Espaço livre no Google Drive (os checkpoints e point clouds do treino, especialmente com 30.000 iterações, ocupam vários GB).
- Tempo disponível: o treino padrão (30.000 iterações) pode levar de ~1h (GPU forte) a várias horas (T4). O pipeline completo de avaliação vs. baselines do paper (etapa opcional) leva ~7h numa A6000 segundo os autores — no Colab free pode não ser viável.

### 1.2 Dados de entrada — pasta "Leao_1" no Google Drive
Antes de começar, o usuário deve:

1. Subir a pasta com as fotos do objeto/cena para o Google Drive, com o nome **`Leao_1`** (mesmo esquema de subpastas por ângulo/cena usado no projeto anterior, por exemplo `cena_01`, `cena_02`, etc., cada uma com suas fotos `.jpg`/`.jpeg`/`.png`).
2. **Criar manualmente**, dentro do ambiente do Colab, a pasta de entrada esperada pelo notebook:
   ```
   /content/gaussian-splatting/data/input
   ```
3. **Copiar todas as imagens** que estão em `Leao_1` (e suas subpastas) para dentro de `/content/gaussian-splatting/data/input`. Isso pode ser feito manualmente (arrastando os arquivos no painel de arquivos do Colab) ou executando um trecho de código como:

```python
import os, shutil

DRIVE_ROOT = "/content/drive/MyDrive/Leao_1"       # pasta de origem no Drive
input_path = "/content/gaussian-splatting/data/input"  # pasta que o notebook espera
os.makedirs(input_path, exist_ok=True)

subpastas = [d for d in os.listdir(DRIVE_ROOT) if os.path.isdir(os.path.join(DRIVE_ROOT, d))]

total = 0
for subpasta in subpastas:
    subpasta_path = os.path.join(DRIVE_ROOT, subpasta)
    for fname in sorted(os.listdir(subpasta_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        # prefixa com o nome da subpasta pra evitar colisão de nomes
        novo_nome = f"{subpasta}_{fname}"
        shutil.copy(os.path.join(subpasta_path, fname), os.path.join(input_path, novo_nome))
        total += 1

print(f"{total} imagens copiadas de {len(subpastas)} subpastas para {input_path}")
```

> **Importante:** diferente do notebook do Gaussian Grouping (que lia direto do Drive), este pipeline exige que a pasta `data/input` já exista **dentro do Colab**, em `/content/gaussian-splatting/data/input`, com todas as fotos já copiadas para lá — o COLMAP e o restante do pipeline leem a partir desse caminho local, não do Drive.

### 1.3 Repositório e submódulos
- `graphdeco-inria/gaussian-splatting` — clonado com `--recursive` (traz os submódulos `diff-gaussian-rasterization` e `simple-knn`).

### 1.4 Bibliotecas e ferramentas instaladas
- `plyfile`, `tqdm`, `opencv-python`, `joblib`
- PyTorch + CUDA (já vem no ambiente do Colab; o notebook checa a versão instalada)
- Submódulos CUDA compilados localmente: `diff-gaussian-rasterization`, `simple-knn`
- **COLMAP** (via `apt-get install colmap`) — necessário apenas se for usar fotos próprias em vez do dataset de exemplo
- Para a etapa opcional de avaliação completa vs. baselines do paper: `torch`/`torchvision` (build `cu121`), `lpips`, `torchmetrics`, `scikit-image`, `pillow`, `numpy`, `pandas`, `matplotlib`

---

## 2. Passo a passo para rodar

### Passo 1 — Checar a GPU
```bash
!nvidia-smi
```
Confirme que uma GPU foi alocada antes de seguir.

### Passo 2 — Montar o Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/gaussian_splatting_output', exist_ok=True)
```
Essa pasta no Drive é onde os checkpoints e resultados do treino serão salvos (`-m` do `train.py`).

### Passo 3 — Clonar o repositório (com submódulos)
```bash
!git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
%cd gaussian-splatting
```

### Passo 4 — Instalar dependências
```bash
!pip install plyfile tqdm opencv-python joblib
```
E confira a versão de PyTorch/CUDA já disponível no Colab.

### Passo 5 — Compilar os submódulos CUDA
```bash
!pip install submodules/diff-gaussian-rasterization
!pip install submodules/simple-knn
```
Essa etapa compila extensões C++/CUDA e pode levar alguns minutos. Se der erro, geralmente é incompatibilidade entre a versão do CUDA do Colab e a esperada pelos submódulos.

### Passo 6 — Escolher a fonte das imagens

**Opção A — Dataset de exemplo oficial** (Tanks & Temples / Deep Blending, ~650 MB, já vem com as poses de câmera calculadas, não precisa rodar COLMAP):
```bash
!wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
!unzip -q tandt_db.zip
!ls tandt
```

**Opção B — Fotos próprias (pasta `Leao_1`)**, que é o caso deste projeto:
1. Instale o COLMAP:
   ```bash
   !apt-get install -y colmap
   !colmap -h  # confirma que instalou
   ```
2. **Crie a pasta** `/content/gaussian-splatting/data/input` e **copie todas as imagens da pasta `Leao_1`** do Drive para dentro dela (ver script completo na seção 1.2).
3. Rode as três etapas do COLMAP sobre as imagens em `data/input`:
   ```bash
   # 1. Extrai features (pontos-chave e descritores)
   !colmap feature_extractor \
       --database_path data/database.db \
       --image_path data/input \
       --ImageReader.camera_model SIMPLE_PINHOLE \
       --SiftExtraction.use_gpu 0

   # 2. Encontra correspondências entre as features de todas as imagens
   !colmap exhaustive_matcher \
       --database_path data/database.db \
       --SiftMatching.use_gpu 0

   # 3. Reconstrói a cena 3D e estima as poses das câmeras
   !mkdir -p data/sparse
   !colmap mapper \
       --database_path data/database.db \
       --image_path data/input \
       --output_path data/sparse
   ```
4. Mova as imagens de `data/input` para `data/images` (pasta que o `train.py` espera encontrar):
   ```python
   import os, shutil
   scene_path = '/content/gaussian-splatting/data'
   input_dir = os.path.join(scene_path, 'input')
   images_dir = os.path.join(scene_path, 'images')
   os.makedirs(images_dir, exist_ok=True)

   num_moved = 0
   for fname in os.listdir(input_dir):
       if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
           shutil.move(os.path.join(input_dir, fname), os.path.join(images_dir, fname))
           num_moved += 1
   print(f"Moved {num_moved} image files from {input_dir} to {images_dir}")
   ```

### Passo 7 — Treinar
```bash
!python train.py -s /content/gaussian-splatting/data -m /content/drive/MyDrive/gaussian_splatting_output/truck --eval --iterations 30000
```
- `-s`: caminho da cena (ajuste para `/content/gaussian-splatting/data` se estiver usando as fotos próprias da pasta `Leao_1`).
- `-m`: pasta de saída no Drive onde ficam os checkpoints (renomeie `truck` para o nome que preferir, ex. `leao`).
- Se estiver na GPU T4 (16 GB) e o treino der erro de memória (`CUDA out of memory`), reduza a densificação com `--densify_grad_threshold 0.0004` (padrão é `0.0002`) ou use imagens em resolução menor.

### Passo 8 — Renderizar e calcular métricas
```bash
!python render.py -m /content/drive/MyDrive/gaussian_splatting_output/truck
!python metrics.py -m /content/drive/MyDrive/gaussian_splatting_output/truck
```
Gera as imagens renderizadas a partir do modelo treinado e calcula PSNR / SSIM / LPIPS comparando com as imagens reais.

### Passo 9 — (Opcional) Pipeline completo de avaliação vs. baselines do paper
Reproduz treino + render + métricas para os datasets de referência (Mip-NeRF360, Tanks & Temples, Deep Blending) usados na comparação do paper original. É pesado — segundo os autores leva ~7h numa A6000; no Colab free pode não ser viável rodar tudo de uma vez.
```bash
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install lpips torchmetrics scikit-image
!pip install pillow numpy pandas matplotlib tqdm
```
Em seguida, o script `full_eval.py` (incluso no repositório) e a função `run_eval_from_scene` (de `eval_model.py`) automatizam essa avaliação completa.

---

## 3. Resumo do checklist antes de rodar
- [ ] GPU ativada no Colab (`nvidia-smi` mostra uma placa).
- [ ] Google Drive montado.
- [ ] Pasta `Leao_1` enviada para o Google Drive com as fotos organizadas em subpastas.
- [ ] Pasta `/content/gaussian-splatting/data/input` criada dentro do Colab.
- [ ] Todas as imagens de `Leao_1` copiadas para `/content/gaussian-splatting/data/input`.
- [ ] COLMAP instalado (se for usar fotos próprias, e não o dataset de exemplo).
- [ ] Repositório `gaussian-splatting` clonado com `--recursive` e submódulos CUDA compilados.

## 4. Observações finais
- Diferente do fluxo do Gaussian Grouping (que lia as fotos direto de uma pasta no Drive), aqui as imagens precisam estar fisicamente copiadas para dentro do ambiente do Colab, em `data/input`, antes de rodar o COLMAP.
- Depois do COLMAP, as imagens são movidas de `data/input` para `data/images` — não pule essa etapa, pois o `train.py` procura as fotos em `images/`, não em `input/`.
- Ajuste os nomes (`SCENE_NAME`, pasta de saída `-m` no `train.py`/`render.py`/`metrics.py`) para refletir o nome do seu objeto/cena em vez de `meu_objeto`/`truck`, que são apenas exemplos deixados no notebook.
