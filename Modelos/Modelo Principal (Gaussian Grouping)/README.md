# Projeto Principal — Gaussian Grouping (Reconstrução 3D, Remoção e Inpainting de Objetos)

Este projeto usa o pipeline **[Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)** rodando no **Google Colab** para:

1. Reconstruir uma cena 3D a partir de fotos (via COLMAP + 3D Gaussian Splatting);
2. Segmentar e identificar objetos na cena (via DEVA + SAM + GroundingDINO);
3. Remover objetos da cena 3D;
4. Fazer *inpainting* (preenchimento) da região removida (via LaMa);
5. Avaliar os resultados com métricas de qualidade (PSNR, SSIM, LPIPS).

---

## 1. Requisitos

### 1.1 Conta e ambiente
- Conta Google (para usar **Google Colab** e **Google Drive**).
- Runtime do Colab com **GPU** habilitada (`Ambiente de execução > Alterar tipo de ambiente de execução > GPU`). O notebook foi testado com uma GPU classe NVIDIA RTX/A100 (≈24–100 GB de VRAM). Sem GPU o treino não roda.
- Espaço livre suficiente no Google Drive (alguns GB, pois o pipeline salva backups de imagens, checkpoints e point clouds).
- Tempo disponível: o pipeline completo (COLMAP + treino de 30.000 iterações + remoção + inpainting) leva **várias horas** (o treino sozinho levou ~3h e o fine-tuning do inpaint ~4h20 no ambiente usado).

### 1.2 Dados de entrada — pasta "Leao" no Google Drive
O notebook espera encontrar as fotos do objeto/cena dentro do Google Drive, organizadas assim:

```
/content/drive/MyDrive/Leao/
├── cena_01/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── cena_02/
│   ├── IMG_0005.jpg
│   └── ...
├── cena_03/
├── cena_04/
├── cena_05/
├── cena_06/
├── cena_07/
└── cena_08/
```

**Como subir:**
1. Acesse [drive.google.com](https://drive.google.com) com a mesma conta que será usada no Colab.
2. Na raiz do "Meu Drive", crie uma pasta chamada exatamente **`Leao`**.
3. Dentro dela, crie subpastas (uma por "cena"/ângulo de captura — no projeto original foram 8 subpastas, `cena_01` a `cena_08`).
4. Coloque as fotos (`.jpg`, `.jpeg` ou `.png`) dentro de cada subpasta.
5. No total, o pipeline espera múltiplas imagens do mesmo objeto capturadas de ângulos diferentes (no projeto original: 33 imagens ao todo, extraídas de 8 subpastas). Quanto mais ângulos e sobreposição entre fotos, melhor a reconstrução 3D via COLMAP.

> Dica: fotografe o objeto de todos os lados, com boa iluminação e sobreposição de ~60-80% entre fotos consecutivas, para o COLMAP conseguir casar os pontos-chave entre as imagens.

### 1.3 Repositórios necessários (clonados automaticamente pelo notebook)
- `lkeab/gaussian-grouping` (repositório principal, inclui submódulos `diff-gaussian-rasterization` e `simple-knn`)
- `Tracking-Anything-with-DEVA` (rastreamento temporal de máscaras)
- `hkchengrex/Grounded-Segment-Anything` (Segment Anything + GroundingDINO)
- `lama` (inpainting) — dentro da pasta do gaussian-grouping
- Pesos pré-treinados: modelos do DEVA (baixados via script `download_models.sh`) e o modelo **big-lama** (baixado de `huggingface.co/smartywu/big-lama`)

### 1.4 Bibliotecas Python instaladas ao longo do notebook
- `plyfile==0.8.1`
- `tqdm`, `scipy`, `wandb`, `opencv-python`, `scikit-learn`, `lpips`
- `ninja` (necessário para compilar as extensões CUDA)
- Submódulos compilados localmente: `diff-gaussian-rasterization`, `simple-knn`
- `easydict==1.9.0`
- `albumentations`, `kornia`, `hydra-core`, `pytorch-lightning`
- `huggingface_hub` (versão fixada em `>=0.34.0,<1.0`)
- `webdataset`
- `transformers==4.57.6` (forçado, sem dependências, para compatibilidade)
- `scikit-image`
- `git-lfs` (via `apt-get`)
- **COLMAP** (via `apt-get install colmap`) — programa de fototriangulação/SfM
- PyTorch + CUDA (já vem pré-instalado no ambiente do Colab)

---

## 2. Passo a passo para rodar

### Passo 1 — Preparar o ambiente
1. Abra o notebook no Google Colab.
2. Ative a GPU (`Ambiente de execução > Alterar tipo de ambiente de execução > T4/A100/GPU`).
3. Suba a pasta `Leao` (com as fotos) para o seu Google Drive, na raiz de "Meu Drive", conforme a seção 1.2.

### Passo 2 — Clonar repositórios e instalar dependências
Execute em ordem as células que:
- Clonam o `gaussian-grouping` e entram na pasta.
- Instalam `plyfile`, `tqdm`, `scipy`, `wandb`, `opencv-python`, `scikit-learn`, `lpips`.
- Instalam os submódulos CUDA (`diff-gaussian-rasterization`, `simple-knn`) — inclui um patch manual que adiciona `#include <cfloat>` no arquivo `simple_knn.cu` antes de compilar.
- Instalam o DEVA (`pip install -e .`) e baixam os pesos com `bash scripts/download_models.sh`.
- Clonam e instalam o `Grounded-Segment-Anything` (SAM + GroundingDINO), definindo as variáveis de ambiente `BUILD_WITH_CUDA=True` e `CUDA_HOME=/usr/local/cuda`.
- Instalam dependências do LaMa (`easydict`, `albumentations`, `kornia`, `hydra-core`, `pytorch-lightning`).

### Passo 3 — Montar o Google Drive e localizar as imagens
```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_IMAGES = "/content/drive/MyDrive/Leao"
```
O notebook varre todas as subpastas dentro de `Leao`, copia as imagens (`.jpg`, `.jpeg`, `.png`) para `data/<nome_da_cena>/input`, prefixando o nome do arquivo com o nome da subpasta para evitar colisão de nomes.

### Passo 4 — Instalar e rodar o COLMAP (Structure-from-Motion)
```bash
apt-get install -y colmap
```
Em seguida, três etapas do COLMAP são executadas sobre as imagens da pasta `input`:
1. `colmap feature_extractor` — extrai pontos-chave (features) de cada imagem.
2. `colmap exhaustive_matcher` — casa os pontos-chave entre todas as imagens.
3. `colmap mapper` — reconstrói a estrutura 3D esparsa e estima as poses de câmera (gera a pasta `sparse/0`).
4. `colmap image_undistorter` — gera as imagens sem distorção (`images/`), usadas no treino.

> Faça backup do resultado do COLMAP no Drive assim que terminar (o notebook faz isso com `shutil.copytree`), pois essa etapa é demorada e não deve ser refeita a cada reinício do runtime do Colab.

### Passo 5 — Segmentação e rastreamento de objetos (DEVA + SAM)
```bash
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path <scene_path>/images \
  --temporal_setting semionline \
  --size 480 \
  --sam_variant original \
  --SAM_NUM_POINTS_PER_SIDE 32 \
  --SAM_NUM_POINTS_PER_BATCH 32 \
  --output <scene_path>/deva_output
```
Isso gera máscaras de segmentação consistentes entre os frames (pasta `Annotations`), identificando cada objeto por um ID único codificado em RGB. Em seguida o notebook decodifica esses IDs, remapeia para classes sequenciais (`1..N`) e salva máscaras single-channel em `object_mask/`.

### Passo 6 — Treinar o modelo de Gaussian Splatting
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  -s data/<nome_da_cena> \
  -m output/<nome_da_cena> \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --densify_from_iter 2000 \
  --densification_interval 500 \
  --densify_grad_threshold 0.0002 \
  --resolution 1 \
  --checkpoint_iterations 5000 10000 20000 30000
```
Antes de treinar, é necessário gerar/atualizar o `config.json` na raiz do projeto com o número de classes (`num_classes`) obtido no passo de segmentação.

> Faça backup da pasta `output/<nome_da_cena>` no Drive ao final do treino — os checkpoints são grandes e o runtime do Colab pode ser reciclado.

### Passo 7 — Remover um objeto da cena 3D
1. Defina no `config/object_removal/<cena>_target.json` os IDs dos objetos a remover (`select_obj_id`), obtidos a partir da segmentação (passo 5).
2. Rode:
```bash
python edit_object_removal.py \
  -m output/<nome_da_cena> \
  --config_file config/object_removal/<cena>_target.json \
  --skip_test
```

### Passo 8 — Inpainting da região removida (LaMa)
1. Gere máscaras binárias (0/255) só dos objetos-alvo a partir de `object_mask/`.
2. Prepare a pasta de entrada do LaMa (`lama/LaMa_test_images/<cena>`) copiando imagem + máscara (`*_mask.png`) e movendo as máscaras para uma subpasta `label/`.
3. Baixe o modelo pré-treinado **big-lama**:
```bash
curl -L --http1.1 -o big-lama.zip https://huggingface.co/smartywu/big-lama/...
unzip -q big-lama.zip
```
4. Rode o LaMa:
```bash
python bin/predict.py \
  model.path=$(pwd)/big-lama \
  indir=$(pwd)/LaMa_test_images/<cena> \
  outdir=$(pwd)/output/<cena> \
  dataset.img_suffix=.png
```
5. Copie as imagens inpaintadas geradas para `data/<cena>/images_inpaint_unseen`.
6. Ajuste o `config/object_inpaint/<cena>.json` (classes, IDs-alvo, `finetune_iteration`, etc.) e rode:
```bash
python edit_object_inpaint.py \
  -m output/<cena> \
  --config_file config/object_inpaint/<cena>.json \
  --skip_test \
  -s data/<cena> \
  --images images
```
Essa etapa faz um *fine-tuning* do modelo 3D usando as imagens inpaintadas como referência para preencher a região do objeto removido de forma consistente em 3D.

### Passo 9 — Avaliar os resultados
O notebook calcula **PSNR**, **SSIM** e **LPIPS** comparando:
- o render do modelo original vs. as imagens reais;
- o render após a remoção (avaliado só na região *fora* do objeto removido);
- o render após o inpainting.

Bibliotecas usadas: `scikit-image` (PSNR/SSIM) e `lpips` (percepção via rede VGG).

### Passo 10 — Backups
Ao longo de todo o processo, é recomendado copiar periodicamente para o Google Drive:
- `data/<cena>` (imagens, máscaras, dados do COLMAP);
- `output/<cena>` (checkpoints, point clouds, renders);
- `deva_output` (máscaras de segmentação).

Isso evita perda de trabalho caso o runtime do Colab seja desconectado/reciclado (comum em sessões gratuitas).

---

## 3. Ajustes/patches aplicados no código (necessários para rodar)

O notebook aplica alguns patches manuais em arquivos do repositório para corrigir incompatibilidades de versão:
- `simple_knn.cu`: adiciona `#include <cfloat>`.
- `deva/inference/...`: separa `BoxAnnotator` e `LabelAnnotator` (API nova do `supervision`).
- `saicinpainting/training/data/aug.py` (LaMa): reescreve classes `IAAAffine2`/`IAAPerspective2` para compatibilidade com `albumentations` atual.
- `saicinpainting/training/modules/fake_fakes.py`: protege o `import` do `kornia` com `try/except`.
- `saicinpainting/training/trainers/__init__.py`: adiciona `weights_only=False` no `torch.load` (exigido em versões recentes do PyTorch).
- `utils/camera_utils.py` e `utils/loss_utils.py`: ajustes de shape/resize das máscaras de objeto.
- `edit_object_inpaint.py`: reescreve o laço principal de fine-tuning para suportar redimensionamento de máscara, uso das imagens inpaintadas do LaMa como *ground truth* e um `dataset_source` extra na assinatura da função.

Esses patches já estão automatizados nas células do notebook — não é necessário editá-los manualmente, apenas rodar as células na ordem correta.

---

## 4. Observações finais
- Cada etapa pesada (COLMAP, treino, DEVA, LaMa, fine-tuning de inpaint) é demorada; rode uma célula por vez e confira os prints de confirmação antes de avançar.
- Os erros de `FFMPEG`/`mpeg4` ao gerar vídeos dos renders (`dimensions too large for MPEG-4`) são apenas relacionados à exportação de vídeo de pré-visualização e não impedem o restante do pipeline.
- Ajuste `SCENE_NAME`/`DRIVE_ROOT` no notebook caso queira usar um nome de pasta diferente de `Leao` ou `meu_objeto`.
