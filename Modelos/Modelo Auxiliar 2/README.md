# SPIn-NeRF no Colab (remoção e inpainting de objetos em 3D via NeRF)

Notebook baseado no repositório oficial **[spinnerf3d/SPIn-NeRF](https://github.com/spinnerf3d/SPIn-NeRF)**. Reconstrói uma cena com NeRF (Neural Radiance Fields), segmenta um objeto multivista (DEVA + Grounded-SAM), remove esse objeto e faz o *inpainting* tanto da profundidade quanto do RGB (via LaMa), retreinando um NeRF final "sem o objeto".

---

## 1. Requisitos

### 1.1 Conta e ambiente
- Conta Google (Colab +, opcionalmente, Drive para salvar resultados/métricas).
- Runtime do Colab com **GPU habilitada**: `Ambiente de execução > Alterar tipo de ambiente de execução > GPU`.
- Este notebook é o mais pesado dos três: além do treino do NeRF (10.001 iterações), ele compila o `tiny-cuda-nn` e roda **duas passagens de LaMa** (profundidade e RGB) além do DEVA/Grounded-SAM. Reserve bastante tempo de sessão.
- Espaço em disco: pesos do DEVA, do Grounded-SAM e do big-lama (~360 MB) são baixados na sessão.

### 1.2 Imagens próprias — pasta "Leao_1"
Aqui o caminho usado em **todo o notebook** é fixo na variável `SCENE_DIR = "/content/data"`, e as fotos originais devem entrar em:

```
/content/data/input/
```

**Passo manual obrigatório:**
1. Rode a célula 1 (`!git clone ... SPIn-NeRF`) e a célula que define `SCENE_DIR = "/content/data"` e cria `INPUT_DIR` (ela já executa `os.makedirs(INPUT_DIR, exist_ok=True)`), ou crie a pasta você mesmo:
   ```
   !mkdir -p /content/data/input
   ```
2. **Copie todas as imagens da pasta local `Leao_1`** para dentro de `/content/data/input/` — soltas, sem subpastas.
   - Mais simples: pelo painel de arquivos do Colab (ícone de pasta na lateral esquerda), navegue até `data/input` e arraste os arquivos de `Leao_1`.
   - Alternativa via Drive: monte o Drive e copie de lá com `shutil.copy`.
3. Confirme que são `.jpg`, `.jpeg` ou `.png`.

> Observação sobre a célula de organização automática (logo após criar `INPUT_DIR`): ela existe para o caso de as fotos virem **já divididas em subpastas** dentro de `input/` (por exemplo, vindas do Drive organizadas por ângulo/cena) — nesse caso ela achata tudo em arquivos soltos, prefixando o nome com o da subpasta. Se você já colocar as fotos de `Leao_1` soltas direto em `input/` (como recomendado acima), essa célula simplesmente não encontra subpastas e não faz nada — pode rodar sem problema, é idempotente.

> Atenção: arquivos enviados direto pelo painel do Colab (sem passar pelo Drive) só existem na sessão atual — se o runtime desconectar, é preciso subir de novo.

### 1.3 Bibliotecas e ferramentas instaladas pelo notebook
- **PyTorch/torchvision/torchaudio** (build `cu121`, instalado antes do resto para não conflitar com o ambiente do Colab)
- `imageio`, `imageio-ffmpeg`, `matplotlib`, `configargparse`, `tensorboard`, `opencv-python`, `mediapy`, `numpy`, `lpips`, `scikit-video`, `ffmpeg-python`, `pytorch-ignite`
- `CLIP` (`git+https://github.com/openai/CLIP.git`)
- Dependências do LaMa: `pyyaml`, `tqdm`, `easydict`, `scikit-image`, `scikit-learn`, `tensorflow`, `joblib`, `pandas`, `albumentations`, `hydra-core`, `pytorch-lightning`, `tabulate`, `kornia`, `webdataset`
- **COLMAP** (via `apt-get install colmap`)
- **DEVA** (`Tracking-Anything-with-DEVA`, instalado em modo editável + pesos baixados via `download_models.sh`)
- **Grounded-Segment-Anything** (Segment Anything + GroundingDINO), com `BUILD_WITH_CUDA=True` e `CUDA_HOME=/usr/local/cuda`
- **tiny-cuda-nn** (`git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`) — necessário porque o `run_nerf.py` importa esse módulo incondicionalmente
- `build-essential`, `ninja`, `python3-tk` (via `apt-get`, dependências de build/runtime)
- Checkpoint **big-lama** (baixado de `huggingface.co/smartywu/big-lama`)
- Para avaliação de métricas: `lpips`, `scikit-image` (SSIM/PSNR), `pandas`

---

## 2. Passo a passo para rodar

### Passo 1 — Ambiente e clone
```bash
!nvidia-smi
%cd /content
!git clone https://github.com/spinnerf3d/SPIn-NeRF.git
%cd SPIn-NeRF
```

### Passo 2 — Instalar dependências
Instale primeiro o PyTorch compatível com a GPU do Colab, e só depois o restante (evita que o pip rebaixe o torch já instalado):
```bash
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q imageio imageio-ffmpeg matplotlib configargparse tensorboard \
    opencv-python mediapy numpy lpips scikit-video ffmpeg-python pytorch-ignite
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q pyyaml tqdm numpy easydict scikit-image scikit-learn opencv-python tensorflow joblib matplotlib pandas albumentations hydra-core pytorch-lightning tabulate kornia webdataset
```
> Se algum pacote falhar por incompatibilidade, rode a célula de novo — geralmente resolve na segunda tentativa.

### Passo 3 — Preparar o dataset e rodar o COLMAP
1. Crie `/content/data/input/` e coloque as fotos de `Leao_1` lá (seção 1.2).
2. Instale o COLMAP:
   ```bash
   !apt-get install -y colmap
   ```
3. Rode as três etapas do COLMAP sobre `SCENE_DIR = "/content/data"`:
   ```bash
   !colmap feature_extractor \
       --database_path /content/data/database.db \
       --image_path /content/data/input \
       --ImageReader.camera_model SIMPLE_PINHOLE \
       --SiftExtraction.use_gpu 0

   !colmap exhaustive_matcher \
       --database_path /content/data/database.db \
       --SiftMatching.use_gpu 0

   !mkdir -p /content/data/sparse
   !colmap mapper \
       --database_path /content/data/database.db \
       --image_path /content/data/input \
       --output_path /content/data/sparse
   ```
4. Mova as imagens de `data/input` para `data/images`:
   ```python
   import os, shutil
   input_dir = "/content/data/input"
   images_dir = "/content/data/images"
   os.makedirs(images_dir, exist_ok=True)
   for fname in os.listdir(input_dir):
       if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
           shutil.move(os.path.join(input_dir, fname), os.path.join(images_dir, fname))
   ```
5. Gere o `poses_bounds.npy` (formato de câmeras esperado pelo NeRF/LLFF):
   ```bash
   %cd /content/SPIn-NeRF
   !python imgs2poses.py --data_dir /content/data
   ```
   O notebook confere automaticamente se o arquivo foi criado.

### Passo 4 — Segmentação multivista (DEVA + Grounded-SAM)
```bash
%cd /content
!git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git
%cd Tracking-Anything-with-DEVA
!pip install -e .
!bash scripts/download_models.sh

!git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
%cd Grounded-Segment-Anything
# BUILD_WITH_CUDA=True e CUDA_HOME configurados no notebook
!python -m pip install -e segment_anything
!python -m pip install -e GroundingDINO
%cd /content
```
Há um patch automático em `eval_args.py` do DEVA (compatibilidade com a API nova do pacote `supervision`) — já vem pronto no notebook e é idempotente.

Rode a segmentação:
```bash
%cd /content/Tracking-Anything-with-DEVA
!python demo/demo_automatic.py \
    --chunk_size 4 \
    --img_path /content/data/images \
    --temporal_setting semionline \
    --size 480 \
    --sam_variant original \
    --SAM_NUM_POINTS_PER_SIDE 32 \
    --SAM_NUM_POINTS_PER_BATCH 32 \
    --output /content/data/deva_output
```
Depois, inspecione os IDs de objeto detectados (`pred.json`) e visualize as máscaras sobre uma imagem de exemplo para identificar visualmente o(s) ID(s) do objeto que você quer remover.

### Passo 5 — Definir o objeto-alvo e gerar máscaras
Anote os IDs do objeto (números nos rótulos da visualização) e ajuste:
```python
TARGET_IDS = [15351475, 14802278]  # <-- troque pelos IDs do seu objeto
factor = 2
```
O notebook então gera `images_2/` (imagens reduzidas por um fator de downscale) e as máscaras binárias correspondentes em `images_2/label/`.

### Passo 6 — Checkpoint do LaMa e pré-requisitos do NeRF
```bash
%cd /content/SPIn-NeRF/lama
!curl -L -o big-lama.zip 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip'
!unzip -oq big-lama.zip -d .
```
Instale o `tiny-cuda-nn` (detectando automaticamente a *compute capability* da GPU) e o `python3-tk`:
```bash
!apt-get -qq install -y build-essential python3-tk
!pip install -q ninja
# TCNN_CUDA_ARCHITECTURES é setado automaticamente a partir do nvidia-smi
!pip install -q git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
O notebook também remove ocorrências de `ignoregamma=True` no código do SPIn-NeRF (incompatível com versões recentes do `imageio`).

### Passo 7 — Inpainting da profundidade (Passo B)
Prepara os pares imagem+máscara em `LaMa_test_images/` e roda:
```bash
%cd /content/SPIn-NeRF/lama
!python bin/predict.py refine=True model.path=$(pwd)/big-lama \
    indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output
```
Antes disso, o notebook aplica patches de compatibilidade no código do LaMa (todos idempotentes):
- `aug.py`: recria o shim `DualIAATransform` (removido de versões recentes do `albumentations`) e corrige o import de `to_tuple`.
- `fake_fakes.py`: corrige o import de `SamplePadding` do `kornia`.
- `trainers/__init__.py`: adiciona `weights_only=False` no `torch.load` (exigido por versões recentes do PyTorch).

O resultado (`output/label/*.png`) é copiado para `images_2/depth/`.

### Passo 8 — Inpainting das imagens RGB (Passo C)
Mesmo padrão do passo anterior, mas aplicado às imagens RGB (não à profundidade). O notebook prepara novamente `LaMa_test_images/` (imagens + máscaras `label/`) e roda o LaMa. O resultado é copiado para `images_2/lama_images/`.

> Há várias células equivalentes/alternativas nessa seção do notebook (diferentes tentativas de preparar os arquivos para o LaMa) — rode a **última** versão de cada bloco (a mais robusta, que testa o dataset antes de chamar `predict.py`) caso as anteriores derem erro; elas foram mantidas no notebook como histórico de depuração.

### Passo 9 — Fitting final do NeRF "inpainted" (Passo D)
Confere a consistência dos pontos de profundidade do COLMAP com as máscaras do objeto e então treina o NeRF final:
```bash
%cd /content/SPIn-NeRF
!python DS_NeRF/run_nerf.py --config DS_NeRF/configs/config.txt \
    --i_feat 200 --lpips \
    --i_weight 1000000000000 --i_video 1000 --N_iters 10001 \
    --expname <nome_da_cena> --datadir /content/data --N_gt 0 --factor 2
```
Troque `<nome_da_cena>` pelo valor de `SCENE_NAME` definido lá no Passo 3 (ex.: `leao`).

### Passo 10 — Visualizar o vídeo renderizado
```python
from IPython.display import Video
import glob
videos = sorted(glob.glob(f"logs/<nome_da_cena>/*_test.mp4"))
display(Video(videos[-1], embed=True, width=640))
```
E, se quiser, baixe o vídeo com `google.colab.files.download`.

### Passo 11 — (Opcional) Avaliação de métricas (PSNR/SSIM/LPIPS)
Depois do treino, é possível rodar novamente `run_nerf.py` com `--N_gt` apontando para os índices de teste, extrair os frames renderizados e comparar com as imagens originais e as máscaras, calculando PSNR, SSIM e LPIPS por imagem. Ajuste `render_dir`, `i_test_indices` etc. conforme o log do treino (procure por "TEST views are" no output do Passo D).

---

## 3. Diferenças em relação aos outros dois notebooks
- **Gaussian Grouping** (primeiro README): pipeline de *Gaussian Splatting* com segmentação/remoção/inpainting, saída é uma point cloud 3D de gaussianas.
- **3D Gaussian Splatting puro** (segundo README): só reconstrução 3D via Gaussian Splatting, sem segmentação/remoção/inpainting.
- **SPIn-NeRF (este)**: usa **NeRF** (não Gaussian Splatting) como representação 3D, e faz o processo completo de segmentação + remoção + inpainting **de profundidade e de RGB**, retreinando o NeRF do zero sobre os dados já editados — é o mais lento e o mais complexo dos três, mas também o único com esse fluxo de "remover objeto e reconstruir a cena preenchida" nativo do NeRF.
- Aqui a pasta de imagens originais fica em `/content/data/input/` (achatada e depois movida para `/content/data/images/`), igual ao do 3D Gaussian Splatting, mas diferente do Gaussian Grouping (`data/<cena>/input`).

## 4. Observações finais
- Rode as células em ordem — há muitas dependências sequenciais (COLMAP → poses → segmentação → máscaras → LaMa depth → LaMa RGB → NeRF final).
- Esse notebook tem trechos de depuração com múltiplas tentativas para a mesma etapa (principalmente na preparação de arquivos para o LaMa) — isso é normal, mantido como histórico; use sempre a célula mais recente/robusta de cada bloco.
- Faça backup de `/content/data` e de `logs/<nome_da_cena>` no Google Drive periodicamente — o treino do NeRF final (10.001 iterações) é demorado e o runtime do Colab pode reciclar.
