# Comparação de Pipelines de Reconstrução 3D — Projeto Leão

## Introdução

Este projeto avalia e compara três abordagens diferentes para reconstrução 3D e edição de cenas (remoção/inpainting de objetos) a partir de um mesmo conjunto de fotos — a estátua do leão fotografada de múltiplos ângulos (pastas `Leao` / `Leao_1`). Cada abordagem é implementada em um notebook Google Colab independente, documentado em seu próprio README:

| # | Modelo / Pipeline | Representação 3D | Faz remoção/inpainting de objeto? | README |
|---|---|---|---|---|
| 1 | **Gaussian Grouping** | 3D Gaussian Splatting + segmentação (SAM/DEVA) | Sim — remoção + inpainting 2D (LaMa) com fine-tuning 3D | `README.md` |
| 2 | **3D Gaussian Splatting** (oficial, graphdeco-inria) | 3D Gaussian Splatting | Não — reconstrução "pura" | `README_gaussian_splatting.md` |
| 3 | **SPIn-NeRF** | NeRF (Neural Radiance Fields) + segmentação (DEVA/Grounded-SAM) | Sim — remoção + inpainting de profundidade e RGB (LaMa) com retreino do NeRF | `README_spin_nerf.md` |

A ideia central é rodar a **mesma cena** (as mesmas fotos do leão) nos três pipelines e, ao final, **comparar os resultados entre si**, tanto:

- **Visualmente**, através das imagens/vídeos renderizados por cada modelo a partir dos mesmos pontos de vista (mesmas câmeras de teste); quanto
- **Quantitativamente**, através de métricas objetivas de qualidade de imagem calculadas entre a renderização de cada modelo e a foto real correspondente:
  - **PSNR** (Peak Signal-to-Noise Ratio) — mede o erro pixel a pixel entre a imagem renderizada e a original; quanto maior, mais fiel.
  - **SSIM** (Structural Similarity Index) — mede a similaridade estrutural (contraste, luminância, textura) entre as duas imagens; varia de 0 a 1, quanto maior, melhor.
  - **LPIPS** (Learned Perceptual Image Patch Similarity) — mede a diferença perceptual usando uma rede neural (VGG/AlexNet) treinada para se aproximar da percepção humana; quanto menor, mais parecidas as imagens parecem "aos olhos humanos".

Essas três métricas já são calculadas individualmente em cada um dos notebooks (nas seções de avaliação de cada README), então o próximo passo é consolidar os resultados dos três pipelines lado a lado — por exemplo, numa tabela comparativa (`modelo x PSNR x SSIM x LPIPS`) e numa grade de imagens (mesma vista, lado a lado: foto original / render Gaussian Grouping / render Gaussian Splatting / render SPIn-NeRF) — para responder perguntas como:

- Qual pipeline reconstrói a cena com mais fidelidade visual?
- Qual lida melhor com a remoção do objeto e o preenchimento (inpainting) da região removida?
- Existe um trade-off entre qualidade e tempo/custo computacional entre Gaussian Splatting e NeRF?

> Observação: essa comparação final ainda não está automatizada — cada notebook, por enquanto, calcula suas próprias métricas de forma isolada. A consolidação entre os três (tabela + grade de imagens) é um passo futuro do projeto.

---

Os detalhes de requisitos, instalação e execução de cada pipeline estão nos respectivos READMEs listados na tabela acima.
