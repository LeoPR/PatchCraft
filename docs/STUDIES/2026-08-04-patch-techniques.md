# Estudo 2026-08-04: o que falta para uma biblioteca acadêmica completa de patches

Diário de pesquisa. Formato: cada entrada tem data, hipótese/observação,
críticas (minhas = Leo, do assistente = marcadas **IA**), e tasks que saem
dela. Tasks só migram para `src/` depois de passarem pelo funil do lab
(hipótese → script em `lab/` → teste em `tests/` → ADR se virar API).

> **Tensão de governança registrada (IA).** O ROADMAP declara o core
> "feature-complete" e gateia qualquer símbolo novo num consumidor real. Este
> estudo propõe expansão ampla. Minha recomendação: manter o gate para o
> **core** (`patchcraft`) e direcionar tudo daqui para **pacotes auxiliares**
> (`patchcraft-quant` já previsto no ROADMAP; talvez `patchcraft-ops` ou
> similar). O estudo não altera o gate; quem decide é o Leo.

## 1. O patch em si (o patch como objeto de estudo)

### 1.1 O que já existe
- Extração regular: `extract`/`Patchify` (unfold, stride, dilation), geometria
  pré-voo (`num_patches`, `tilings`), pareamento LR/HR (`pair`).
- Métricas por patch (`patch_metrics`, `per_patch_mse/psnr`).

### 1.2 O que falta (levantamento acadêmico)

**Amostragem e geometria da grade**
- *Random/crop sampling*: extrair N patches em posições aleatórias com seed
  (padrão em treinamento de SR/denoising: RCAN, EDSR usam random crop).
- *Padding modes* para grade não-divisível: `constant`/`reflect`/`replicate`
  antes de extrair, com metadado para des-paddar no round-trip. Hoje a guarda
  de cobertura (0.2.1) rejeita grade truncada; padding é a resposta canônica.
- *Grades adaptativas por conteúdo*: quadtree patching (patch maior em região
  lisa, menor em textura), saliency-guided, superpixels (SLIC). Acadêmico mas
  foge do "grid regular" que torna `fold` possível.
- *Jittered grids*: grade regular com deslocamento aleatório por patch
  (data augmentation clássico em patch-based texture synthesis).

**Transformações do patch**
- *Rotação com reversão* (pedido do Leo, ver §4).
- *Flips* (h/v) triviais com reversão exata.
- *Domínio da frequência por patch*: DCT-II (JPEG, codecs), FFT, wavelet
  (Haar/Daubechies por bloco). Base de compressão e de análise de textura.
- *Descritores*: histograma, entropia, estatísticas de gradiente, LBP.
  Usados em patch matching e quality assessment.

**Matching e agrupamento**
- *Patch matching* (busca de vizinhos mais próximos entre patches): base de
  NLM, BM3D, PatchMatch, inpainting, texture synthesis (Efros-Leung).
- *Self-similarity grouping*: empilhar os k patches mais similares (BM3D faz
  denoising colaborativo nesse stack).

## 2. Patch como ferramenta de decomposição/reconstrução

### 2.1 O que já existe
- `reconstruct` (fold + count map, bit-exato sob a regra do count
  map), `stitch` (blend com janelas
  uniform/hann/gaussian).

### 2.2 O que falta

**Reconstrução**
- *Round-trip com padding*: decompõe imagem de tamanho arbitrário via pad →
  extract → ... → reconstruct → crop. Fecha o par com §1.2 padding.
- *Gradient-domain blending* (Poisson/Pérez 2003): costura patches modificados
  impondo gradientes em vez de valores; superior a janelas quando o conteúdo
  modificado é estruturado (edição, inpainting).
- *Reconstrução multi-escala*: pirâmide Laplaciana, que decompõe em bandas,
  patchifica por banda, reconstrói bottom-up. Base de SR e de blending de
  alta qualidade (Burt-Adelson).
- *OLA generalizado*: weighted overlap-add com janelas customizadas pelo
  caller (hoje só 3 kernels nomeados; §9.9 lista como fora de escopo).

**Técnicas clássicas baseadas em patch (literatura)**
- *KSVD / sparse coding*: dicionário aprendido sobre patches; reconstrução
  por combinação esparsa (Aharon-Elad 2006; base do SR de Yang et al.).
- *NLM/BM3D*: denoising por média não-local / filtragem colaborativa 3D de
  stacks de patches similares.
- *EPLL* (Zoran-Weiss 2011): prior de patches via GMM; restauração iterativa.
- *Shift-and-add / drizzle*: SR por registro sub-pixel de múltiplas grades
  deslocadas, o que conecta com `pair` e grades jittered.
- *MAE-style masking*: esconder fração dos patches e reconstruir os ausentes
  (consumer-side é o modelo; a lib forneceria mask/unmask exato).

## 3. Quantização (pedido do Leo)

Escopo natural de um `patchcraft-quant` (ROADMAP já prevê):
- *Uniform scalar*: round-to-nearest com step Δ, com e sem deadzone (JPEG,
  H.26x). Reversível só quando Δ=1.
- *Lloyd-Max / ótima por histograma*: níveis que minimizam MSE para o
  histograma do patch.
- *Vector quantization*: codebook por K-Means sobre patches (Linde-Buzo-Gray);
  base de compressão e de tokenização (VQ-VAE sem a rede).
- *Dithering*: Floyd-Steinberg e dither ordenado (Bayer) antes de quantizar
  (preserva textura percebida).
- *Bit-depth reduction*: 8→4/2/1 bits com métricas de erro (`patch_metrics`
  já mede).
- *Quantização por bloco estilo JPEG*: DCT por patch 8×8 + matriz de
  quantização, o que fecha o loop com §1.2 DCT.

## 4. Rotação com reversão (pedido do Leo)

- *Exata (bit-exact reversível)*: múltiplos de 90° (`torch.rot90`, k∈{0,1,2,3})
  e flips. Reversão trivial: rot90 com k' = (−k) mod 4. Aplicável ao patch e
  à imagem. **Sem risco de contrato.**
- *Arbitrária (aproximada)*: ângulo qualquer via `affine_grid`/`grid_sample`.
  Reversão NÃO é exata (interpolação dupla degrada); o contrato tem que ser
  "reversão aproximada com erro medido" (PSNR reportado). Opções de mitigação:
  guardar o patch original no metadado (reversão trivial mas trapaceia),
  ou quantificar o erro por round-trip no lab.

**Crítica (IA).** Rotação arbitrária "com reversão" não existe de forma exata
fora dos múltiplos de 90°: qualquer interpolação é com perda. Recomendo
entregar os dois níveis explicitamente nomeados (`rot90` exato no core-candidato;
`rotate` arbitrário no auxiliar com o erro documentado), para não prometer o
que a matemática não entrega. Ver task T7.

## 5. Críticas abertas

- **(IA) Scope creep.** §1.2 e §2.2 cobrem ~20 anos de literatura patch-based.
  Implementar tudo é um projeto de anos. Sugestão: ordenar por "o que o lab
  consegue validar em uma sessão" e pelo consumidor PatchSR (quando nascer).
  KSVD/EPLL/BM3D são bibliotecas inteiras; talvez nunca devam entrar, porque o
  estudo os registra como *referência*, não como compromisso.
- **(IA) Onde cada coisa mora.** Core: só o que é primitivo geométrico com
  contrato bit-exato (padding round-trip, rot90, flips, OLA com janela do
  caller?). Auxiliar: quantização, DCT/FFT/wavelet, descritores, matching,
  rotação arbitrária. Consumer-side: KSVD, EPLL, BM3D, MAE.
- **(IA) Reversão como princípio de design.** O pedido de "rotação com
  reversão" generaliza: toda transformação nova deveria declarar sua classe de
  reversibilidade (bit-exata / aproximada com erro medido / irreversível) no
  contrato, como §9 já faz com rejeições. Proposta: ADR 0003 formalizando as
  três classes antes de qualquer implementação nova.
- **(Leo)** *(espaço para as críticas do Leo na revisão deste estudo)*

## 6. Tasks (funil lab → teste → ADR)

Critério de promoção: hipótese escrita → script em `lab/YYYY-MM-DD-*.py` com
saída em `Z:\outputs\patchcraft\` → se o resultado sustenta a hipótese, vira
teste em `tests/` → se vira API, ADR. Tasks na ordem sugerida (ROI):

| ID | Task | Hipótese a testar no lab | Esforço |
|----|------|--------------------------|---------|
| T1 | ADR 0003: classes de reversibilidade | n/a (decisão de design) | P |
| T2 | Pad → extract → reconstruct → crop bit-exato | round-trip exato em H,W arbitrários (ex: 27×33) com reflect/constant | P |
| T3 | rot90/flip com reversão exata | k∈{0..3} e flips invertem bit-exato em patch e imagem | P |
| T4 | Quantização uniforme + deadzone | erro medido por `patch_metrics` bate com Δ²/12 teórico | M |
| T5 | LBG/K-Means vector quantization de patches | PSNR vs tamanho do codebook em MNIST; curva plausível | M |
| T6 | DCT por patch 8×8 + quantização JPEG | round-trip com Q=50 reproduz PSNR conhecido da literatura | M |
| T7 | Rotação arbitrária + reversão | medir PSNR de round-trip por ângulo; confirmar que degrada (crítica §4) e quantificar | M |
| T8 | Dithering (FS/Bayer) antes de quantizar | erro percebido (PSNR/SSIM caller-side) melhora vs quantização seca | M |
| T9 | Patch matching (k-NN exato por L2) | patches de textura repetida se agrupam; base para NLM | G |
| T10 | OLA com janela do caller | igualdade com `stitch` nos 3 kernels atuais | P |
| T11 | Grades jittered com seed | patch count estável, cobertura verificável, reproduzível | M |
| T12 | Poisson blending de patches | costura invisível vs `stitch` hann em patch modificado sintético | G |
| T13 | Pirâmide Laplaciana + patchify por banda | reconstrução exata da pirâmide sem patches primeiro | G |
| T14 | Random crop sampling com seed | reprodutibilidade e cobertura estatística da imagem | P |

**Crítica (IA) sobre as tasks.** T2, T3, T10, T14 são as únicas candidatas ao
core (primitivas geométricas com reversão exata). T4–T8 formam o
`patchcraft-quant`. T7 é deliberadamente uma task de *falsificação*: o lab
existe para medir o quão ruim é a reversão aproximada antes de prometê-la.

## 7. Log do diário

- **2026-08-04 (IA):** estudo criado a pedido do Leo após o fechamento do
  backlog 0.2.1. `uv publish` verificado funcional (uv 0.11.11; cuidado:
  `dist/` contém 0.2.0 e 0.2.1, publicar com glob explícito). Estrutura do
  documento e tasks T1–T14 propostas. Aguardando críticas do Leo.
- **2026-08-04 (IA, segunda entrada):** interlúdio de infraestrutura fechado:
  0.2.1 publicada manualmente via `uv publish`, Trusted Publishing da pipeline
  diagnosticado e corrigido (publisher do PyPI desalinhado dos 4 campos;
  corrigido no registro do publisher, run `30887862622` verde, GitHub Release
  criada). ROADMAP M8 marcado como verificado. **Crítica (IA):** a depuração
  consumiu tempo por falta de acesso ao log do job (API exige admin; `gh`
  desautenticado). Lição registrada: autenticar `gh` na máquina dev ou aceitar
  que diagnóstico de CI passa por steps verbosos commitados. Execução do
  estudo retomada: T1 (ADR 0003) e T3/T7 (rotação) iniciados.
- **2026-08-04 (IA, terceira entrada):** T1 entregue como **ADR 0003
  (Proposed)**: três classes de reversibilidade (R1 bit-exata, R2 aproximada
  medida, R3 irreversível), com regra de naming (inversos R2 nunca usam
  vocabulário de inversão exata) e core restrito a R1. Aguarda revisão do Leo
  para virar Accepted. **T3 confirmado no lab**: rot90 k=0..3 e flips
  invertem bit-exato em imagem e em patch stack, float32/64 e uint8
  (`lab/2026-08-04-rotation-reversibility.py`). **T7 medido**: round-trip de
  ângulo arbitrário via `grid_sample` bilinear teto de ~24 dB entre 5° e 45°
  (max_abs ~0.37 em escala [0,1], ou seja, erro visível), e mesmo 90.0° pelo caminho
  interpolado dá 132.8 dB onde `rot90` dá infinito. **Conclusão dos dados:**
  a hipótese da crítica §4 estava certa, porque a rotação arbitrária é R2 no melhor
  caso, e a separação rot90 (R1, candidata a core) de rotate (R2, auxiliar)
  é obrigatória, não estilística. **Crítica (IA):** a curva de erro não é
  monotônica no ângulo (1° dá 32 dB, 5° cai para 24 dB e achata); vale uma
  entrada futura investigando se `bicubic` ou `padding_mode="reflection"`
  melhora o teto antes de qualquer contrato R2 ser escrito.
