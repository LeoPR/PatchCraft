# Fase 3 — G1: predicado bit-exato e a suíte que pode derrubá-lo (alvo: 0.5.0)

Data: 2026-09-01. Status: aprovado pelo fluxo (auto mode); decisões registradas abaixo.

Contexto: `docs/FOCO-1.0.md` define o que o 1.0 congela e lista seis bloqueadores.
B5 já fechou (0.2.1/0.3.0). B4 é escopo de G2. Esta fase é o **G1 do FOCO**: o
predicado correto escrito em todo lugar (B1 restante), uma suíte capaz de
falsificá-lo (B2), a superfície pública fixada (B3), a enumeração sem lixo e a
assimetria de guardas resolvida (B6), mais dois itens baratos do mesmo grupo
(ramo sem zstandard; `hand.py`×`pc.py` como teste). O FOCO dizia "G1 sai como
0.3.0"; 0.3.0 e 0.4.0 já existem, então **G1 sai como 0.5.0** — a correção do
predicado é retratação pública e merece nota de release própria, e a mudança na
enumeração de `tilings` muda comportamento.

Auditoria de partida (2026-09-01, head 0.4.0): o predicado de potência-de-dois
já está correto em ADR 0003, `reconstruct.py:34-42`, THEORY §2/§9, READMEs e
GUIDE. Segue errado ou sem qualificação em: `docs/SCOPE.md:229-230`,
`README.md:86`, `README.pt-BR.md:86`, `src/patchcraft/stitch.py:3` e `:123`, e
`docs/THEORY.md:157` (fecha com a justificativa de tolerância frouxa
"`allclose` on overlap" que B2 existe para matar). `docs/USAGE.md` tem claims
sem qualificação em `:88-91`, `:134`, `:149` — correção mínima aqui; a
regeneração completa do USAGE é G2.

## 1. Decisões

**D1 — B6/enumeração: corrigir agora.** `tilings((1,28,28), allow_overlap=True)`
emite hoje 100 specs, 27 deles degenerados: patch único (`nh == nw == 1`) com
`s < p` marcado `overlap=True`, onde não há com o que sobrepor
(`geometry.py:178-190`). Com um único patch o stride é inobservável e o spec é
duplicata semântica do tile exato `(p, p)/(p, p)`, sempre emitido para o mesmo
`p`. **Fix:** pular o ramo de overlap quando `nh == 1 and nw == 1`. A saída de
enumeração de `tilings`/`paired_tilings` é o item 5 do congelamento — mudar
depois do 1.0 é quebra, então é agora. `paired_tilings` herda o fix via
`tilings`. Nenhum teste atual depende da contagem 100 (verificado por grep).

**D2 — B6/trunca: NÃO mudar comportamento; documentar a assimetria.** Hoje
`extract` trunca em silêncio (130×130, p=32, s=32 → 16 patches, 2 linhas e 2
colunas descartadas) e `reconstruct` recusa grade que não cobre exatamente.
Manter assim, e escrever a assimetria no THEORY §9.1 (hoje está só no docstring
de `extract`, em THEORY §1 e em USAGE): "na ida, trunca é a política de borda
(ADR 0001); na volta, cobertura parcial é `ValueError`; quem precisa de
cobertura exata confere `num_patches`/`tilings` antes". Razão: reverter a
trunca do ADR 0001 quebraria o caso de uso principal (Patchify em pipeline de
treino sobre imagens de tamanho arbitrário) e o conjunto aceita/rejeita do §9
(item 3 do congelamento) na sua pior forma; a resposta para "minha imagem não
tem tiling exato" é o `pad` da 1.1 (T2, candidata nomeada no FOCO §3).

**D3 — `WeightKind` é conjunto aberto.** Novas janelas são adição compatível; o
consumidor não pode assumir exaustividade. A prosa da política de
compatibilidade entra no G2; aqui o teste fixa os três valores atuais
(`uniform`, `hann`, `gaussian`) como ponto de partida do conjunto.

**D4 — superfície pública = 20 nomes.** Os 19 do FOCO mais `accel_available`
(0.4.0). O congelamento do 1.0 será sobre os 20; B3 fixa agora.

**D5 — contrato numérico escrito na comparação em que vale.** Dentro do
predicado (todo valor do count map é potência de dois), `reconstruct` devolve
os mesmos bits: verificável por `torch.equal` para entradas sem NaN e por
comparação da view inteira (`view(torch.int32/int64)`) em qualquer caso —
`torch.equal` não é reflexivo em NaN mesmo quando os bits voltam idênticos.
Fora do predicado, o erro é limitado a 1 ULP. Essa frase já vive em
`reconstruct.py:34-42` e ADR 0003; B1 estende a forma qualificada aos pontos
restantes e a suíte passa a escrevê-la em código.

**D6 — `hand.py`×`pc.py` reimplementado, não recuperado.** Os scripts citados
no FOCO §2 ("o substituto fechável já está medido") não estão no git
(`lab/.gitignore` é `*`; scratchpads fora do repo não existem mais). O
substituto do consumer gate entra como `tests/test_reference.py`: uma
referência naive de extract+reconstruct (loops puros, sem `F.fold`, acumulação
no dtype de entrada) comparada bit a bit com os caminhos rápidos dentro do
predicado, e a ≤1 ULP fora dele.

## 2. Componentes

### 2.1 Helper auditado de dados — `tests/_rng.py`

```python
def rand_image(c: int, h: int, w: int, dtype: torch.dtype, seed: int) -> torch.Tensor
def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool
def exact_axes_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool
def count_map_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool
```

Regras do helper (FOCO §4/G1): aleatório (`torch.rand` com
`torch.Generator(device).manual_seed(seed)`), mantissa cheia, gerado **no dtype
alvo** — proibido derivar float64 de float32 (`.double()` preenche metade da
mantissa e round-trippa True onde `rand(dtype=float64)` dá False; medido no
FOCO §0). `bit_equal` compara `a.view(int_dtype) == b.view(int_dtype)` — NaN-safe
(quarto gerador de falso negativo). `exact_axes_pow2` é a receita barata do
FOCO §0 (por eixo, toda contagem de cobertura distinta é potência de dois,
O(H+W), sem `fold`); `count_map_pow2` é o predicado pelo count map real. Ambos
assumem cobertura exata; fora de cobertura exata não há round-trip.

### 2.2 Reescrita dos casos de round-trip existentes

Os treze casos de round-trip que hoje usam rampa inteira (`_ramp` /
`torch.arange`, gerador de falso negativo — ADR 0003 "Input values are a third
regime axis") passam a usar `rand_image`. A rampa continua onde o que se testa
é **ordem/posição** (row-major, count map com imagem de uns, mismatch de
grade) — lá o valor não importa. `test_reconstruct.py:78`: `rtol=1e-5` (três
ordens acima do erro medido de 2.4e-07) é substituído pela asserção de ≤1 ULP
via helper. Os testes de caracterização 0.3.0 contra `_fold_reference`
(`test_reconstruct_matches_fold_reference`) já usam `torch.randn` no dtype alvo
— ficam como estão.

### 2.3 Suíte falsificadora — `tests/test_exactness.py`

O ponto do grupo (FOCO §5): derivar o predicado da aritmética, depois enumerar
o espaço legal independentemente do predicado e rodar para quebrá-lo.

- **Enumerador do espaço legal**: H, W ∈ 4..24, p ∈ 2..9, strides de cobertura
  exata (incluindo s < p com `(H-p) % s == 0`), sem listas escritas à mão.
  Espaço completo ≈ 126 mil geometrias (FOCO §0).
- **Amostra seedada**: 256 geometrias × float32/float64 no modo default
  (segundos de CI); `PATCHCRAFT_SWEEP_FULL=1` roda o espaço completo localmente.
- **Metade positiva**: geometria dentro do predicado ⇒ `bit_equal` True para
  toda seed de um conjunto fixo.
- **Metade negativa**: geometria fora do predicado ⇒ **pelo menos uma de N=50
  seeds é inexata** E toda seed tem erro ≤ 1 ULP. Forma de conjunto, não de
  execução única: medido no FOCO, na geometria `(1,4,14)` p=(4,4) s=(1,1),
  `torch.equal` é True em 63 de 300 seeds float32 — exatidão fora do predicado
  é propriedade da amostra, não da geometria. Com N=50, a chance de falso
  verde é (63/300)^50 ≈ 0 — sem flake.
- **Receita × predicado**: `exact_axes_pow2 == count_map_pow2` em toda a
  amostra (zero divergências, como no FOCO §0).
- **Caso NaN**: imagem com NaNs dentro do predicado ⇒ `bit_equal` True mesmo
  com `torch.equal` False (documenta D5 em código).

### 2.4 Referência naive — `tests/test_reference.py`

Substituto fechável do consumer gate (FOCO §2, "o que não bloqueia").
Implementação de extract+reconstruct com loops puros sobre uma grade de
geometrias pequenas: extract fatia pixel a pixel; reconstruct acumula soma e
contagem por pixel no dtype de entrada e divide. Comparação com
`patchcraft.extract`/`reconstruct`: bit-idêntica dentro do predicado (qualquer
ordem de soma de k parcelas idênticas com k potência de dois é exata), ≤1 ULP
fora. Roda também contra o caminho accel quando disponível (a grade de
equivalência accel×puro já existe em `tests/test_accel.py`; aqui basta não
pular o teste quando o acelerador está ativo).

### 2.5 Superfície pública — `tests/test_public_api.py`

- `patchcraft.__all__` igual ao conjunto exato dos 20 nomes (D4), e cada nome
  importável do topo.
- `inspect.signature` fixado para `extract`, `reconstruct`, `stitch`, `resize`,
  `pair`, `num_patches`, `tilings`, `paired_tilings`, `scale_factor`,
  `patch_metrics`, `per_patch_mse`, `per_patch_psnr`, `accel_available`:
  nome, ordem de parâmetros, marcação keyword-only, tipo de retorno.
- Campos e ordem dos quatro carriers (`TilingSpec`, `PairedTilingSpec`,
  `PatchPair`, `PatchMeta`) — inclui os fatos corrigidos: `tilings`/
  `paired_tilings` retornam `list`; `scale_factor` retorna `int | None`.
- `Cache`: assinaturas de `__init__`, `key_for`, `put`, `get` e propriedades
  `root`/`namespace`/`version`.
- `Patchify`: `__init__(patch_size, stride, dilation=1)` keyword-only conforme
  head.
- `WeightKind`: `typing.get_args` == `("uniform", "hann", "gaussian")` (D3).
- Verificação de mutação (FOCO §4): remover um símbolo do `__all__` ⇒ vermelho.

### 2.6 B1 — predicado qualificado nos pontos restantes

- `docs/SCOPE.md:229-230`: "`reconstruct` is bit-exact for unmodified patches"
  → qualificar com a regra do count map (potência de dois; ~1 ULP fora).
- `README.md:86` e `README.pt-BR.md:86` (cheat-sheet "inverso exato"):
  adicionar a condição ("quando toda contagem de cobertura é potência de dois —
  sempre vale para `stride == patch_size`").
- `src/patchcraft/stitch.py:3` e `:123`: "bit-exact inverse" → "inverse, exact
  under the count-map rule (see `reconstruct`)". A equivalência
  `stitch(weight="uniform")` ≡ `reconstruct` já está certa em THEORY:153,157;
  em `:157` remover a justificativa "validated by bit-exact equality test on
  no-overlap and `allclose` on overlap" (padrão de tolerância frouxa) e citar a
  suíte de duas metades.
- `docs/USAGE.md:88-91,134,149`: qualificação mínima dos claims de exatidão
  (o corpo segue 0.2.0 com o banner de staleness; regeneração é G2/B4).
- Verificação (FOCO §4): `grep -rn "bit-exact\|bit a bit\|exato" src/ docs/
  README.md README.pypi.md README.pt-BR.md` devolve só frases condicionadas.

### 2.7 B6 — código e texto

- `src/patchcraft/geometry.py:178-190`: no ramo `allow_overlap`, pular quando
  `nh == 1 and nw == 1` (D1). Atualizar o docstring de `TilingSpec.overlap` se
  necessário ("``overlap=True`` means stride < patch_size **and more than one
  patch**").
- `tests/test_geometry.py`: ajustar expectativas da enumeração 28×28
  (100 → 73 specs com overlap; os 27 degenerados saem) e adicionar teste
  explícito: nenhum spec com `total_patches == 1` tem `overlap=True`; todo spec
  emitido cobre exatamente (propriedade já existente, reafirmar na amostra).
- `docs/THEORY.md` §9.1: escrever a assimetria (D2) — trunca na ida é política
  de borda documentada; cobertura parcial na volta é `ValueError`; preflight
  com `num_patches`/`tilings`.
- `tests/test_extract.py`: `test_truncation_drops_trailing` permanece (D2 não
  muda comportamento) — reforçar o docstring apontando a política escrita.

### 2.8 Ramo sem zstandard — `tests/test_cache.py`

`monkeypatch` de `patchcraft.cache._try_zstandard` (ou do atributo `_zstd` da
instância) para `None`; exercitar `put`/`get` e conferir round-trip dos bytes e
`compressed: false` no sidecar JSON. ~10 linhas (FOCO §2, "entra por barateza").

### 2.9 Versão e nota de release

- `src/patchcraft/__init__.py`: `__version__ = "0.5.0"`; `pyproject.toml`
  idem.
- `CHANGELOG.md`: entrada 0.5.0 como **retratação pública**: o predicado
  `k_max <= 4` / "stride == patch/2" era falso; a regra correta é "todo valor
  do count map é potência de dois" (ADR 0003, com a refutação registrada).
  Mudança de comportamento: `tilings`/`paired_tilings` não emitem mais specs
  degenerados de patch único com `overlap=True` (27 specs a menos em 28×28).
  Sem mudança de assinatura; os 20 nomes ficam fixados por teste.

## 3. Verificação do grupo (FOCO §4)

1. **Mutação 1 ULP**: somar 1 ULP na acumulação do `reconstruct` ⇒ suíte
   vermelha; reverter. (Manual, no plano.)
2. **Mutação `__all__`**: remover um símbolo ⇒ `test_public_api.py` vermelho;
   reverter.
3. **Grep**: `grep -rn "bit-exact\|bit a bit\|exato" src/ docs/ README.md
   README.pypi.md README.pt-BR.md` ⇒ só frases condicionadas.
4. Suite completa verde nos dois modos (puro e com accel), `ruff` + `mypy
   --strict` limpos, `cargo test` 6/6 (intocado por esta fase).
5. Sweep completo local (`PATCHCRAFT_SWEEP_FULL=1`) uma vez antes do merge:
   zero contraexemplos ao predicado, zero divergências receita×count-map.

## 4. Fora de escopo (fica para G2/G3)

- Regeneração do `docs/USAGE.md` contra o head, doctests ligados
  (`NORMALIZE_WHITESPACE` + `ELLIPSIS`), ADR 0003 → Accepted, política de
  compatibilidade em prosa, §8 do THEORY como "adiado, aditivo se chegar",
  ROADMAP.md (parado em 0.2.x), caminhos `Z:\` em AUXILIARY.md — **G2** (sai
  como 1.0.0).
- SECURITY.md, CITATION.cff, AGENTS.md raiz, pre-commit, dependabot, templates
  — **G3** (sem versão).
- T2 (padding) — candidata nomeada da 1.1 (FOCO §3).
