# Fase 2 — `patchcraft-accel`: camada nativa Rust opcional (alvo: 0.4.0 + accel 0.1.0)

Data: 2026-09-01. Status: aprovado pelo usuário (design), aguardando revisão da spec.

Contexto: após a Fase 1 (0.3.0), o gargalo restante é o `F.fold` (col2im) dos
caminhos com overlap de `reconstruct`/`stitch` (~17–19 ms para 3×512×512,
p=32, s=16, CPU). Spike (`lab/spike_phase2.py`, 2026-09-01):

| Alternativa pura torch | Tempo | Veredito |
|---|---|---|
| baseline (`F.fold` + count map fechado) | 17.75 ms | — |
| `conv_transpose2d` com peso one-hot | 82.33 ms | 4.6× pior |
| `index_add_` com índices pré-computados | 18.69 ms | igual, +24 MB de índices |
| `torch.compile` | falha sem MSVC; não embarcável | descartado |
| `F.fold` isolado | 18.78 ms | é o gargalo |

`F.fold` mal paraleliza com batch=1 (torch usa 4 threads de 36 núcleos). A
vitória real é paralelismo por pixel de saída — território nativo.

Decisões já tomadas com o usuário:

1. **Pacote separado + extra**: `patchcraft-accel` no PyPI;
   `pip install patchcraft[accel]`; detecção em runtime com fallback silencioso
   para o caminho puro (padrão zstd do `Cache`; precedente uvicorn[standard] /
   aiohttp[speedups]). Falha/ausência do acelerador nunca quebra o base.
2. **4 alvos mainstream de wheel**: Windows x64, Linux x86_64 (manylinux),
   macOS arm64, macOS x86_64. abi3 → um wheel por alto cobre Python 3.12+.
3. Rust aceito como linguagem da camada nativa.

## 1. Arquitetura

Monorepo: subdiretório `accel/` com o crate Rust (`pyo3`, `rayon`) e
`pyproject.toml` próprio (maturin, abi3 = cp312). O pacote `patchcraft`
permanece 100% puro (hatchling) e ganha o extra `accel`.

### Kernel nativo — uma primitiva

Formulação *gather* do fold com overlap:

```
out[c, y, x] = Σ_p patches[p, c, y − p·row, x − p·col] · w[y − p·row, x − p·col]
```

sobre os patches `p` que cobrem `(y, x)`, com `w` opcional (`None` → uniforme).
Paralelizada sobre os pixels de saída com rayon: cada pixel é independente —
sem atomics, sem corridas, determinístico entre execuções e contagens de
threads (ordem de soma fixa: índice de patch crescente por pixel).

Interface nativa (sem acoplamento de ABI com torch — não linka libtorch, não
importa torch; recebe ponteiros crus):

```
fold_add(patches_ptr: int, out_ptr: int,
         L, C, ph, pw, H, W, sh, sw: int,
         kernel_ptr: int | None, dtype: str)  # "f32" | "f64"
```

Suporte: f32/f64, CPU, contíguo. f16/bf16 nunca chegam (patchcraft já promove
a f32). CUDA nunca chega (caminho torch). Wheels auto-contidos (link estático
Rust; zero dependências de sistema extras em qualquer OS).

Estimativa de ganho: fold 17–19 ms → ~2–3 ms em 36 núcleos (a validar nos
benchmarks da fase).

### Lado patchcraft

Novo módulo privado `src/patchcraft/_accel.py`:

- `try: import patchcraft_accel` na primeira chamada; checa
  `patchcraft_accel._ABI_VERSION == 1`; guarda o resultado (import once).
- Expõe `accel_available() -> bool` e `fold_weighted(patches, image_shape,
  stride, kernel | None) -> torch.Tensor | None` (None = indisponível/
  inaplicável → chamador cai no caminho torch).
- Override de debug: `PATCHCRAFT_ACCEL=0` no ambiente força o caminho puro.
- `accel_available()` reflete exatamente o que os caminhos de overlap
  usariam: accel importável E `_ABI_VERSION` ok E env não desabilitado.

`reconstruct` e `stitch` consultam `_accel.fold_weighted` nos caminhos com
overlap (o fast path sem overlap de `reconstruct` já é rearranjo puro — nunca
usa accel). Condições para usar: tensor CPU, dtype pós-promoção ∈ {f32, f64},
accel importável e ABI ok.

Nova API pública mínima em `patchcraft`: `accel_available() -> bool`
(suporte/debug). Entra em `__all__`.

### Fluxo de dados (reconstruct, overlap, accel presente)

1. `check_fold_geometry` (inalterado).
2. `work = patches.to(accum_dtype).contiguous()`.
3. `out = torch.empty(C, H, W, dtype=accum_dtype)`.
4. `_accel.fold_weighted(work, (C,H,W), (sh,sw), None)` → nativo com GIL
   liberado (`py.allow_threads`).
5. `out /= count` (forma fechada da Fase 1); `.to(patches.dtype)`.

`stitch`: idem com `kernel = wh[:,None] * ww[None,:]` (já construído).

## 2. Numérico

A soma por pixel segue índice de patch crescente — a mesma ordem do col2im do
`F.fold`. **Objetivo: bit-exato** vs o caminho puro, verificado com
`torch.equal` na grade de testes. Se divergir por ULP, os testes passam a
`assert_close` e a divergência é documentada (a decisão e a evidência ficam
registradas no CHANGELOG). Contagens uniformes (reconstruct) são inteiros
exatos em ambos os caminhos.

## 3. Erros

- patchcraft valida toda a geometria antes de chamar o nativo; o Rust assume
  entrada válida (asserts de debug apenas).
- Pânico em Rust vira exceção Python via PyO3 (sem `panic=abort`).
- ABI mismatch / import failure / env override → caminho puro silencioso.

## 4. Testes

- Rust: `cargo test` — kernel vs loop de referência naïve (grade de
  geometrias com overlap, retangular, sh∤ph, patch único).
- Python `tests/test_accel.py`: detecção, checagem de ABI, fallback, override
  por env, equivalência accel×puro na grade (overlap × retangular ×
  C∈{1,3,4} × f32/f64), determinismo (duas execuções idênticas), entrada
  não-contígua, kernel ausente (reconstruct) vs presente (stitch).
- A suíte existente continua verde nos DOIS modos (com e sem accel
  importável). Detalhe de modo: se a equivalência for bit-exata, os testes de
  caracterização da 0.3.0 passam intactos nos dois modos; se houver
  divergência de ULP, esses testes passam a fixar o modo puro (fixture com
  `PATCHCRAFT_ACCEL=0`) e a comparação accel×puro vive em `test_accel.py`
  com `assert_close` — a decisão sai na primeira execução da grade e fica
  registrada no CHANGELOG.
- Benchmarks `lab/`: puro × accel no caso 512² e num caso 2048².

## 5. CI/release

- `test.yml`: jobs existentes rodam puros (accel ausente → caminho puro);
  novo job (ubuntu + windows) compila o accel com maturin e roda a suíte com
  ele instalado (caminho acelerado). Ambos os caminhos testados a cada push.
- `release.yml`: novo job matriz (4 alvos) com maturin-action publicando
  `patchcraft-accel` via Trusted Publishing. **Ação do usuário na primeira
  vez**: configurar o publisher do pacote novo no PyPI. Depois publica o
  `patchcraft` como hoje.
- Sdist do accel: build requer Rust — documentado no README do `accel/`.
- Pré-release: validar o wheel manylinux localmente no WSL Ubuntu 26.04
  (`pip install` do .whl construído + rodar a suíte).

## 6. Versionamento

- `patchcraft` → 0.4.0 (feature nova, sem quebra).
- `patchcraft-accel` → 0.1.0; `_ABI_VERSION = 1`. O extra declara
  `patchcraft-accel>=0.1.0`; compatibilidade futura controlada pelo ABI check
  em runtime (bump de ABI = fallback puro, nunca crash).

## 7. Fora de escopo

Kernels GPU; dtypes inteiros; dilation (o fold nunca tem); batch (patchcraft é
single-image); Linux aarch64; publicação efetiva no PyPI (a configuração do
Trusted Publisher é do usuário; esta fase entrega CI pronta e validação local
Windows + WSL).

## 8. Critérios de aceite

1. `cargo test` verde; suíte pytest verde nos dois modos (com/sem accel).
2. Equivalência accel×puro na grade (`torch.equal`, ou `assert_close`
   documentado se ULPs divergirem).
3. Benchmark puro × accel documentado em `lab/` (meta: ≥4× no fold com
   overlap na máquina do usuário).
4. Wheel abi3 construído no Windows e validado no WSL (Ubuntu 26.04) com a
   suíte rodando acelerada.
5. `pip install patchcraft` puro continua funcional sem Rust, sem accel, sem
   mudança de comportamento.
6. CHANGELOG 0.4.0 + READMEs documentam o extra `[accel]`.
