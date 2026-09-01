# Fase 1 — Correção de bugs + performance em puro torch (alvo: 0.3.0)

Data: 2026-09-01. Status: aprovado pelo usuário (design), aguardando revisão da spec.

Contexto: análise crítica do PatchCraft 0.2.2 (código ~1.614 LOC, 346 testes
passando) identificou bugs pequenos e vitórias grandes de performance sem
mudança de API. A camada nativa (Rust) fica para a Fase 2, com spec própria.

Abordagem escolhida (entre A/B/C): **A — fast paths internos + formas
fechadas**, sem cache, sem mudança de assinatura, bit-exato. Descoberta-chave:
o count map do `reconstruct` e o denominador do `stitch` são separáveis e têm
forma fechada O(H+W), eliminando o segundo `F.fold` sem nenhum cache.

Medições de base (CPU, imagem 3×512×512, `lab/bench_quick.py`):

| Operação | Atual | Fast path | Speedup |
|---|---|---|---|
| `extract` sem overlap (p=32, s=32) | 2.19 ms | 0.16 ms | 13.7× |
| `extract` com overlap (p=32, s=16) | 9.53 ms | 0.45 ms | 21× |
| `reconstruct` sem overlap | 4.32 ms | 0.16 ms | 27× |
| `reconstruct` com overlap | 26.1 ms | −4.3 ms (fold de ones eliminado) | ~17% |
| `patch_metrics` (256 patches) | 1.71 ms | 1.13 ms | 1.5× |

## 1. Correções de bugs (código)

1. `src/patchcraft/resize.py` `_tensor_to_pil_u8`, ramo não-float: clamp ao
   range do dtype (`torch.iinfo`) antes do cast para uint8. Hoje um tensor
   int32 com valor 300 vira 44 (wrap). Simétrico ao fix 0.2.0 do backend
   torch. Teste novo.
2. `src/patchcraft/stitch.py:103` docstring: o piso da janela gaussiana 2-D é
   `exp(-4)` no canto (produto de duas 1-D com piso `exp(-2)`), não
   "strictly above exp(-2) everywhere". Medido: min=0.0195 em n=128.
3. `pyproject.toml:74`: remover `cache_dir = "Z:\\caches\\pytest"` (caminho
   local de máquina Windows; em Linux cria diretório literal `Z:\caches\...`).
4. `src/patchcraft/metrics.py:154`: remover ramo morto — `per_patch_mse`
   garante float64, então `torch.finfo(mse.dtype)` é incondicional.
5. `src/patchcraft/resize.py:125,183`: `torch.empty(0, dtype=d).is_floating_point()`
   → `d.is_floating_point` (sem alocação).
6. `src/patchcraft/cache.py`: manter prefixo de 16 hex chars (colisão exige
   ~10⁹ entradas); documentar o tradeoff no docstring da classe.
7. `reconstruct.py`/`stitch.py`: extrair a validação geométrica duplicada
   (~60 linhas verbatim em cada) para helper interno comum, com mensagens de
   erro idênticas às atuais.

## 2. Performance

Zero mudança de API; device-agnostic (CUDA incluso); fp16/bf16 mantêm
promoção interna a f32. Precisão da equivalência, por função:

- `extract`, `reconstruct` (ambos os caminhos): **bit-exato**. O count map da
  forma fechada produz inteiros idênticos aos do fold de `ones`, logo a
  divisão é idêntica.
- `stitch` com `weight="uniform"`: bit-exato (mesma razão).
- `stitch` com `hann`/`gaussian`: o denominador é soma de valores de kernel e
  a ordem de soma pode diferir da do `F.fold` — equivalência dentro de ULPs
  de float (tolerância apertada, ex.: `torch.testing.assert_close` padrão
  para o dtype), documentada no teste.
- `metrics`: bit-exato se a redução com `dtype=` seguir a mesma ordem; se
  divergir em ULPs em algum backend, cobrir com tolerância de f64.

### 2.1 `extract`

Fast path para `dilation==1`: `Tensor.unfold` (view) → permute → reshape
(cópia). `dilation>1` mantém `F.unfold`. Contrato exige memória independente
da imagem: o reshape copia na esmagadora maioria das geometrias, mas há caso
degenerado em que a janela permutada é contígua (ex.: `ph==1` com cobertura
exata) e o reshape devolve view — nesse caso forçar cópia explícita. Teste de
independência de memória obrigatório.

### 2.2 `reconstruct`

- Fast path `stride==patch_size` (count==1 em todo pixel): reshape+permute
  direto, sem fold nem count map.
- Caso geral: count map em forma fechada, sem fold de `ones`:
  `count_h[y] = min(y//sh + 1, num_h, (h-1-y)//sh + 1)` (idem W), e
  `count = count_h[:,None] * count_w[None,:]`. O `F.fold` dos patches
  permanece (é o custo real do col2im).

### 2.3 `stitch`

Denominador separável: `den[y,x] = (Σᵢ w_h[y − i·sh]) · (Σⱼ w_w[x − j·sw])`,
computado como duas dobras 1-D (O(H), O(W)) + produto externo, eliminando o
segundo `F.fold` 2-D. Numerador inalterado (fold dos patches ponderados).
Com `weight="uniform"` o resultado deve bater com `reconstruct` como hoje.

### 2.4 `metrics`

- `patch_metrics`/`per_patch_mse`: acumular com redução `dtype=torch.float64`
  (sem materializar cópias f64 dos operandos).
- `patch_metrics`: empilhar os 3 escalares (mae, mse, max_abs) num único
  tensor e um só `.tolist()` — 1 sincronização GPU em vez de 3.
- Valores de retorno devem permanecer bit-exatos; se a redução com `dtype=`
  divergir em ULPs da ordem atual em algum backend, documentar e cobrir com
  tolerância de f64 nos testes.

## 3. Testes

- Suíte atual (346 testes) passa sem modificação — prova de não-regressão.
- Novos testes de equivalência fast path vs referência
  (`F.unfold`/`F.fold` chamados diretamente no teste) sobre grade:
  overlap × não-overlap, retangular, `dilation>1` (caminho lento),
  C ∈ {1,3,4}, H≠W, f32/f64/f16; CUDA marcada `gpu` (skip sem placa).
  Igualdade exata onde a spec §2 declara bit-exato; `assert_close` onde
  declara tolerância (stitch hann/gaussian).
- Teste novo: wrap uint8 no resize (bug 1).
- Teste novo: independência de memória do `extract` (mutar patch não altera
  imagem).
- `lab/bench_quick.py` estendido como benchmark antes/depois.

## 4. Docs — apenas correções factuais

- 0.2.1 → 0.3.0 em `README.md:136`, `README.pt-BR.md:136`,
  `README.pypi.md:119`, `docs/GUIDE.md` (linhas 9, 726, 791, 845).
- `docs/SCOPE.md:241-248` §4.4: `reconstruct` tem guarda de dtype desde
  0.2.1 — corrigir a seção.
- Regra de bit-exatidão (power-of-two do count map): corrigir
  `docs/THEORY.md:100,153,157` e `docs/USAGE.md:149-160`, que ainda trazem a
  regra antiga.
- `docs/THEORY.md` §9.2: remover "Promoção automática float16 → float32" da
  lista de fora-de-escopo (implementado desde 0.2.1, linha 269 já diz).
- `CHANGELOG.md`: nova entrada 0.3.0; link targets para `[Unreleased]`,
  `[0.2.2]`, `[0.2.1]`; corrigir "137 lines" (linha 65) e o sigma da
  gaussiana (linhas 235-236, per-axis `max(1.0, n/4.0)`).
- `docs/AUXILIARY.md:114`: links markdown para `Z:\...` viram texto puro.

## 5. Versionamento

Bump para **0.3.0** em `src/patchcraft/__init__.py` (melhoria de performance
significativa, sem quebra de API). CHANGELOG atualizado.

## 6. Fora de escopo (Fase 2 em diante)

- Camada nativa Rust (PyO3/maturin, wheels abi3, módulo opcional com
  fallback puro — "instalação inteligente" no padrão do zstd do `Cache`).
- Consolidação/corte de docs (USAGE.md, SCOPE, AUXILIARY, ROADMAP, archive/).
- Protótipo `index_add_` com mapas de índices (não-determinístico em CUDA;
  pertence à investigação da Fase 2).

## 7. Critérios de aceite

1. Suíte completa passa (346 + novos), com os níveis de equivalência da
   spec §2 (exata / `assert_close`) verificados.
2. Benchmark antes/depois em `lab/` mostra os ganhos da tabela (ordem de
   grandeza) em CPU.
3. Nenhuma assinatura pública alterada; `__all__` inalterado.
4. Nenhuma menção de versão 0.2.1/0.2.2 stale nos pontos listados em §4.
