# `lab/` — bancada de experimentos efêmeros

Espaço para exploração rápida: scripts, notebooks e checagens visuais que **não passam por revisão**, não têm testes formais e podem ser apagados a qualquer momento.

## Regras

1. **Tudo aqui é descartável.** Se um experimento der certo, ele migra: vira teste em `tests/`, ou (no futuro) vira `experiments/<NNN>-slug/`. Se ficar parado 2+ semanas sem virar nada, é candidato a apagar.
2. **Nada escreve no project root.** Outputs (PNG, JSON, CSV, checkpoints, dumps) vão sempre pra `Z:\outputs\patchforge\<slug>\` — o script cria a pasta. O project root continua limpo.
3. **Nada de `lab/` é importado por `src/patchforge/`.** O core depende só de torch/numpy/pillow; `lab/` pode importar tudo (torchvision, matplotlib, etc) — está do lado do framework auxiliar, não do core.
4. **Nomenclatura sugerida:** `YYYY-MM-DD-slug.{py,ipynb}` — prefixo de data dá ordem cronológica no `ls`.
5. **Git rastreia só este `README.md` e o `.gitignore`.** Qualquer outro arquivo aqui é local; não aparece em `git status`.

## O que esta bancada NÃO é

- Não é `experiments/` (futuro): aquele será para experimentos reprodutíveis, com seed fixo, dataset versionado e métrica esperada. Lab é o estágio anterior — onde a hipótese ainda está tomando forma.
- Não é `tests/`: testes definem o contrato; lab define a intuição.

## Como exemplo, um script típico

```python
# lab/2026-05-16-roundtrip-mnist.py
from pathlib import Path
from tests._datasets import mnist_subset
from patchforge import extract  # reconstruct virá com M3

out = Path(r"Z:\outputs\patchforge\2026-05-16-roundtrip-mnist")
out.mkdir(parents=True, exist_ok=True)

for img, label in mnist_subset(n_per_label=2, seed=0):
    patches = extract(img, patch_size=7, stride=7)
    # ... visualizar, salvar PNGs em `out`, comparar, etc.
```

Não preciso commitar esse script — ele é local. Se virar algo formal, promovo.
