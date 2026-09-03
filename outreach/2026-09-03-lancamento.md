<!-- l10n: doc_id=patchcraft-outreach-2026-09-03 · lang=pt-BR · canonical -->
**Português** · [English](2026-09-03-release.en.md)

# PatchCraft 0.5.1, fonte de notícia

Documento datado que serve de fonte para os textos de canal em [`linkedin/`](linkedin/).
Regra: nenhum texto de canal muda sem esta fonte mudar antes. Nada aqui é estimativa;
cada número tem um comando que o reproduz.

## O que a biblioteca é

Recorta uma imagem em patches e remonta. Uma imagem por chamada, um tensor entra e um
tensor sai. Não há eixo de lote, nem dataset, nem dataloader, nem modelo, e essa fronteira
está escrita como vinculante em `docs/THEORY.md` §0.

`pip install patchcraft` · MIT · pré-1.0 · Python 3.12 a 3.14 · torch 2.6 ou mais novo.

## As manchetes

### 1. A retratação pública

A biblioteca afirmava, nas próprias docstrings e em quinze pontos da documentação, quando
o round-trip é exato bit a bit. A afirmação estava errada, e não por pouco: dizia que o
erro fora da regra ficava em torno de 1 ULP, quando ele cresce com a contagem de cobertura
do pixel e chega a 19 ULP em float32.

O contrato correto, medido: o round-trip é exato se e somente se todo valor do mapa de
cobertura for potência de dois. Fora disso o erro por pixel é limitado por `(k+1)·eps·|v|`,
com `k` a contagem de cobertura daquele pixel.

O motivo de a suíte não ter pegado isso antes também está registrado: os testes construíam
as imagens com `torch.arange`, e dado inteiro faz o round-trip fechar exato onde dado
aleatório não fecha.

### 2. A suíte cujo trabalho é derrubar o contrato

Depois da correção veio um teste com a função explícita de falsificar a nova afirmação.
Ele enumera as 126.736 geometrias legais do espaço, independentemente do predicado, sorteia
uma amostra semeada e tenta quebrar as duas metades: procura um caso dentro do predicado
que não seja exato, e um caso fora que seja exato por sorte. A varredura completa fica
atrás de `PATCHCRAFT_SWEEP_FULL=1`.

### 3. Os dois defeitos silenciosos que motivam a biblioteca

O primeiro é o reshape. O `F.unfold` devolve `(1, C*ph*pw, L)`, e o reshape intuitivo para
`(L, C, ph, pw)` entrega a forma certa com os pixels errados. Nada reclama.

O segundo é o stride que não cobre a imagem. Numa imagem 128 por 128 com patch 32 e stride
20, a grade para no pixel 112 e deixa 3840 dos 16384 pixels em zero. Um `fold` escrito à
mão devolve essa imagem parcialmente preta sem erro.

### 4. O acelerador nativo, e por que o ganho não é contagem de núcleos

Cinco das seis wheels trazem um kernel Rust para o fold com sobreposição, compilado dentro
da própria wheel. Não existe extra para habilitar.

| Geometria | Chamada | Torch puro | Acelerado | Ganho |
|---|---|---|---|---|
| 3x512x512, patch 32, stride 16 | `reconstruct` | 16,3 ms | 2,3 ms | 7,1x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 453,7 ms | 32,1 ms | 14,1x |
| 3x2048x2048, patch 64, stride 32 | `stitch` hann | 460,9 ms | 37,9 ms | 12,2x |

A objeção óbvia é que o acelerador só usa mais núcleos. Ela foi medida: forçando o torch a
4, 8, 16 e 36 threads no caso maior, o caminho puro fica entre 365 ms e 465 ms, e com 36
threads não é melhor do que com 8. O `F.fold` não escala com lote 1, que é a razão de o
kernel existir.

O benchmark compara os dois caminhos com `torch.equal` antes de reportar qualquer tempo, e
sai com erro se eles diferirem.

## Estado, dito por inteiro

São 1534 testes passando, com CI em {Ubuntu, Windows} x {Python 3.12, 3.13, 3.14} no
caminho puro mais um job acelerado nos dois sistemas. A superfície pública são 20 nomes,
congelados por teste com `inspect.signature`.

Nenhum projeto externo consumiu a biblioteca ainda. Nenhum caminho CUDA dela jamais
executou, nem na CI nem fora dela. Três das cinco wheels aceleradas nunca tiveram o kernel
executado em CI: as de macOS e aarch64 são construídas e conferidas, e só.

## Reprodução

```
pip install patchcraft
python tools/benchmark.py --markdown
PATCHCRAFT_SWEEP_FULL=1 pytest tests/test_exactness.py
```

## Ligações

- Repositório: https://github.com/LeoPR/PatchCraft
- Desempenho, com máquina, versões e data: `docs/PERFORMANCE.md`
- O contrato de exatidão: `docs/ADR/0003-reversibility-classes.md`, ainda em `Proposed`
- O que o projeto se recusa a afirmar: `docs/GUIDE.md` seção 8
