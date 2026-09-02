<!-- l10n: doc_id=patchcraft-readme · lang=pt-BR · source_lang=en · translation_of=README.md -->
[English](README.md) · **Português**

# PatchCraft

[![CI status for main](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml/badge.svg)](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE)

**Codifica uma imagem em patches e decodifica de volta.** O PatchCraft cuida da aritmética do `unfold` e do `fold`, da validação da geometria e da mistura nas emendas, para que o seu pipeline possa cuidar de todo o resto.

**Uma imagem por vez, de propósito.** Cada chamada recebe um tensor float `(C, H, W)` e devolve um tensor, porque a quantidade de patches depende da imagem, e uma API em lote teria que preencher com padding ou devolver uma lista. Quem fornece o lote é o seu `for`, o `torch.vmap` ou o `DataLoader`.

```
   uma imagem                    a pilha de patches             a imagem de volta
   (1, 4, 4)                     (4, 1, 2, 2), ordem row-major  (1, 4, 4)

   +-----+-----+                 +-----+   +-----+              +-----+-----+
   | A A | B B |                 | A A |   | B B |              | A A | B B |
   | A A | B B |    extract      | A A |   | B B |  reconstruct | A A | B B |
   +-----+-----+   ---------->   +--p0-+   +--p1-+  ----------> +-----+-----+
   | C C | D D |   patch_size=2  +-----+   +-----+   stride=2   | C C | D D |
   | C C | D D |   stride=2      | C C |   | D D |              | C C | D D |
   +-----+-----+                 | C C |   | D D |              +-----+-----+
                                 +--p2-+   +--p3-+
```

A imagem sai como uma pilha de patches, você faz o seu trabalho sobre a pilha, e ela volta como uma
imagem só. Aquela última seta tem duas portas: o `reconstruct`, quando os patches estão intocados, e
o `stitch`, quando um modelo os reescreveu e as emendas precisam sumir.

Esta é a página de chamada. O manual está em [docs/GUIDE.md](docs/GUIDE.md), e é lá que ficam as medições, as tabelas e os exemplos longos.

## Instalação

```
pip install patchcraft
pip install "patchcraft[cache]"     # acrescenta o zstandard, que comprime o conteúdo do Cache
```

Ou com o acelerador nativo opcional (wheels prontos para Windows x64,
Linux x86_64, macOS arm64 e macOS x86_64; o pacote continua 100% Python
puro e funcional sem ele):

```bash
pip install patchcraft[accel]
```

`patchcraft.accel_available()` informa em runtime se o acelerador está
ativo; `PATCHCRAFT_ACCEL=0` no ambiente força o caminho puro.

O nome da distribuição e o nome de importação são os dois `patchcraft`. As dependências de execução são `torch>=2.6`, `numpy>=1.26` e `pillow>=10`. As versões de Python suportadas estão [no manual](docs/GUIDE.md#9-install-details-and-citation), junto com a observação que você precisa ler antes de instalar uma wheel de GPU.

## Sessenta segundos

```python
import torch
from patchcraft import extract, reconstruct, stitch

torch.manual_seed(0)
image = torch.rand(3, 256, 256)                     # um tensor float (C, H, W)

patches = extract(image, patch_size=32, stride=32)  # (L, C, ph, pw) == (64, 3, 32, 32)

back = reconstruct(patches, image.shape, stride=32) # o inverso, para patches intocados
assert torch.equal(back, image)                     # o mesmo tensor, bit a bit

edited = patches * 1.05 - 0.01                      # faz as vezes de um modelo por patch
blended = stitch(edited, image.shape, stride=32, weight="hann")
assert blended.shape == image.shape                 # emendas suavizadas, geometria preservada
```

O laço é esse. Você extrai, faz o seu trabalho em cada patch, e depois volta com o `reconstruct` ou com o `stitch`. O dtype e o device da entrada sobrevivem nos dois sentidos.

## `reconstruct` ou `stitch`

As duas funções respondem a perguntas diferentes, então escolher entre elas é a primeira decisão que você toma.

O `reconstruct` é o inverso. Ele parte do princípio de que os patches ainda guardam os pixels que o `extract` entregou, divide pelo mapa de cobertura e, numa geometria que cobre a imagem, devolve a imagem inalterada.

O `stitch` é para patches que um modelo reescreveu. Nesse caso os patches vizinhos passam a discordar sobre os pixels que compartilham, e a média uniforme deixa essa discordância visível como uma grade de emendas, então o `stitch` pondera cada patch por uma janela que desce até a borda.

| Você quer | Chame | Porque |
|---|---|---|
| Os patches de volta como imagem, intocados | `reconstruct` | Inverso exato do `extract` quando toda contagem de sobreposição é potência de dois, o que sempre vale com `stride == patch_size` |
| A saída de um modelo de volta como imagem | `stitch` | Patches sobrepostos discordam, e a janela esconde a grade |

## Por que não usar `unfold` e `fold` direto

Porque existem dois defeitos esperando ali, os dois silenciosos, e você encontra ambos na primeira hora.

O primeiro é o reshape. O `F.unfold` devolve `(1, C*ph*pw, L)`, e o reshape intuitivo para `(L, C, ph, pw)` te dá a forma certa com os pixels errados.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract

image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
patches = extract(image, patch_size=4, stride=4)             # (4, 1, 4, 4)
cols = F.unfold(image.unsqueeze(0), kernel_size=4, stride=4) # (1, C*ph*pw, L)

scrambled = cols[0].view(-1, 1, 4, 4)                        # o reshape intuitivo
assert scrambled.shape == patches.shape                      # a forma certa
assert not torch.equal(scrambled, patches)                   # e os pixels errados
```

O segundo é um stride que não cobre a imagem. Numa imagem de 128 por 128 com `patch=32, stride=20`, a grade para no pixel 112, o que deixa 3840 dos 16384 pixels em zero, e um `fold` escrito à mão devolve essa imagem parcialmente preta sem reclamar.

Escrever o laço de recortar e remontar à mão custa 17 linhas não vazias contra 3 aqui, e os dois resultados são idênticos bit a bit. [O manual](docs/GUIDE.md#1-why-not-unfold-and-fold-directly) roda as duas versões lado a lado.

## A geometria precisa cobrir a imagem

O `reconstruct` e o `stitch` recusam uma grade que não cobre a imagem inteira, e a mensagem de erro diz até onde ela cobriu. A saída é escolher uma geometria válida, e não preencher com padding, porque o padding sintetiza pixels que você nunca teve.

```python
import torch
from patchcraft import extract, reconstruct, tilings

image = torch.rand(1, 128, 128)
patches = extract(image, patch_size=32, stride=20)   # uma grade que para no pixel 112
try:
    reconstruct(patches, image.shape, stride=20)
except ValueError as error:
    print(error)                                     # ... covers (112, 112) of (128, 128)

print([s.patch_size for s in tilings(image.shape)])  # 7 tilings exatos, só a partir da forma
```

O `tilings` é aritmética sobre a forma, então ele não lê nada e não aloca nada. Passe `allow_overlap=True` quando quiser também as geometrias com sobreposição.

## Onde você está se metendo

A superfície é um tensor de entrada e um tensor de saída. Não existe eixo de lote, nem dataset, nem dataloader, nem treino, e essa fronteira é vinculante em vez de provisória, registrada na [docs/THEORY.md](docs/THEORY.md) §0.

O PatchCraft ajuda quando você recorta uma imagem, faz alguma coisa em cada patch e remonta. Isso cobre inferência numa imagem grande demais para um único forward, datasets de patches em baixa e alta resolução, análise por janela deslizante e mapas de erro por patch.

O PatchCraft não ajuda quando você quer uma operação em lote sobre N imagens, um dataset, um dataloader, um padding que force uma geometria incômoda a caber, ou um modelo. Os três primeiros são trabalho do seu pipeline, e o quarto é uma rede neural. A [docs/SCOPE.md](docs/SCOPE.md) traça a linha inteira.

## Quando o round-trip é bit a bit

O round-trip é exato quando todo valor do mapa de cobertura é potência de dois, e isso acontece porque dividir um float por uma potência de dois é a única divisão que nunca arredonda. Fora dessa regra, o erro por pixel é limitado por (k+1)·eps·|v|, com k a contagem de cobertura do pixel. Um float mais largo compra um erro menor e nunca a exatidão, então o float64 não é um porto seguro. Quem decide a resposta é a geometria, e não o dtype. [O manual](docs/GUIDE.md#4-when-the-round-trip-is-bit-for-bit) traz a varredura que mede isso.

## Estado

**0.5.0, pré-1.0.** Os valores de saída ainda podem mudar numa versão menor, e o [CHANGELOG.md](CHANGELOG.md) registra cada mudança com a medição por trás dela.

São 1571 testes coletados, com CI verde em {Ubuntu, Windows} x {Python 3.12, 3.13, 3.14}, e com `ruff check` e `mypy --strict` na mesma execução. O pacote é tipado e distribui o `py.typed`.

Nenhum projeto externo consumiu a biblioteca ainda, e nenhum caminho CUDA dela jamais executou. [O manual](docs/GUIDE.md#8-what-this-project-does-not-claim) lista o que mais este projeto se recusa a afirmar.

## Onde ler em seguida

| Se você quer | Abra |
|---|---|
| O manual: cada argumento acima, medido, com a saída | [docs/GUIDE.md](docs/GUIDE.md) |
| Um passeio por cada um dos 20 símbolos públicos | [docs/USAGE.md](docs/USAGE.md) |
| A linha entre esta biblioteca e o seu pipeline | [docs/SCOPE.md](docs/SCOPE.md) |
| A matemática, as decisões e o contrato por função | [docs/THEORY.md](docs/THEORY.md) |
| Por que a API tem essa cara | [docs/ADR/](docs/ADR) |
| O que mudou em cada versão | [CHANGELOG.md](CHANGELOG.md) |
| Como clonar, testar e contribuir | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Licença e citação

MIT, em [LICENSE](LICENSE). Ainda não existe DOI, então, se você precisar citar este trabalho, a entrada BibTeX está [no manual](docs/GUIDE.md#9-install-details-and-citation).
