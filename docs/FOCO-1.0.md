# Foco do 1.0 do PatchCraft

## 0. A fronteira verdadeira, e por que as duas versões anteriores erraram

O predicado do ADR 0003 na árvore de trabalho, `k_max <= 4`, é falso. O predicado correto é: **o round-trip `extract` + `reconstruct` é bit a bit exato quando todo valor do count map é potência de dois.**

A justificativa é aritmética, não estatística, e é isto que precisa entrar no ADR no lugar da contagem de geometrias. Para patches não modificados, o valor reconstruído de um pixel é a soma de `k` parcelas idênticas dividida por `k`, onde `k` é a contagem **daquele pixel**. Em IEEE 754, somar `k` cópias de um mesmo valor e dividir por `k` é exato quando `k` é potência de dois, porque cada passo é um deslocamento de expoente sem perda de mantissa. O ADR aplicou esse argumento ao **máximo** do mapa, e um mapa com máximo 4 admite pixels com contagem 3. Dividir por 3 não é exato. Daí o erro.

A varredura entra como falsificador do predicado, nunca como sua origem. Enumerei o espaço legal com H, W em 4..24, patch 2..9 e stride de cobertura exata: 126.736 geometrias. Sobre uma amostra de 1.500 delas nos dois dtypes:

| predicado | previu exato | contraexemplos |
|---|---:|---:|
| `k_max <= 4` | 1.142 | **286** (25%) |
| todo valor do count map é potência de dois | 850 | **0** |

Caso mínimo: imagem `(1, 4, 14)`, patch `(4, 8)`, stride `(4, 2)`. Count map `{1, 2, 3, 4}`, logo `k_max == 4`, e `torch.equal` devolve False em float32 (5,96e-08) e em float64 (1,11e-16).

**Por que `k_max <= 4` sobreviveu à própria evidência:** sobre a amostra, o maior `k` em qualquer mapa todo-potência-de-dois é exatamente 4. O predicado antigo é condição **necessária e não suficiente**, então uma varredura que só visite geometrias já dentro do predicado correto nunca o derruba. Esse fato entra no ADR, porque é a explicação do erro.

**Forma barata de avaliar, sem símbolo novo.** O predicado 2-D equivale ao teste por eixo: em cada eixo independentemente, toda contagem de cobertura distinta é potência de dois. Zero divergências sobre a amostra de 1.500. As contagens por eixo são O(H+W) e não exigem `fold`, então o predicado é uma receita documentada de três linhas sobre `patch_size`, `stride` e a forma da imagem. Isso fecha a pergunta "campo em `TilingSpec`, função nova ou receita", que era a única decisão de S1 capaz de reabrir o congelamento dos 19 nomes: **receita, sem nome novo.**

**Não adotar o atalho `stride == patch_size` ou `patch_size / 2`.** Ele nunca promete demais, mas deixa de fora 27.913 das 34.969 geometrias que satisfazem o predicado no espaço enumerado, 79,8%. Prometer menos do que a medição sustenta é o espelho do defeito que estamos corrigindo, e leria como uma segunda retratação depois.

**Quarto e quinto geradores de falso negativo.** Além da rampa inteira já documentada: um float64 construído como `torch.rand(...).double()` (mantissa parcialmente preenchida) round-trippa True na mesma geometria em que `torch.rand(..., dtype=torch.float64)` devolve False. E `torch.equal` é False para uma imagem toda NaN mesmo quando os bits voltam idênticos (`view(torch.int32)` compara igual). O contrato numérico precisa dizer em que comparação está escrito.

Scripts: `...\scratchpad\final_check.py`, `final_check2.py`, rodados com `.venv/Scripts/python.exe`.

---

## 1. O que 1.0 congela

Quatro coisas, e apenas essas quatro.

1. **Os 19 nomes e suas assinaturas.** Nome, ordem dos parâmetros, keyword-only, tipo de retorno. Inclui os campos e a ordem dos quatro carregadores de dados (`TilingSpec`, `PairedTilingSpec`, `PatchPair`, `PatchMeta`). Duas correções de fato que o texto precisa refletir: `tilings` e `paired_tilings` retornam `list`, não tupla, e `scale_factor` retorna `int | None`.
2. **`WeightKind` é um `Literal`, não um carregador de dados** (`stitch.py:26`). Congelá-lo fecha o conjunto de janelas: acrescentar uma quarta depois de 1.0 muda um tipo público e quebra qualquer consumidor com verificação de exaustividade sob `mypy --strict`. Isso não é crescimento aditivo por keyword-only. A política de compatibilidade tem que declarar o conjunto **aberto** (novas janelas são adição compatível, o consumidor não pode assumir exaustividade) ou **fechado**. Recomendo aberto, escrito.
3. **O conjunto aceita/rejeita do THEORY §9.** Toda entrada que hoje funciona continua funcionando, todo `ValueError` de hoje continua `ValueError`. O texto da mensagem fica livre.
4. **O contrato numérico como predicado que o caller avalia antes de chamar**, escrito na comparação em que vale. A frase congelada é: dentro do predicado, `reconstruct` devolve os mesmos bits, verificável por `torch.equal` para entradas sem NaN e por comparação da view inteira em qualquer caso; fora do predicado, o erro é limitado a 1 ULP.
5. **A saída de enumeração de `tilings` e `paired_tilings`.** É dado consumido em loop, então mudar o conjunto retornado depois de 1.0 é quebra.

Não congela: implementação interna, texto de mensagem de erro, performance, layout de `lab/`, `docs/STUDIES`.

Teste de aceitação da definição inteira, em uma frase: **um estranho instala pelo PyPI, lê a página de documentação, e sabe quando o round-trip é exato sem abrir o código.**

---

## 2. Bloqueadores

**B1. O predicado errado, em quinze lugares, corrigido em um commit.** Medido por grep sobre `src/`, `docs/`, `README.md` e `README.pypi.md`:

`src/patchcraft/reconstruct.py:24`; `src/patchcraft/stitch.py:3`, `:11`, `:89`; `docs/THEORY.md:100`, `:153`, `:157`; `docs/USAGE.md:74`, `:128`, `:163`; `docs/SCOPE.md:229`; `docs/ADR/0003-reversibility-classes.md:16`, `:66`, `:72`, `:78`; `README.md:52`, `:120`; `README.pypi.md:11`, `:41`, `:47`, `:61`, `:66`.

Dois desses fazem a afirmação em outras palavras e não estavam na lista anterior: `stitch.py:11` e `THEORY.md:157` dizem que `weight="uniform"` é matematicamente equivalente a `reconstruct`, e o segundo justifica isso com "teste de igualdade bit a bit em não-sobreposição e `allclose` em sobreposição", que é exatamente o padrão de tolerância frouxa que B2 existe para matar. O CHANGELOG é registro histórico e não se reescreve; a retratação entra como entrada da 0.3.0.

O grep de verificação precisa incluir a raiz do repositório, porque `README.md` e `README.pypi.md` não estão em `src/` nem em `docs/`. Nota de 2026-08-30: `pyproject.toml:8` passou a apontar para `README.pypi.md`, então a descrição publicada no PyPI é essa página e não o `README.md`.

**B2. A suíte não consegue falsificar B1.** Cinco geradores de falso negativo: rampa inteira (`test_reconstruct.py:10`), float64 alargado de float32, NaN sob `torch.equal`, varreduras que só visitam `stride == patch/2`, e `rtol=1e-5` três ordens acima do erro medido.

A metade negativa precisa ser escrita como afirmação sobre um conjunto de sementes, não sobre uma execução. Medido na geometria `(1, 4, 14)`, patch `(4, 4)`, stride `(1, 1)`, fora do predicado: `torch.equal` devolve True em 63 de 300 sementes float32 e 57 de 300 float64. Exatidão fora do predicado é propriedade da amostra, não da geometria. A forma correta é: **pelo menos uma de N sementes é inexata, e toda semente tem erro dentro de 1 ULP.** Escrita como assert simples, essa metade instala flake no CI e chama isso de detector.

**B3. Nada fixa a superfície pública.** `tests/test_import.py` verifica duas coisas. Fixar os 19 nomes com `inspect.signature`, mais a decisão do item 2 da seção 1.

**B4. As duas páginas que o estranho lê estão erradas, e uma delas não está no git.** `USAGE.md` afirma 0.2.0, omite `WeightKind`, e seus exemplos não rodam em lugar nenhum: 87 tentados, 69 passam, 18 falham. Uma das falhas é alinhamento de coluna, então os flags `NORMALIZE_WHITESPACE` e `ELLIPSIS` são parte do contrato, não detalhe de fiação. Nota de 2026-08-30: `README.pypi.md` foi reescrito (222 linhas), commitado e fiado no `pyproject.toml`, e já enuncia a regra na forma correta (todo valor do mapa de cobertura é potência de dois). O que resta de B4 é o `USAGE.md`. `README.md` já tem todos os links absolutos, então esse item está fechado para o arquivo que o PyPI publica hoje.

**B5. THEORY §9 se contradiz sobre a promoção fp16.** ~~§9 se declara árbitro do contrato e o árbitro aceita e exclui o mesmo comportamento.~~ **Fechado em 0.5.2.**

A contradição que este bloco nomeava já tinha caído em `33d3002`, durante a 0.3.0: a linha "Fora de escopo v0.1: Promoção automática float16 → float32" foi removida da §9.2, e nada aqui registrou. Ao conferir para fechar, apareceu um segundo defeito na mesma frase, esse ainda vivo e mais grave, porque a §9 é o árbitro do contrato e estava afirmando um fato falso: a promoção de `float16` **e** `bfloat16` era justificada pelo estouro do máximo finito de fp16, e o `bfloat16` carrega o expoente do `float32` e não estoura. Medido: numerador em `9.0112e+04` contra máximo finito de `3.390e+38`, zero `inf`. A promoção dele compra precisão, não alcance, e a §9.2 agora tem uma entrada para cada formato com a medição de cada um.

**B6. Enumeração com lixo e guardas assimétricas.** `tilings((1,28,28), allow_overlap=True)` devolve 100 specs, 28 degenerados de patch único, 27 deles marcados `overlap=True`, onde não há com o que sobrepor (guarda `nh > 1 or nw > 1` em `geometry.py`). E a guarda de cobertura existe só na volta: `extract(torch.rand(3,130,130), patch_size=32, stride=32)` devolve `(16, 3, 32, 32)` sem erro, descartando 2 linhas e 2 colunas em silêncio, e `Patchify` faz o mesmo. Quem extrai, roda o modelo e nunca reconstrói recebe exatamente a perda silenciosa que a biblioteca promete impedir. Ambos são itens 4 e 3 do congelamento: corrigir depois de 1.0 é quebra. Ou corrige agora, ou documenta a assimetria por escrito.

**O que não bloqueia, contra o inventário:**

- **Tabela de regimes em seis docstrings.** Linha de regime onde há regime: `reconstruct`, `stitch`, `resize`. Os outros três têm um regime só.
- **CUDA.** Não há build CUDA nesta máquina, então o item é espera sem fim. Escopar por escrito, "medido em CPU", uma frase no §9 e uma na docstring. Escopo declarado não é dívida.
- **O gate do consumidor.** PatchSR não começou. Esperar por um consumidor inexistente é esperar indefinidamente. 1.0 não significa "a forma foi provada ideal", significa "as promessas são verdadeiras e testadas". O substituto fechável já está medido: o comparativo `hand.py` contra `pc.py`, bit-idêntico, entra como teste.
- **Ramo sem zstandard** (`cache.py:143-145`): dez linhas de `monkeypatch`, entra por barateza.

---

## 3. Fora do 1.0

**T2 a T14, inteiros.** Cada tarefa do estudo adiciona superfície, e 1.0 é a versão que para de adicionar superfície. Duas ressalvas: **T2 (padding)** é a única com pressão real, porque hoje a resposta a "minha imagem não tem tiling exato" é um `ValueError`; fica fora mesmo assim, registrada como candidata nomeada da 1.1, porque `pad` é puramente aditivo. **T3 e T7** foram confirmados no lab, e `lab/.gitignore` é `*`, então a única evidência existe nesta máquina. Ou os scripts que sustentam afirmação em documento entram no git, ou a afirmação sai do documento.

Fora também, e reais: CITATION.cff, SECURITY.md, code of conduct, templates, pre-commit, dependabot, badges, link refs do CHANGELOG, e os ADRs 0001/0002 citando `archive/` que não existe mais. AGENTS.md merece nota: não bloqueia, e é o único item da lista que protege o trabalho das próximas sessões.

O §8 do THEORY (channels-last, PIL, batched) sai de "pergunta aberta" e vira "adiado, aditivo se chegar". Seção de perguntas abertas viva em cima de API congelada é contradição.

---

## 4. Três grupos que fecham juntos

**Commit zero: feito em 2026-08-30.** ADR 0003 revisado, `README.md` reescrito, `README.pypi.md` criado e fiado, e este documento entraram no git no mesmo commit. O predicado do ADR foi corrigido de `k_max <= 4` para a regra de potência de dois antes de commitar, com a refutação registrada no próprio ADR.

**G1, o predicado e a suíte que pode derrubá-lo. Sai como 0.3.0.** Os dois não fecham separados: a verificação de B1 exige o enumerador sobre o espaço legal e o helper único de geração de dados, que são material de B2. Escopo: predicado corrigido nos quinze lugares, com a prova aritmética no ADR e a varredura como falsificador; helper auditado (aleatório, mantissa cheia, no dtype alvo, proibido derivar float64 de float32) aplicado aos treze casos de round-trip; `rtol` apertado; contrato escrito sobre `torch.equal` mais view inteira; os 19 nomes fixados com `inspect.signature`; decisão sobre `WeightKind`; ramo sem zstandard; `hand.py` contra `pc.py` como teste; decisão sobre B6.
*Verificação:* teste de mutação, e é o ponto do grupo. Some 1 ULP na acumulação do `reconstruct`, confirme vermelho, reverta. Remova um símbolo do `__all__`, confirme vermelho, reverta. `grep -rn "bit-exact\|bit a bit\|exato" src/ docs/ README.md README.pypi.md` devolve só frases condicionadas. Sai como 0.3.0 porque a correção é retratação pública e merece nota de release própria, e porque B6, se corrigido, muda comportamento.

**G2, as duas páginas e o contrato ratificado. Sai como 1.0.0.** Os dois editam o THEORY §9 e ambos escrevem o escopo "medido em CPU", então colidem no mesmo arquivo. Escopo: `USAGE.md` regenerado contra 0.2.x com `WeightKind` e as quatro mudanças de comportamento da 0.2.1; doctests ligados no `addopts` e nos dois workflows, com `NORMALIZE_WHITESPACE` e `ELLIPSIS` declarados no contrato; `README.pypi.md` corrigido e fiado no `pyproject.toml` com comentário inline dando a razão, ou descartado; §9.2 corrigido; §8 convertido em decisões adiadas; linhas de regime só onde há regime; ADR 0003 para Accepted; seção de política de compatibilidade em prosa.
*Verificação:* `pytest --doctest-glob='*.md' docs/` verde nos dois sistemas operacionais do CI; `twine check --strict`; um job que instala a wheel construída em ambiente limpo e importa o pacote; um script que lista cada bullet "Aceita"/"Rejeita" do §9 sem id de teste correspondente, e sai vazio.

**G3, higiene. Sem versão própria.** SECURITY, CITATION, templates, pre-commit, dependabot, badges, link refs, AGENTS.md.
*Verificação:* o community profile do GitHub fecha todos os itens.

---

## 5. O risco principal

**O predicado foi derivado dos mesmos dados usados para validá-lo, então a varredura não podia falsificar uma fronteira à qual tinha sido ajustada.** Isso já aconteceu duas vezes na mesma linhagem de documento: o primeiro rascunho disse "tiling exato", a reescrita disse `k_max <= 4` e exibiu 86 geometrias com zero erros. A segunda estava errada **enquanto exibia sua evidência**, porque `k_max <= 4` é necessário e não suficiente, e uma varredura de geometrias já corretas nunca o encontra em falta.

A correção é de procedimento, não de diligência: **derivar o predicado numérico da aritmética primeiro, depois enumerar a varredura independentemente do predicado e rodá-la para quebrá-lo.** A lista de eixos (valor do dado, ocupação da mantissa, forma da geometria, dtype, NaN) vira requisito do enumerador, não enunciado de risco, porque essa lista estará sempre um item incompleta.

Um detector, e é a única coisa desta análise que eu colocaria em um hook: **todo teste de exatidão tem as duas metades.** Exatidão dentro do predicado, e não exatidão fora dele, esta última como "pelo menos uma de N sementes é inexata" mais erro limitado por caso. Uma suíte que só verifica exatidão onde a espera nunca detecta um predicado largo demais, porque um predicado largo demais só erra na região que ela não visita.

Três apoios em volta: a lista de parâmetros sai de um enumerador com amostragem aleatória sobre o espaço legal, nunca de uma lista escrita à mão; a geração de dados passa por um helper único e auditado; e toda frase em prosa com "exato", "idêntico" ou "bit-exact" carrega o id do teste que a sustenta. Isso vale para os números do README também, incluindo o 191x do seam, que só é verdadeiro junto da geometria.