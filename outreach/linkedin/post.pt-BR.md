<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=pt-BR · canonical -->
**Português** · [English](post.en.md)

# Post curto para o LinkedIn

> Pronto para publicar. Todo número foi medido e é reproduzível no repositório.
> Fonte: [`../2026-09-04-lancamento.md`](../2026-09-04-lancamento.md).
>
> **O que este texto é:** o resumo que leva ao artigo e ao repositório. Objetivo e
> informativo, com fluxo, sem suspense e sem jogo de pergunta e resposta. Apresenta o
> assunto, diz o que o `fold`/`unfold` faz, mostra rapidamente os casos que pedem cuidado,
> e entrega o link. A teoria fica no artigo; a íntegra, no repositório.
>
>**Imagens para acompanhar**, em [`figuras/pt-BR/`](figuras/pt-BR/), com o SVG de cada uma
> ao lado para editar, geradas por `python tools/make_outreach_figures.py`. Nada é desenhado:
> cada painel é o tensor que aquele caminho devolve, e o dígito é do MNIST.
>
> A página com o texto que liga as três está em [`pagina.md`](pagina.md), separada das
> imagens de propósito, para você montar na ordem que quiser:
> `1-recorte.png`, `2-stride.png` e `3-mnist.png`.

---

**Dividir uma imagem em pedaços, processar cada um, e remontar**

Para o computador, uma imagem é uma matriz de números: cada pixel é um valor, e manipular a
imagem é fazer contas sobre essa matriz. Dá para trabalhar com ela inteira, e dá para
dividi-la em pedaços menores, tratar cada pedaço separadamente e juntar tudo de volta no
fim. Esses pedaços se chamam patches.

O PyTorch traz duas funções para isso. O `unfold` percorre a imagem com uma janela e
devolve todas as janelas empilhadas. O `fold` faz o caminho inverso, somando cada janela de
volta na posição de origem. A distância que a janela anda de um patch para o próximo é o
stride. Para recortes simples, com o stride igual ao tamanho da janela, as duas resolvem.

Fora desse caso aparecem detalhes que pedem cuidado. O `unfold` devolve os patches num
formato empacotado, e reorganizá-lo para a ordem intuitiva embaralha os pixels sem mudar a
forma do tensor. Quando o stride é menor que a janela, os patches se sobrepõem e o `fold`
soma as sobreposições, então remontar exige dividir cada pixel pelo número de vezes que ele
foi coberto. E quando o stride não fecha a imagem, a grade para antes da borda e o
resto volta em zero.

O PatchCraft cobre esses casos. Valida a geometria antes de recortar, divide pela cobertura
na remontagem, recusa configurações que descartariam pixels em silêncio, e documenta em que
condição a ida e a volta devolvem exatamente os mesmos bits. Traz também colagem com janela
de suavização, para quando a emenda entre patches fica visível, e um kernel nativo para o
caminho com sobreposição.

`pip install patchcraft`. Python 3.12 a 3.14, torch 2.6 ou mais novo, MIT, pré-1.0.

Escrevi um artigo com as medições, as figuras e os limites de onde isto se aplica hoje:
👉 https://www.linkedin.com/pulse/patchcraft-desmontar-e-montar-imagens-leonardo-marques-de-souza-q0mvf/

O código, as medições e a documentação do que não funciona:
👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
