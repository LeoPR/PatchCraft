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
>**Imagem para acompanhar:** [`figuras/pt-BR/pagina.png`](figuras/pt-BR/pagina.png), uma
> página só, com o SVG ao lado para editar, gerada por
> `python tools/make_outreach_figures.py`. Nada nela é desenhado: os painéis são os tensores
> que cada caminho devolve, e o dígito é do MNIST.
>
> Três blocos, na ordem do texto: o recorte e o reshape que embaralha; a comparação por
> passo entre o `fold`/`unfold` à mão e o PatchCraft, com o mapa de erro explicado; e o caso
> típico, um dígito 28x28 com patch 7, onde a geometria fecha exata.

---

**Dividir uma imagem em pedaços, processar cada um, e remontar**

Para o computador, uma imagem é uma matriz de números: cada pixel é um valor, e manipular a
imagem é fazer contas sobre essa matriz. Dá para trabalhar com ela inteira, e dá para
dividi-la em pedaços menores, tratar cada pedaço separadamente e juntar tudo de volta no
fim. Esses pedaços se chamam patches.

O PyTorch traz duas funções para isso. O `unfold` percorre a imagem com uma janela e
devolve todas as janelas empilhadas. O `fold` faz o caminho inverso, somando cada janela de
volta na posição de origem. A distância que a janela anda entre um patch e o próximo chama-se
passo, ou `stride`, que é o nome que você vai ver em toda documentação. Para recortes
simples, com a janela andando um tamanho inteiro por vez, as duas funções resolvem.

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

As medições e a documentação estão abertas:
👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
