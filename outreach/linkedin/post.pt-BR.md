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
>**Imagens para acompanhar**, em [`figuras/pt-BR/`](figuras/pt-BR/), PNG para subir e SVG
> ao lado para editar, todas geradas por `python tools/make_outreach_figures.py`. Nenhuma é
> desenhada: os painéis são os tensores de verdade, os erros são medidos, e a recusa traz o
> texto que a biblioteca levanta. Vão nesta ordem:
>
> 1. [`fold-unfold.png`](figuras/pt-BR/fold-unfold.png), o problema, e é a mais forte das
>    três porque se lê sem legenda. A mesma imagem por quatro caminhos: original, o reshape
>    intuitivo que devolve a forma certa embaralhada, o passo 20 com a faixa preta, e o
>    passo 32 feito certo.
> 2. [`patchcraft.png`](figuras/pt-BR/patchcraft.png), a resposta, nos três regimes que a
>    biblioteca tem de fato: idêntica no passo 32 e no 16, aproximada no passo 12 com o erro
>    real ampliado no canto, e recusa no passo 20 que o `unfold` aceitaria.
> 3. [`cobertura.png`](figuras/pt-BR/cobertura.png), o mecanismo por trás das duas, se
>    quiser explicar por que o passo 12 sai aproximado e o 16 não.

---

**Dividir uma imagem em pedaços, processar cada um, e remontar**

Para o computador, uma imagem é uma matriz de números: cada pixel é um valor, e manipular a
imagem é fazer contas sobre essa matriz. Dá para trabalhar com ela inteira, e dá para
dividi-la em pedaços menores, tratar cada pedaço separadamente e juntar tudo de volta no
fim. Esses pedaços se chamam patches.

O PyTorch traz duas funções para isso. O `unfold` percorre a imagem com uma janela e
devolve todas as janelas empilhadas. O `fold` faz o caminho inverso, somando cada janela de
volta na posição de origem. Para recortes simples, com a janela andando um tamanho inteiro
por vez, as duas resolvem.

Fora desse caso aparecem detalhes que pedem cuidado. O `unfold` devolve os patches num
formato empacotado, e reorganizá-lo para a ordem intuitiva embaralha os pixels sem mudar a
forma do tensor. Quando o passo é menor que a janela, os patches se sobrepõem e o `fold`
soma as sobreposições, então remontar exige dividir cada pixel pelo número de vezes que ele
foi coberto. E quando o passo não fecha a imagem, a grade para antes da borda e o resto
volta em zero.

O PatchCraft cobre esses casos. Valida a geometria antes de recortar, divide pela cobertura
na remontagem, recusa configurações que descartariam pixels em silêncio, e documenta em que
condição a ida e a volta devolvem exatamente os mesmos bits. Traz também colagem com janela
de suavização, para quando a emenda entre patches fica visível, e um kernel nativo para o
caminho com sobreposição.

`pip install patchcraft`. Python 3.12 a 3.14, torch 2.6 ou mais novo, MIT, pré-1.0.

As medições e a documentação estão abertas:
👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #VisaoComputacional #EngenhariaDeSoftware
