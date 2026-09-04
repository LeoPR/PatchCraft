<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=en · translation_of=post.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](post.pt-BR.md)

# Short LinkedIn post

> Ready to publish. Every number was measured and reproduces in the repository.
> Source: [`../2026-09-04-release.en.md`](../2026-09-04-release.en.md).
>
> **What this text is:** the summary that leads to the article and to the repository.
> Objective and informative, with flow, no suspense and no question-and-answer game. It
> introduces the subject, says what `fold`/`unfold` do, shows briefly the cases that need
> care, and hands over the link. The theory belongs in the article, the full account in
> the repository.
>
>**Images to go with it**, in [`figuras/en/`](figuras/en/), each with its SVG beside it for
> editing, produced by `python tools/make_outreach_figures.py`. Nothing is drawn: each panel
> is the tensor that path returns, and the digit is from MNIST.
>
> The page whose prose ties the three together is [`page.md`](page.md), kept separate from
> the images on purpose so they can be assembled in any order:
> `1-cut.png`, `2-stride.png` and `3-mnist.png`.

---

**Splitting an image into pieces, processing each one, and putting it back**

To a computer an image is a matrix of numbers: every pixel is a value, and working on the
image means doing arithmetic on that matrix. You can work on it whole, and you can split it
into smaller pieces, handle each piece on its own and join everything back at the end. Those
pieces are called patches.

PyTorch ships two functions for this. `unfold` slides a window across the image and returns
every window stacked. `fold` goes the other way, adding each window back into the position
it came from. For simple cuts, with the window moving one full size at a time, the two are
enough.

Beyond that case, details start asking for care. `unfold` returns the patches in a packed
layout, and rearranging it into the intuitive order scrambles the pixels without changing
the tensor's shape. When the step is smaller than the window the patches overlap and `fold`
sums the overlaps, so putting the image back means dividing each pixel by the number of
times it was covered. And when the step does not close the image, the grid stops before the
edge and the remainder comes back as zero.

PatchCraft covers those cases. It validates the geometry before cutting, divides by the
coverage on the way back, refuses configurations that would drop pixels silently, and
documents the condition under which the round trip returns exactly the same bits. It also
brings windowed blending, for when the seam between patches is visible, and a native kernel
for the overlapping path.

`pip install patchcraft`. Python 3.12 to 3.14, torch 2.6 or newer, MIT, pre-1.0.

The measurements and the documentation are open:
👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
