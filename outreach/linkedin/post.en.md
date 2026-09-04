<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=en · translation_of=post.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](post.pt-BR.md)

# Short LinkedIn post

> Ready to publish. Every number was measured and reproduces in the repository.
> The hook is the silent defect, the one thing here the reader might have in their own
> code right now. Source: [`../2026-09-04-release.en.md`](../2026-09-04-release.en.md).

---

**If you cut images into patches with `F.unfold`, there is a good chance you are
scrambling the pixels and getting no error at all.**

Torch's `F.unfold` returns `(1, C*ph*pw, L)`. The intuitive reshape into `(L, C, ph, pw)`
gives you the right shape with the wrong pixels. The shape `assert` passes, training runs,
the loss falls a little less, and there is no message anywhere.

Its neighbour is the stride that does not cover the image. On a 128 by 128 image with patch
32 and stride 20, the grid stops at pixel 112 and leaves 3840 of the 16384 pixels at zero.
A hand-rolled `fold` returns that partly black image, also without complaining.

I wrote those twenty lines more times than I would like to admit, which is why they became
a library. PatchCraft cuts an image into patches and puts it back, with tests around both
defects above.

What it claims about the numerics is a condition you evaluate **before** you call: the
round trip is bit-exact if and only if every value in the coverage count map is a power of
two, and outside that the per-pixel error is bounded by `(k+1)·eps·|v|`. That is computed
from the geometry alone, without running anything.

And there is a test whose explicit job is to bring that claim down. It enumerates the
126,736 legal geometries of the space without consulting the predicate, and hunts the two
counterexamples: a case inside the rule that is not exact, and a case outside it that is
exact by luck.

It has that shape because an earlier version of this contract was published, measured and
found false. I think a numerical library is worth less for the guarantee it announces and
more for the test it keeps pointed at its own guarantee.

Python 3.12 to 3.14, MIT, pre-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
