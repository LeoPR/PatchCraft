<!-- l10n: doc_id=patchcraft-outreach-linkedin-post · lang=en · translation_of=post.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](post.pt-BR.md)

# Short LinkedIn post

> Ready to publish. Every number was measured and reproduces in the repository.
> Source: [`../2026-09-04-release.en.md`](../2026-09-04-release.en.md).
>
> **Structure, in order:** one paragraph of context with no jargon, one that names the
> subject, the turn, the two defects, the library, the contract, the test, the close. The
> first two lines are the only ones that show before "see more", so they cannot contain a
> single word the reader has to know already.

---

**Every large image that goes into a neural network is cut into pieces first.**

It does not fit in memory whole, and even when it does, much of the work comes out better
piece by piece. So you cut it up, process each piece, and glue everything back at the end.
Those pieces have a name: patches.

It looks like a twenty-line problem. I have written those twenty lines more times than I
would like to admit, and got them wrong often enough to start distrusting them.

Here is how they go wrong.

On the cutting side, the torch function that does this job hands the pieces back in a
packed layout. The intuitive rearrangement into the layout you actually want gives you the
right shape with the wrong pixels. The shape `assert` passes. Training runs. The loss falls
a little less. Nothing warns you.

On the glueing side, the step from one piece to the next may not cover the whole image. On
a 128 by 128 image, with a piece of 32 and a step of 20, the grid stops at pixel 112 and
leaves 3840 of the 16384 pixels at zero. Almost a quarter of the image comes back black,
and the function returns it without raising anything.

Neither one is hard to fix. Both are easy to miss, and that difference is what justifies
writing it once, with tests around it, instead of rewriting it every project.

That is what I did, and it is called PatchCraft.

What it claims about its own arithmetic is a condition you evaluate **before** you call:
cutting and reassembling returns exactly the same bits if, and only if, every pixel is
covered by a number of pieces that is a power of two. Outside that, the error has a written
bound. You settle it by looking at the geometry, on paper, without running anything.

And there is a test whose job is to bring that claim down. It sweeps the 126,736 possible
geometries without consulting the rule, hunting for the case that contradicts it.

It has that shape because the first version of that claim was published, measured and found
false.

I think a numerical library is worth less for the guarantee it announces and more for the
test it keeps pointed at its own guarantee.

Python 3.12 to 3.14, MIT, pre-1.0. `pip install patchcraft`

👉 https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
