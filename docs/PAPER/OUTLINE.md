# Outline

**Working title.** A mean that never leaves its domain: divide first, then sum.

## Working abstract

The mean of values in a numeric domain lies in that domain, but the usual
computation, sum then divide, passes through a sum that may not. Libraries
answer by widening the accumulator, which works while a wider type exists and
fails silently when the width is chosen too small. We study the other order,
dividing first. For unsigned integers we give an online quotient-and-remainder
mean that returns the exact floor or rounded mean with every register in the
input's own width, for up to `2^w - 1` terms, and we show that the naive form,
flooring each term, is biased low by `(n - 1)/2` units on average. For floating
point, dividing first removes the range problem only under a stated
condition, keeps the worst-case error bound, and is free when `n` is a power
of two. On a patch grid the same analysis explains an exactness rule: in a
legal geometry the counts of a power-of-two coverage map are 1, 2 and 4, the
only small counts for which adding copies one at a time is exact. Every claim
comes with a measurement on synthetic data.

## Contributions

1. The online carry mean for unsigned integers, exact, with every register in
   the input's width (FORMAL Proposition 4; I4).
2. The failure modes of the alternatives, each stated and measured: the
   widened accumulator wraps (Proposition 1; I1), the naive divide-first is
   biased (Proposition 2; I2), and two separate sums overflow from `n = 17` in
   `uint8` (Proposition 3; I3).
3. What dividing first does and does not buy in floating point (Propositions
   8 and 9; F3 to F5).
4. The exactness of a patch-grid mean, reduced to Propositions 6 and 10, with
   the `float16` fold as the range case (Proposition 11; P1 to P4).

## Sections

1. **Introduction.** A mean's result is always in the domain; its
   computation need not be. Two cases frame the paper: averaging `uint8`
   camera frames, and putting an image back together from overlapping
   patches. Contributions.
2. **Related work.**
   - The precision analysis of summation assumes the range: no overflow,
     usually no underflow [higham1993; croci2022; ogita2005; higham2002asna].
   - Two values: the integer midpoint and its overflow [bloch2006;
     ruggieri2003; warren2013; dietz-magic; wong1999; chen2022; dietz2012], the
     floating-point midpoint [boldo2015; goualard2014], and the standard
     library [herring2019-p0811r3; cppref-midpoint; cxx-draft-midpoint].
   - Running means and their stability [welford1962; west1979; chan1983].
   - Systems widen the accumulator [pytorch-accumulatetype;
     pytorch-opmathtype; numpy-mean-docs; numpy-methods-src;
     micikevicius-2018; cmsis-dsp-mean; intel-sdm-pavgb; haeberli-akeley-1990;
     higham2019squeeze].
   - Imaging blends overlapping estimates by summing then dividing
     [knutsson-westin-1993; piroddi-petrou-2003; pytorch-fold-docs;
     dabov-2007; elad-aharon-2006; swinir-code; wronski-2019; szeliski-2006],
     or by windows normalized to sum to one first [hasinoff-2016;
     pielawski-2020; glaccum-manpage; redbook-accpersp].
   - The gap: the precedents for dividing first are for two values; no
     source found gives the general-`n` integer form (see Open questions).
3. **Integer means.** Propositions 1 to 5, measurements I1 to I5. The online
   carry is the centre of the paper: the naive divide-first made exact.
4. **Floating-point means.** Propositions 6 to 9, measurements F1 to F6. The
   honest version of the thesis: in floating point, dividing first is a
   condition, not a guarantee.
5. **The patch grid.** Propositions 10 and 11, measurements P1 to P4, with
   PatchCraft's exactness rule as the corollary and its `float16`
   promotion as the widening answer.
6. **Case study: frame averaging in an optical mark reader** (below).
7. **Discussion.** When to widen and when to divide first; what the carry
   costs; where the domain argument stops (signed values, weights).
8. **Conclusion.**

## Case study: frame averaging in an optical mark reader

An optical mark reader for lottery betting slips, written in 2024 under a
commercial contract, averaged several camera frames before reading the marks,
to suppress sensor noise. Its code is private. The paper cites the averaging
functions by file, line and date and reproduces only their pattern; no image,
document or data of that project enters this repository, and every
measurement here is taken on synthetic frames.

- `Monitor_V4.py:117-126` (2024-05-15) and `check_static_V1605.py:14-30`
  (2024-05-16) sum the frames in `uint32` and floor-divide; the second also
  clips the result to `[10, 200]`. A comment in a copy of the same function
  (`libs/check_static.py:27-28`, 2024-05-16) records why it avoided `np.mean`:
  the default `float64` path was too slow.
- `check_static_20052024.py:34-41` (2024-05-20) and `Monitor.py:128-131`
  (2024-05-22) use `np.mean(frames, axis=0, dtype=np.uint16).astype(np.uint8)`.
- `test.py:45-61` (2024-06-15) is the author's own working note: the sum of
  three `uint8` frames can pass 255, so divide each frame first and accept
  losing a little precision to stay in `uint8`.
- `libs/check_static.py:34-49` (2024-08-05), the library version, takes the
  first, middle and last frame and does exactly that: divide first, floor
  each term.

What the paper says about it, all measured on synthetic frames:

- The reader used 3 to 15 frames (`Monitor.py:211`, `check_static_V1605.py:82`),
  so the `uint16` accumulator, which wraps from 258 frames of 255 (I1), never
  wrapped there. The widening was sound for that use and would fail silently
  at a longer exposure.
- The library's divide-first is biased by about one gray level at three
  frames (I2; Proposition 2). Whether one level mattered to the mark
  thresholds is a question about that reader, which the paper does not
  answer.
- The online carry (Proposition 4) is the note's idea made exact: same
  registers, no bias (I4).

## Open questions

1. **Novelty of the online carry.** No source found gives the general-`n`
   integer quotient-and-remainder mean with an online carry; every precedent
   is for two values. Knuth's volume 2 and Sterbenz were not opened, and the
   fixed-point DSP and hardware literature was not searched. That search comes
   before any claim of novelty.
2. **Cost.** The carry is a handful of integer operations per term. Its price
   against a widened accumulator has to be measured in a compiled
   implementation; a numpy measurement mostly times numpy's temporaries.
3. **Proposition 6, converse.** That every count other than 1, 2 and 4 fails
   for some value is measured up to 16 in `float16` and at 3, 8, 16 and 32 in
   `float32`, not proved.
4. **Weights.** Windows that sum to one keep every partial sum bounded but
   give up exactness; the research record has the measurement for the HDR+
   raised cosine, and it is not yet ported here.
5. **Signed integers and other roundings** (ties to even, toward zero).
6. **Venue and format.** A short paper is the likely shape; the typesetting
   comes after the content is settled.
