# Integer arithmetic in the quantized scan

`QuantizedFlatIndex` scans with narrow item codes and a floating-point query. A natural
extension is to quantize the query as well and run the inner loop as
int8×int8→int32, which is where real quantized indexes get their speed: one ARM `udot`
does sixteen multiply-accumulates against NEON's two `f64` ones. It was built and
measured, and not adopted. This note records why.

The arithmetic works out cleanly — an integer accumulator can only hold `Σ qc·vc`, so
every scale has to come out of the sum:

```text
<q,v> ~= sq·sv·Σ(qc·vc) + sq·ov·Σqc + oq·sv·Σvc + oq·ov·dim
```

with `Σvc` precomputed per item. Three findings decided against it:

1. **It was worth 1.26x, not 7x.** Measured on the 60,000-item catalog: `5.44` → `4.33`
   ns per candidate at eight bits, `5.66` → `3.97` at four. Still `4.8x` slower than the
   exact path, so nothing about the decision changes.
2. **`udot` never appeared.** LLVM emits `u32` multiplies from safe Rust, not the
   sixteen-lane widening dot product the estimate assumed. Reaching it needs
   `std::arch::aarch64` intrinsics, which are `unsafe`, and this crate does not use
   `unsafe` anywhere. That is a deliberate property worth more than 1.3x.
3. **It set a trap for `EASE`.** An integer accumulator forces one global scale in place
   of the per-dimension ones. `EASE`'s space is dense, so it qualified at fit time — but
   it always scores through *sparse* queries, so it fell back to the float kernel and paid
   the accuracy cost for none of the speed. Recall went to `0.059` at four bits and
   `0.000` at two, against `0.998` with per-dimension scales. Silent, catastrophic, and
   exactly the "fast and wrong" outcome the index benchmarks in the README exist to catch.

Tiling survived because it is free: it changes the loop order, not the numbers. Integer
arithmetic is not free — it trades away per-dimension resolution — and at 1.26x the trade
does not pay.
