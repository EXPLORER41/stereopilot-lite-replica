# Stereo4D Data Notes

The current real-data smoke tests used HuggingFace pre-rectified Stereo4D-derived left/right perspective clips:

- `KevinMathew/stereo4d-lefteye-perspective`
- `KevinMathew/stereo4d-righteye-perspective`

These are not committed to this repo. They are suitable for research prototype validation only, not commercial training.

Prepared training format:

```text
left/
  clip_000001.mp4
right/
  clip_000001.mp4
  clip_000001.txt
manifest.jsonl
```

The current preprocessing target is:

```text
512x288
16fps
84 frames saved
81-frame training bucket
domain_label = "parallel"
```

The extra frames are intentional so mp4 duration/frame estimation does not accidentally drop the sample below the
81-frame bucket.
