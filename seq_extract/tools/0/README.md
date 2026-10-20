# GIF helper working directory

This initially empty directory was created by the GIF helper for the example basename `0`. It is an intermediate output directory, not an executable tool or a model. The verified animation and its frame PNGs were retained under `.local_setup/verification/coarse_extra/00_sum/`. The complete extractor below also invokes the GIF helper while saving the sequence.

## Launch

Use the [local environment launcher](../../../.local_setup/activate.sh) from the repository root:

```bash
# Run from the repository root.
source .local_setup/activate.sh coarse
cd seq_extract
python test.py --input ../.local_setup/verification/coarse_input --model new_train_phase_2
```

## Expected result

Expected result: an animation gradually draws the recovered character. The existing verified animation is embedded below. Main sequence data and comparison PNGs are under `seq_extract/outputs/sampling/__new_train_phase_2/`.

![Verified extraction example](../../../.local_setup/verification/coarse_extra/00_sum/dynamic.gif)

The extractor chooses a random initial cursor, so a rerun can vary in stroke order and exact appearance. The embedded image is an existing verified output. Model sources and hashes are listed in the [asset manifest](../../../.local_setup/downloads/assets.json).
