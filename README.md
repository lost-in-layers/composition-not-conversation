# Composition, Not Conversation: VLMs Lose the Scene, Not the Thread

Code for **Layered-VQA**, a controlled benchmark that asks what vision-language
models lose when the same visual information is fragmented.

Every scene is split into ordered RGBA layers that exactly recompose the
original image, and every question is annotated with its supporting,
minimal-sufficient, and distractor layers. Sixteen delivery conditions keep the
pixels and the question fixed and change only how they are composed and
delivered.

- **Code:** this repository
- **Data:** [huggingface.co/lost-in-layer/Layered-VQA](https://huggingface.co/lost-in-layer/Layered-VQA)

## Findings

- **Loss in composition.** Splitting the question costs little. Splitting the
  scene into layers costs a lot, even when every layer arrives in one turn.
  Recomposing the same layers restores most of it.
- **Oracle inversion.** Showing only the annotated sufficient layers is worse
  than showing the whole scene.
- **Loss in grounding.** As questions need more evidence, naming the right
  evidence falls much faster than answer accuracy.

## Layout

    data/derived/lic/       the benchmark: 16 delivery conditions as JSON
    src/vecgen/             dataset construction and the LIC export
    configs/                annotation prompts, one file per question category
    vendor/conv-smi/        the simulator, judges, and scoring
    vendor/*-review/        the annotation and validation review apps
    analysis/               the scripts that produce every table and figure
    scripts/                data setup helpers
    tests/                  simulator tests
    docs/                   pipeline notes

## Data

The benchmark annotations (`data/derived/lic/*.json`) are in this repository.
The images are on Hugging Face:
[lost-in-layer/Layered-VQA](https://huggingface.co/lost-in-layer/Layered-VQA).

The full Hugging Face repository is about 110 GB, but the benchmark reads only
436 images (93 composite scenes and 343 layer renders, about 110 MB).
`scripts/setup_data.sh` downloads exactly those:

    bash scripts/setup_data.sh            # ~110 MB, what the benchmark reads
    FULL=1 bash scripts/setup_data.sh     # ~2.2 GB, every scene in the pool

Image paths inside the JSONs are relative to the JSON's own directory, so keep
the downloaded files where the script puts them.

## Reproducing

    # 1. fetch the images the benchmark reads
    bash scripts/setup_data.sh

    # 2. run the conditions for one model (from vendor/conv-smi)
    bash run_all_open_models.sh

    # 3. re-grade every conversation with every judge
    bash run_posthoc_judging.sh

    # 4. build the tables and figures
    python analysis/extract_behaviour.py <out-dir>      # from vendor/conv-smi
    python analysis/make_exp_a_table.py <csv-dir>
    python analysis/make_exp_b_table.py <csv-dir>
    python analysis/make_behaviour_tables.py <csv-dir>
    python analysis/make_ci_table.py <csv-dir>
    python analysis/make_figures.py <csv-dir> <fig-dir>

`vendor/conv-smi/CLAUDE.md` documents the simulator in detail, including which
condition maps onto which entry point.
