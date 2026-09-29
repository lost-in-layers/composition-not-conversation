# MuLAn Category Review

Separate single-reviewer UI for category selection over the shared `selected_1200` scene cache. The existing `vendor/human-review` application and its state are not modified.

```powershell
powershell -ExecutionPolicy Bypass -File vendor\category-review\run.ps1 -Port 8792
```

The app autosaves drafts and final decisions under `vendor/category-review/runtime/`. Accepted selections are atomically exported after every decision to:

```text
data/derived/mulan-category-review/category_selections.json
```

Each exported entry contains only `image_id` and its category selection (`primary` plus `selected`). The header tracks accepted category-question slots toward the target of 350.
