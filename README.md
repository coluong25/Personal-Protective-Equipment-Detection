## PPE Detection — Pretrained Model

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/coluong25/Personal-Protective-Equipment-Detection.git
   cd Personal-Protective-Equipment-Detection
   ```

2. Create and activate environment:
   ```bash
   conda create -n ppe-detect python=3.10
   conda activate ppe-detect
   pip install -r requirements.txt
   ```

3. Download the pretrained model and place it in `models/final/`:
   [Google Drive — model](https://drive.google.com/drive/folders/1RAe0WY32povuZAXzHA0VJLdTCOZRDkLm)

4. Place your images in `data/test/images/`
   (or use the provided test set: [Google Drive — test data](https://drive.google.com/drive/folders/19XcQr78xkkTVzqFxQcCNNh9BqE2QMm5p))

### Run

```bash
python src/test_external.py
```

Results are saved to `outputs/test_external/`.

### Config

Both settings are in `src/test_external.py`.

**`CLASS_CONF`** — confidence threshold per class:
```python
CLASS_CONF = {
    0: 0.2,   # bare_head       — raise to reduce false alarms
    1: 0.62,  # constructor_hat — lower to catch more helmets
    2: 0.56,  # wrong_headgear_type
}
```

**`CLASS_PRIORITY`** — tiebreaker when two boxes overlap with similar confidence:
```python
CLASS_PRIORITY = {
    0: 1,  # bare_head       — lowest priority
    1: 3,  # constructor_hat — highest priority
    2: 2,  # wrong_headgear_type
}
```

### Limitations

- `wrong_headgear_type` includes many hat types (e.g. bicycle helmets vs. hard hats) — class ambiguity
- Weaker on small/distant objects

### Future Work

- Split `wrong_headgear_type` into more specific subclasses
- Apply SAHI (slicing inference) and higher input resolution (1280px) to improve detection of small/distant objects

---

Raw dataset: https://app.roboflow.com/ds/mbshq5SS7k?key=Px0FUE856s
