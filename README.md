## PPE Detection API

YOLOv8m-OBB model for detecting Personal Protective Equipment (hard hats, wrong headgear).

### Quick Start

1. Clone & install:
   ```bash
   git clone https://github.com/coluong25/Personal-Protective-Equipment-Detection.git
   cd Personal-Protective-Equipment-Detection
   conda create -n ppe-detect python=3.10
   conda activate ppe-detect
   pip install -r requirements.txt
   ```

2. [Download the pretrained model](https://drive.google.com/drive/folders/1RAe0WY32povuZAXzHA0VJLdTCOZRDkLm) and place it at `models/final/`.

3. Run:
   ```bash
   uvicorn app.main:app --reload
   ```

### API Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| GET | `/health` | — | model status |
| POST | `/predict/json` | image file | JSON detections |
| POST | `/predict/view` | image file | annotated JPEG |
| POST | `/predict/base64/json` | `{"image_base64": "..."}` | JSON detections |

Swagger UI available at `http://localhost:8000/docs`.

### Config

Edit thresholds in `src/test_external.py`:

```python
CLASS_CONF = {
    0: 0.2,   # bare_head
    1: 0.62,  # constructor_hat
    2: 0.56,  # wrong_headgear_type
}
```

### Limitations

- `wrong_headgear_type` covers many hat types — class ambiguity
- Weaker on small/distant objects
- Static class weights in `data.yaml` — not flexible when class sample or data distribution changes

### Future Work

- Split `wrong_headgear_type` into more specific subclasses
- Apply SAHI + higher resolution (1280px) for small object detection

---

> Training data: [Roboflow](https://app.roboflow.com/ds/mbshq5SS7k?key=Px0FUE856s)
