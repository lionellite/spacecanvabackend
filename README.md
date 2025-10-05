# Backend - Exoplanet Analysis API

## Installation

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your model:
```bash
mkdir -p model
# Place your exoplanet_model.keras file in the model/ directory
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```

### Predict Exoplanet
```
POST /api/predict
Content-Type: application/json

{
  "mission": "kepler",
  "period": 3.52,
  "duration": 2.5,
  "depth": 1000,
  "impact": 0.5,
  "snr": 15.2,
  "steff": 5778,
  "srad": 1.0,
  "slogg": 4.5,
  "tmag": 12.5
}
```

Response:
```json
{
  "prediction": {
    "is_exoplanet": true,
    "probability": 0.95,
    "confidence": 0.95,
    "classification": "Confirmed Exoplanet"
  },
  "metadata": {
    "mission": "kepler",
    "period": 3.52,
    ...
    "estimated_planet_radius": 1.2
  },
  "analysis": {
    "transit_quality": "good",
    "transit_type": "central",
    "star_type": "main_sequence"

### Inputs (2)

1. **curve_input** : Shape (batch_size, CURVE_LENGTH, 1)
   - Courbe de lumière générée à partir de period, duration, depth
   - CURVE_LENGTH = 2001 points par défaut (ajustable)

2. **feat_input** : Shape (batch_size, 6)
   - Features supplémentaires (dans cet ordre):
     1. impact parameter
     2. SNR
     3. stellar temperature (K)
     4. stellar radius (R☉)
     5. stellar log g (dex)
     6. T magnitude

### Outputs (3)

1. **label** : Shape (batch_size, 3)
   - Probabilités pour 3 classes : [CANDIDATE, CONFIRMED, FALSE POSITIVE]
   - Activation : softmax

2. **period_out** : Shape (batch_size, 1)
   - Période orbitale prédite (jours)
   - Activation : linéaire

3. **depth_out** : Shape (batch_size, 1)
   - Profondeur du transit prédite (fraction, pas ppm)
   - Activation : linéaire
