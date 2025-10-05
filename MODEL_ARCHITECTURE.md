# 🧠 Architecture du Modèle d'Exoplanète

## 📊 Vue d'ensemble

Le modèle utilise une architecture **multi-input/multi-output** pour analyser les candidats exoplanètes.

## 🔌 Inputs (2)

### 1. curve_input
- **Shape** : `(batch_size, CURVE_LENGTH, 1)`
- **Description** : Courbe de lumière du transit
- **Génération** : Créée automatiquement à partir de :
  - `period` : Période orbitale (jours)
  - `duration` : Durée du transit (heures)
  - `depth` : Profondeur du transit (ppm)
- **CURVE_LENGTH** : 2001 points par défaut (configurable dans `app.py`)

### 2. feat_input
- **Shape** : `(batch_size, 6)`
- **Description** : Features supplémentaires (hors period, duration, depth)
- **Ordre des features** :
  1. `impact` : Paramètre d'impact
  2. `snr` : Rapport signal/bruit
  3. `steff` : Température stellaire (K)
  4. `srad` : Rayon stellaire (R☉)
  5. `slogg` : Log g stellaire (dex)
  6. `tmag` : Magnitude T

## 🎯 Outputs (3)

### 1. label
- **Shape** : `(batch_size, 3)`
- **Description** : Probabilités de classification
- **Classes** :
  - Index 0 : `CANDIDATE` (candidat)
  - Index 1 : `CONFIRMED` (confirmé)
  - Index 2 : `FALSE POSITIVE` (faux positif)
- **Activation** : `softmax`
- **Utilisation** : `np.argmax(label_probs)` pour obtenir la classe prédite

### 2. period_out
- **Shape** : `(batch_size, 1)`
- **Description** : Période orbitale prédite
- **Unité** : Jours
- **Activation** : Linéaire
- **Utilisation** : Vérifier la cohérence avec l'input

### 3. depth_out
- **Shape** : `(batch_size, 1)`
- **Description** : Profondeur du transit prédite
- **Unité** : **Fraction** (pas ppm !)
- **Conversion** : `depth_ppm = depth_fraction * 1e6`
- **Activation** : Linéaire
- **Utilisation** : Vérifier la cohérence avec l'input

## 🔄 Flux de Prédiction

```python
# 1. Générer curve_input
curve = generate_transit_curve(period, duration, depth)
curve_input = curve.reshape(1, CURVE_LENGTH, 1)

# 2. Préparer feat_input
feat_input = np.array([impact, snr, steff, srad, slogg, tmag]).reshape(1, 6)

# 3. Faire la prédiction
predictions = model.predict([curve_input, feat_input])

# 4. Extraire les résultats
label_probs = predictions[0][0]  # (3,)
period_pred = predictions[1][0][0]  # scalar
depth_pred = predictions[2][0][0]  # scalar (fraction)

# 5. Déterminer la classe
predicted_class = np.argmax(label_probs)
confidence = label_probs[predicted_class]
```

## 📐 Exemple de Réponse API

```json
{
  "prediction": {
    "is_exoplanet": true,
    "label": "CONFIRMED",
    "label_probabilities": {
      "CANDIDATE": 0.15,
      "CONFIRMED": 0.82,
      "FALSE_POSITIVE": 0.03
    },
    "confidence": 0.82,
    "classification": "Confirmed Exoplanet"
  },
  "model_predictions": {
    "predicted_period": 3.48,
    "predicted_depth_ppm": 985.2,
    "input_period": 3.52,
    "input_depth_ppm": 1000,
    "period_error": 0.04,
    "depth_error_ppm": 14.8
  },
  "metadata": {
    "period": 3.52,
    "duration": 2.5,
    "depth": 1000,
    "impact": 0.5,
    "snr": 15.2,
    "steff": 5778,
    "srad": 1.0,
    "slogg": 4.5,
    "tmag": 12.5,
    "estimated_planet_radius": 1.15
  },
  "analysis": {
    "transit_quality": "good",
    "transit_type": "central",
    "star_type": "main_sequence",
    "model_consistency": "high"
  }
}
```

## ⚙️ Configuration

### Ajuster CURVE_LENGTH

Dans `backend/app.py`, ligne 15 :
```python
CURVE_LENGTH = 2001  # Changez selon votre modèle
```

### Fonction de Génération de Courbe

La fonction `generate_transit_curve()` crée une courbe simplifiée. Pour une meilleure précision, vous pouvez :
- Utiliser un modèle physique plus sophistiqué (limb darkening, etc.)
- Charger des courbes pré-calculées
- Ajuster les paramètres d'ingress/egress

## 🧪 Test du Modèle

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "period": 3.52,
    "duration": 2.5,
    "depth": 1000,
    "impact": 0.5,
    "snr": 15.2,
    "steff": 5778,
    "srad": 1.0,
    "slogg": 4.5,
    "tmag": 12.5
  }'
```

## 📝 Notes Importantes

1. **Depth** : L'input est en ppm, mais le modèle prédit en fraction
2. **Cohérence** : Le backend calcule automatiquement les erreurs entre input et prédictions
3. **Classification** : 3 classes au lieu de binaire (CANDIDATE/CONFIRMED/FALSE POSITIVE)
4. **Curve** : Générée automatiquement, pas besoin de la fournir en input
