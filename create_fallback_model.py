import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def create_basic_model():
    """
    Create a basic exoplanet model for testing purposes
    """
    # Input 1: Transit curve
    curve_input = keras.Input(shape=(2001, 1), name='curve_input')

    # CNN for curve processing
    x = keras.layers.Conv1D(32, 11, activation='relu', padding='same')(curve_input)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 7, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Input 2: Features
    feat_input = keras.Input(shape=(6,), name='feat_input')

    # Feature processing
    y = keras.layers.Dense(64, activation='relu')(feat_input)
    y = keras.layers.Dropout(0.3)(y)
    y = keras.layers.Dense(32, activation='relu')(y)

    # Concatenate
    combined = keras.layers.concatenate([x, y])

    # Shared layers
    z = keras.layers.Dense(128, activation='relu')(combined)
    z = keras.layers.Dropout(0.4)(z)
    z = keras.layers.Dense(64, activation='relu')(z)

    # Output 1: Classification (3 classes)
    label_output = keras.layers.Dense(3, activation='softmax', name='label')(z)

    # Output 2: Period regression
    period_output = keras.layers.Dense(1, name='period_out')(z)

    # Output 3: Depth regression
    depth_output = keras.layers.Dense(1, name='depth_out')(z)

    # Create model
    model = keras.Model(
        inputs=[curve_input, feat_input],
        outputs=[label_output, period_output, depth_output]
    )

    # Compile with multiple losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'label': 'categorical_crossentropy',
            'period_out': 'mse',
            'depth_out': 'mse'
        },
        loss_weights={
            'label': 1.0,
            'period_out': 0.5,
            'depth_out': 0.5
        },
        metrics={
            'label': ['accuracy'],
            'period_out': ['mae'],
            'depth_out': ['mae']
        }
    )

    return model

def create_fallback_model():
    """
    Create a simple fallback model that always returns safe predictions
    """
    model = create_basic_model()

    # Create a simple dataset for training
    np.random.seed(42)

    # Generate some fake training data
    num_samples = 100
    curve_length = 2001

    # Fake curves (mostly flat with some noise)
    curves = np.random.normal(1.0, 0.01, (num_samples, curve_length, 1))

    # Fake features
    features = np.random.uniform(0, 1, (num_samples, 6))

    # Fake labels (balanced)
    labels = keras.utils.to_categorical(np.random.randint(0, 3, num_samples), num_classes=3)

    # Fake period targets
    period_targets = np.random.uniform(1, 100, (num_samples, 1))

    # Fake depth targets (in fraction)
    depth_targets = np.random.uniform(0.001, 0.01, (num_samples, 1))

    print("🧠 Training fallback model...")
    model.fit(
        [curves, features],
        {'label': labels, 'period_out': period_targets, 'depth_out': depth_targets},
        epochs=5,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    return model

if __name__ == "__main__":
    # Create and save a basic model
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'exoplanet_model.keras')

    if not os.path.exists(model_path):
        print("🏗️ Creating fallback model for testing...")
        model = create_fallback_model()
        model.save(model_path)
        print(f"✅ Model saved to {model_path}")
    else:
        print(f"✅ Model already exists at {model_path}")
