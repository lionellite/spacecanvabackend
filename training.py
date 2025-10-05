import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import json
from datetime import datetime
from database import save_training_session, update_training_session

# Training configuration
CURVE_LENGTH = 2001
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def generate_transit_curve(period, duration, depth):
    """
    Generate a synthetic transit light curve from period, duration, and depth.
    """
    depth_fraction = depth / 1e6
    time = np.linspace(-0.5, 0.5, CURVE_LENGTH)
    duration_phase = (duration / 24.0) / period
    
    curve = np.ones(CURVE_LENGTH)
    transit_mask = np.abs(time) < (duration_phase / 2)
    curve[transit_mask] = 1.0 - depth_fraction
    
    # Add smooth edges
    edge_width = int(CURVE_LENGTH * 0.02)
    for i in range(len(curve)):
        if transit_mask[i]:
            dist_to_edge = min(
                abs(i - np.where(transit_mask)[0][0]),
                abs(i - np.where(transit_mask)[0][-1])
            )
            if dist_to_edge < edge_width:
                smooth_factor = dist_to_edge / edge_width
                curve[i] = 1.0 - depth_fraction * smooth_factor
    
    return curve

def prepare_dataset_from_csv(csv_path):
    """
    Prepare training dataset from CSV file
    
    Expected CSV columns:
    - period, duration, depth, impact, snr, steff, srad, slogg, tmag, label
    
    Label encoding:
    - 0: CANDIDATE
    - 1: CONFIRMED
    - 2: FALSE POSITIVE
    """
    print(f"📊 Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['period', 'duration', 'depth', 'impact', 'snr', 'steff', 'srad', 'slogg', 'tmag', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    
    # Generate curve inputs
    print("🔄 Generating transit curves...")
    curves = []
    for idx, row in df.iterrows():
        curve = generate_transit_curve(row['period'], row['duration'], row['depth'])
        curves.append(curve)
        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{len(df)} curves")
    
    curve_input = np.array(curves).reshape(-1, CURVE_LENGTH, 1)
    
    # Prepare feature inputs (excluding period, duration, depth)
    feat_input = df[['impact', 'snr', 'steff', 'srad', 'slogg', 'tmag']].values
    
    # Prepare labels (one-hot encoding for classification)
    labels = keras.utils.to_categorical(df['label'].values, num_classes=3)
    
    # Prepare regression targets
    period_target = df['period'].values.reshape(-1, 1)
    depth_target = (df['depth'].values / 1e6).reshape(-1, 1)  # Convert to fraction
    
    print(f"✅ Dataset prepared:")
    print(f"  - Curve input shape: {curve_input.shape}")
    print(f"  - Feature input shape: {feat_input.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return {
        'curve_input': curve_input,
        'feat_input': feat_input,
        'labels': labels,
        'period_target': period_target,
        'depth_target': depth_target
    }

def create_model():
    """
    Create the exoplanet detection model
    Multi-input, multi-output architecture
    """
    # Input 1: Transit curve
    curve_input = keras.Input(shape=(CURVE_LENGTH, 1), name='curve_input')
    
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

def train_model(dataset, workspace_id, model_name='exoplanet_model', epochs=EPOCHS):
    """
    Train the model on the provided dataset
    """
    print("🚀 Starting model training...")
    
    # Create training session in database
    training_id = save_training_session(
        workspace_id=workspace_id,
        model_name=model_name,
        dataset_size=len(dataset['labels']),
        status='training'
    )
    
    try:
        # Create model
        print("🏗️ Creating model architecture...")
        model = create_model()
        model.summary()
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_label_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            TrainingProgressCallback(training_id)
        ]
        
        # Train model
        print(f"🎯 Training for {epochs} epochs...")
        history = model.fit(
            x=[dataset['curve_input'], dataset['feat_input']],
            y={
                'label': dataset['labels'],
                'period_out': dataset['period_target'],
                'depth_out': dataset['depth_target']
            },
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'{model_name}_{timestamp}.keras')
        model.save(model_path)
        
        # Also save as current model
        current_model_path = os.path.join(model_dir, 'exoplanet_model.keras')
        model.save(current_model_path)
        
        # Get final metrics
        final_metrics = {
            'label_accuracy': float(history.history['label_accuracy'][-1]),
            'val_label_accuracy': float(history.history['val_label_accuracy'][-1]),
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1])
        }
        
        # Update training session
        update_training_session(
            training_id=training_id,
            status='completed',
            model_path=model_path,
            metrics=json.dumps(final_metrics),
            epochs_completed=len(history.history['loss'])
        )
        
        print(f"✅ Training completed!")
        print(f"📊 Final metrics:")
        print(f"  - Accuracy: {final_metrics['label_accuracy']:.4f}")
        print(f"  - Val Accuracy: {final_metrics['val_label_accuracy']:.4f}")
        print(f"  - Loss: {final_metrics['loss']:.4f}")
        print(f"  - Val Loss: {final_metrics['val_loss']:.4f}")
        print(f"💾 Model saved to: {model_path}")
        
        return {
            'training_id': training_id,
            'model_path': model_path,
            'metrics': final_metrics,
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
        
    except Exception as e:
        # Update training session as failed
        update_training_session(
            training_id=training_id,
            status='failed',
            metrics=json.dumps({'error': str(e)})
        )
        raise e

class TrainingProgressCallback(keras.callbacks.Callback):
    """Callback to update training progress in database"""
    
    def __init__(self, training_id):
        super().__init__()
        self.training_id = training_id
    
    def on_epoch_end(self, epoch, logs=None):
        # Update progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = {
                'epoch': epoch + 1,
                'accuracy': float(logs.get('label_accuracy', 0)),
                'val_accuracy': float(logs.get('val_label_accuracy', 0)),
                'loss': float(logs.get('loss', 0))
            }
            update_training_session(
                training_id=self.training_id,
                epochs_completed=epoch + 1,
                metrics=json.dumps(metrics)
            )

def evaluate_model(model_path, test_dataset):
    """
    Evaluate a trained model on test data
    """
    print(f"📊 Evaluating model: {model_path}")
    
    model = keras.models.load_model(model_path)
    
    results = model.evaluate(
        x=[test_dataset['curve_input'], test_dataset['feat_input']],
        y={
            'label': test_dataset['labels'],
            'period_out': test_dataset['period_target'],
            'depth_out': test_dataset['depth_target']
        },
        verbose=1
    )
    
    print("✅ Evaluation completed")
    return results
