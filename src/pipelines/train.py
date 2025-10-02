#####################################################################################
# ----------------------------------- Model Trainer ---------------------------------
#####################################################################################

########################
# ---- Dependencies ----
########################

import os
import pathlib
import ast
from src.core.load_env import EnvLoader
import numpy as np
import mlflow
import mlflow.keras as mlflow_keras
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.pipelines.preprocess import split_and_balance_dataset
from src.builders.builders import ModelBuilder
from src.utils.utils import flatten_data


#######################
# ---- Fine Tunner ----
#######################

def train(backbone_name: str = 'VGG16', split_ratios: tuple = (0.7, 0.15, 0.15), balanced: str = "oversample"):
    """
    Función principal para orquestar el proceso de entrenamiento y la búsqueda
    de hiperparámetros con Keras Tuner, rastreando los experimentos con MLflow.

    Args:
        backbone_name (str): Nombre del modelo base a usar. Options: 'VGG16', 'ResNet50', 'YOLO'.
        split_ratios (tuple): Ratios de división para train, val y test. Must sum to 1.0.
        balanced (str): Estrategia de balanceo. Options: "oversample", "downsample", or "none".

    Returns:
        tuple: (kt.Tuner object with search results, (X_test, y_test) test data)

    Raises:
        ValueError: If split_ratios don't sum to 1.0 or if invalid backbone_name provided.
    """
    # --- 0. VALIDACIÓN DE PARÁMETROS ---
    if not isinstance(split_ratios, tuple) or len(split_ratios) != 3:
        raise ValueError(f"split_ratios must be a tuple of 3 floats, got: {split_ratios}")

    if not abs(sum(split_ratios) - 1.0) < 0.001:
        raise ValueError(f"split_ratios must sum to 1.0, got sum: {sum(split_ratios)}")

    valid_backbones = ['VGG16', 'ResNet50', 'YOLO']
    if backbone_name not in valid_backbones:
        raise ValueError(f"backbone_name must be one of {valid_backbones}, got: {backbone_name}")

    valid_balance_modes = ['oversample', 'downsample', 'none']
    if balanced not in valid_balance_modes:
        raise ValueError(f"balanced must be one of {valid_balance_modes}, got: {balanced}")

    # --- 1. INICIALIZACION DE VARIABLES ---
    env_vars = EnvLoader().get_all()

    # --- 2. CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
    MLRUNS_PATH = os.path.join(os.path.dirname(os.getcwd()), 'models', 'mlruns')
    print('mlruns directory =', MLRUNS_PATH)
    os.makedirs(MLRUNS_PATH, exist_ok=True)

    # Tracking URI (usar formato file:/// con / en lugar de \)
    mlruns_uri = f"file:///{os.path.abspath(MLRUNS_PATH).replace(os.sep, '/')}"
    mlflow.set_tracking_uri(mlruns_uri)

    print("📂 Tracking URI actual:", mlflow.get_tracking_uri())

    # Asegurar que el experimento existe
    experiment_name = "image_classification_experiment"
    mlflow.set_experiment(experiment_name)

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    
    try:
        image_size_str = env_vars.get("IMAGE_SIZE", "(224, 224)")

        if not image_size_str or len(image_size_str.strip()) == 0:
            raise ValueError("IMAGE_SIZE is empty in .env file.")

        # ast.literal_eval evalúa la cadena de forma segura
        IMAGE_SIZE = ast.literal_eval(image_size_str)
        
        # Añadir una verificación de seguridad para asegurar que la tupla tiene 2 elementos
        if not isinstance(IMAGE_SIZE, (tuple, list)) or len(IMAGE_SIZE) != 2:
            raise TypeError("IMAGE_SIZE must be a sequence of length 2.")

    except (ValueError, SyntaxError, TypeError) as e:
        print(f"❌ Error: La variable de entorno IMAGE_SIZE no es válida. Usando valor por defecto. Error: {e}")
        IMAGE_SIZE = (224, 224)


    NUM_CLASSES = int(env_vars['NUM_CLASSES']) #4
    BATCH_SIZE = int(env_vars['BATCH_SIZE']) #32
    
    # Parámetros para la búsqueda de Keras Tuner
    MAX_TRIALS = int(env_vars['MAX_TRIALS']) #10  # Número total de modelos a probar
    TUNER_EPOCHS = int(env_vars['TUNER_EPOCHS']) #10 # Número de épocas para cada modelo durante la búsqueda
    FACTOR = int(env_vars['FACTOR'])  #3     # Factor de reducción para el algoritmo Hyperband.
    MAX_EPOCHS = int(env_vars['MAX_EPOCHS']) #20 # Número máximo de épocas para cualquier modelo.

    print("Iniciando el proceso de entrenamiento y búsqueda de hiperparámetros con Hyperband...")
    print(f"Dataset de origen: {DATA_DIR.name}")
    print(f"Número de clases: {NUM_CLASSES}")

    # --- 3. CARGAR Y PREPARAR LOS DATOS ---
    print("\n📦 Cargando y preparando los datos en memoria...")
    
    raw_dataset = split_and_balance_dataset(
        balanced=balanced,
        split_ratios=split_ratios
    )

    # Use shared flatten_data function from utils
    X_train, y_train = flatten_data(raw_dataset['train'], image_size=IMAGE_SIZE)
    X_val, y_val = flatten_data(raw_dataset['val'], image_size=IMAGE_SIZE)
    X_test, y_test = flatten_data(raw_dataset['test'], image_size=IMAGE_SIZE)

    # Después de la función flatten_data()
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        print("Error: Los datos de entrenamiento contienen valores no válidos.")
        exit()

    label_to_int = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_to_int[l] for l in y_train])
    y_val = np.array([label_to_int[l] for l in y_val])
    y_test = np.array([label_to_int[l] for l in y_test])
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    print("✅ Datos convertidos a tensores de NumPy.")
    
    # --- 4. INICIALIZAR EL MODEL BUILDER E INSTANCIAR EL TUNER ---
    print("\n🛠️  Inicializando el constructor de modelos para el Tuner...")
    
    hypermodel = ModelBuilder(
        backbone_name=backbone_name,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=NUM_CLASSES
    )

    tuner_dir = PROJECT_ROOT / 'models' / 'tuner_checkpoints'
    tuner_dir.mkdir(parents=True, exist_ok=True)

    print('KERAS TUNER DIR =', tuner_dir)
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=MAX_EPOCHS,
        factor=FACTOR,
        directory=tuner_dir,
        project_name='image_classification'
    )
    
    tuner.search_space_summary()
    
    # --- 5. CONFIGURAR CALLBACKS ---
    print("\n⚙️  Configurando Callbacks para la búsqueda...")
    
    checkpoint_cb = ModelCheckpoint(
        filepath=tuner_dir / 'best_trial_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    early_stopping_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    callbacks = [
        checkpoint_cb, 
        early_stopping_cb
    ]

    # --- 6. EJECUTAR LA BÚSQUEDA DE HIPERPARÁMETROS CON MLflow ---
    print("\n" + "="*70)
    print("🚀 ¡Comenzando la búsqueda de hiperparámetros con MLflow!")
    print("="*70)

    # Iniciar un run de MLflow que encapsula toda la búsqueda
    with mlflow.start_run(run_name=f"{backbone_name}_tuner_search"):
        tuner.search(
            x=X_train,
            y=y_train,
            epochs=TUNER_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            batch_size=BATCH_SIZE
        )

        # Al terminar, log de cada trial manualmente
        print("\n" + "="*70)
        print("📊 Registrando métricas con MLflow:")
        for trial in tuner.oracle.trials.values():
            with mlflow.start_run(nested=True, run_name=f"trial-{trial.trial_id}"):
                for hp_name, hp_value in trial.hyperparameters.values.items():
                    mlflow.log_param(hp_name, str(hp_value))

                if trial.metrics.metrics:
                    
                    for metric_name, metric_obj in trial.metrics.metrics.items():
                    # metric_obj.history es una lista de floats (uno por epoch)
                        history = metric_obj.get_history() if hasattr(metric_obj, "get_history") else metric_obj.history
                        if history:
                        # loggea todos los valores por epoch
                            for obs in history:
                                # obs puede ser un MetricObservation o un número
                                if hasattr(obs, "value"):
                                    val = obs.value
                                    if isinstance(val, (list, tuple)):
                                        # recorrer cada valor dentro de la lista
                                        for i, v in enumerate(val):
                                            mlflow.log_metric(metric_name, float(v), step=(getattr(obs, "step", 0) or 0) + i)
                                    else:
                                        mlflow.log_metric(
                                            metric_name,
                                            float(val),
                                            step=getattr(obs, "step", None) or 0
                                        )
                                else:
                                    # obs es ya un float o int
                                    mlflow.log_metric(metric_name, float(obs), step=history.index(obs))
        print("="*70)
    # --- 7. OBTENER Y GUARDAR EL MEJOR MODELO ---
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    if best_hps:
        print("\n✅ Best Hyperparameters found. Report in progress.")
    else:
        print("⚠️ No se encontraron hiperparámetros óptimos, usando los iniciales por defecto")
        best_hps = tuner.oracle.get_space().get_hyperparameters()  

    with mlflow.start_run(run_name=f"{backbone_name}_best_model", nested=True):
        print("\n📊 Evaluando el mejor modelo en el conjunto de prueba...")
        test_loss, test_acc = best_model.evaluate(x=X_test, y=y_test)
        print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")
        mlflow.log_params(best_hps.values)
        mlflow.keras.log_model(best_model, "final_corn_model")
        mlflow.log_metric("test_accuracy", test_acc)

    print(f"\n🏆 El mejor modelo se encontró con los siguientes hiperparámetros:")
    for hp_name, value in best_hps.values.items():
        print(f"   - {hp_name}: {value}")

    print("\n" + "="*70)
    print("✅ ¡Búsqueda de hiperparámetros completada exitosamente!")
    print("="*70)

    exported_model_dir = PROJECT_ROOT / 'models' / 'exported'
    exported_model_dir.mkdir(parents=True, exist_ok=True)

    # Save with timestamp for versioning
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save versioned model
    versioned_model_path = exported_model_dir / f'{backbone_name}_{timestamp}_acc{test_acc:.4f}.keras'
    best_model.save(versioned_model_path)
    print(f"\n💾 Modelo versionado guardado en: {versioned_model_path}")

    # Also save as "best" for easy loading
    best_model_path = exported_model_dir / f'best_{backbone_name}.keras'
    best_model.save(best_model_path)
    print(f"💾 Modelo 'best' actualizado en: {best_model_path}")

    # Save hyperparameters as JSON
    import json
    metadata = {
        'backbone': backbone_name,
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'hyperparameters': best_hps.values,
        'split_ratios': split_ratios,
        'balanced': balanced,
        'image_size': IMAGE_SIZE,
        'num_classes': NUM_CLASSES
    }
    metadata_path = exported_model_dir / f'{backbone_name}_{timestamp}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadatos guardados en: {metadata_path}")

    # --- 8. EVALUAR EL MEJOR MODELO EN EL CONJUNTO DE PRUEBA ---
    print("\n" + "="*70)
    print("📊 Evaluando el mejor modelo en el conjunto de prueba...")
    print("="*70)
    
    test_loss, test_acc = best_model.evaluate(x=X_test, y=y_test)
    print(f"\n✅ Precisión en el conjunto de prueba: {test_acc:.4f}")
    
    return tuner, (X_test, y_test)