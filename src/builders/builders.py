#####################################################################################
# ---------------------------------- Model Builders ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
from tensorflow.keras import layers, models, Model
import keras_tuner as kt
from tensorflow.keras.initializers import HeNormal, GlorotUniform, LecunNormal
from src.builders.base_models import (
    load_vgg16, 
    load_resnet50, 
    load_yolo,
    load_mobilenetv3_large,
    load_efficientnet_lite_b2,
    load_mobilevit,
    load_pmvt
)

#########################
# ---- Model Builder ----
#########################

class ModelBuilder(kt.HyperModel):

    def __init__(self, 
                 backbone_name, 
                 input_shape=(224, 224, 3),
                 num_classes=4,
                 n_layers=(2, 10),
                 units=(16, 512, 32),
                 activation=['relu', 'tanh'],
                 learning_rates=[0.001, 0.0001, 0.00001],
                 metrics=['accuracy'],
                 dropout_range=(0.1, 0.5),
                 initializer=["he_normal", "glorot_uniform"]):
        
        # Parámetros de la clase
        self.backbone_name = backbone_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.units = units
        self.activation = activation
        self.learning_rates = learning_rates
        self.metrics = metrics
        self.dropout_range = dropout_range
        self.initializer = initializer

    def build(self, hp):
        """
        Construye un modelo de clasificación de imágenes utilizando Transfer Learning.
        """
        # --- 1. Cargar el Backbone ---
        backbone_dict = {
            'VGG16': load_vgg16, 
            'ResNet50': load_resnet50,
            'YOLO': load_yolo,
            # Edge Computing Models (4 arquitecturas seleccionadas)
            'MobileNetV3Large': load_mobilenetv3_large,
            'EfficientNetLiteB2': load_efficientnet_lite_b2,
            'MobileViT': load_mobilevit,
            'PMVT': load_pmvt
        }


        # Obtener el modelo base de forma dinámica
        backbone_loader = backbone_dict.get(self.backbone_name)
        if backbone_loader is None:
            raise ValueError(f"Backbone '{self.backbone_name}' no soportado. Elige entre: {list(backbone_dict.keys())}.")
        
        base_model = backbone_loader(input_shape=self.input_shape)

        # --- 2. Congelar el Backbone ---
        base_model.trainable = False
        print(f"[OK] Backbone '{self.backbone_name}' cargado y congelado.")
        
        # --- 3. Construir la Cabeza de Clasificación ---
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.Flatten())

        # Adicionar capas densas (hiperparámetros)
        num_layers = hp.Int('num_layers', min_value=self.n_layers[0], max_value=self.n_layers[1])
        for i in range(num_layers):
            units = hp.Int(f'units_{i}', min_value=self.units[0], max_value=self.units[1], step=self.units[2])
            activation = hp.Choice(f'activation_{i}', values=self.activation)
            initializer = hp.Choice(f'initializer_{i}', values=self.initializer)
            
            # Agregar la capa densa
            model.add(layers.Dense(units, activation=activation, kernel_initializer=initializer))
            
            # Agregar capa de Dropout
            dropout_rate = hp.Float(f'dropout_{i}', min_value=self.dropout_range[0], max_value=self.dropout_range[1])
            model.add(layers.Dropout(dropout_rate))

        # --- 4. Capa de Salida ---
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # --- 5. Compilar el Modelo (hiperparámetro) ---
        learning_rate = hp.Choice('learning_rate', values=self.learning_rates)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        
        print("[OK] Modelo final construido y compilado.")
        return model
    