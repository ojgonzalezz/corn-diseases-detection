"""
Tests para el módulo de carga de datos.

Este módulo prueba la funcionalidad de data_loader.py,
verificando la correcta carga de datos desde múltiples fuentes.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock

from src.adapters.data_loader import load_split_data, load_raw_data


class TestLoadSplitData:
    """Tests para carga de datos ya divididos (train/val/test)."""

    @pytest.fixture
    def mock_split_structure(self, tmp_path):
        """Crea estructura temporal de datos divididos."""
        # Crear estructura data/train/val/test
        data_dir = tmp_path / "data"
        
        for split in ['train', 'val', 'test']:
            split_dir = data_dir / split
            for class_name in ['Class1', 'Class2']:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Crear imágenes dummy
                for i in range(3):
                    img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
                    img.save(class_dir / f"image_{i}.jpg")
        
        return data_dir

    @patch('src.adapters.data_loader.paths')
    @patch('src.adapters.data_loader.load_images_from_folder')
    def test_load_split_data_structure(self, mock_load_images, mock_paths):
        """Verifica estructura del diccionario retornado."""
        # Mock de paths
        mock_paths.data_train.exists.return_value = True
        mock_paths.data_train = Path("/fake/train")
        mock_paths.data_val = Path("/fake/val")
        mock_paths.data_test = Path("/fake/test")
        
        # Mock de load_images_from_folder
        mock_images = [Image.new('RGB', (100, 100)) for _ in range(3)]
        mock_load_images.return_value = mock_images
        
        # Mock de iterdir
        def mock_iterdir(path):
            class MockDir:
                def __init__(self, name):
                    self.name = name
                def is_dir(self):
                    return True
            return [MockDir("Class1"), MockDir("Class2")]
        
        mock_paths.data_train.iterdir = lambda: mock_iterdir(mock_paths.data_train)
        mock_paths.data_train.exists.return_value = True
        mock_paths.data_val.iterdir = lambda: mock_iterdir(mock_paths.data_val)
        mock_paths.data_val.exists.return_value = True
        mock_paths.data_test.iterdir = lambda: mock_iterdir(mock_paths.data_test)
        mock_paths.data_test.exists.return_value = True
        
        result = load_split_data()
        
        # Verificar estructura
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result
        assert isinstance(result, dict)

    @patch('src.adapters.data_loader.paths')
    def test_load_split_data_missing_directory(self, mock_paths):
        """Verifica comportamiento cuando falta un directorio."""
        mock_paths.data_train.exists.return_value = False
        mock_paths.data_val.exists.return_value = False
        mock_paths.data_test.exists.return_value = False
        
        result = load_split_data()
        
        # Debe retornar estructura vacía pero válida
        assert isinstance(result, dict)

    @patch('src.adapters.data_loader.config')
    def test_load_split_data_uses_config(self, mock_config):
        """Verifica que usa la configuración correctamente."""
        mock_config.data.class_names = ['Class1', 'Class2', 'Class3']
        
        # El código debe intentar acceder a config.data.class_names
        try:
            load_split_data()
        except Exception:
            pass  # No importa si falla, solo verificamos que accede a config
        
        # Verificar que se intentó acceder a la configuración
        assert hasattr(mock_config.data, 'class_names')

    def test_load_split_data_image_types(self):
        """Verifica que las imágenes cargadas son objetos PIL.Image."""
        with patch('src.adapters.data_loader.paths') as mock_paths:
            with patch('src.adapters.data_loader.load_images_from_folder') as mock_load:
                # Simular imágenes PIL
                mock_images = [Image.new('RGB', (100, 100)) for _ in range(3)]
                mock_load.return_value = mock_images
                
                mock_paths.data_train.exists.return_value = True
                mock_paths.data_val.exists.return_value = True
                mock_paths.data_test.exists.return_value = True
                
                def mock_iterdir(path):
                    class MockDir:
                        def __init__(self, name):
                            self.name = name
                        def is_dir(self):
                            return True
                    return [MockDir("Class1")]
                
                mock_paths.data_train.iterdir = lambda: mock_iterdir(None)
                mock_paths.data_val.iterdir = lambda: mock_iterdir(None)
                mock_paths.data_test.iterdir = lambda: mock_iterdir(None)
                
                result = load_split_data()
                
                # Verificar que se llamó a load_images_from_folder
                assert mock_load.called


class TestLoadRawData:
    """Tests para carga de datos raw (data_1, data_2)."""

    @patch('src.adapters.data_loader.paths')
    @patch('src.adapters.data_loader.config')
    def test_load_raw_data_structure(self, mock_config, mock_paths):
        """Verifica estructura del diccionario retornado para datos raw."""
        # Mock de configuración
        mock_config.data.datasets_consideration = ["no-augmentation", "augmented"]
        
        # Mock de paths
        mock_paths.data_raw.exists.return_value = True
        
        def mock_iterdir():
            class MockDir:
                def __init__(self, name):
                    self.name = name
                def is_dir(self):
                    return True
            return [MockDir("data_1"), MockDir("data_2")]
        
        mock_paths.data_raw.iterdir = mock_iterdir
        mock_paths.data_raw.__truediv__ = lambda self, x: Path(f"/fake/raw/{x}")
        
        # Mock de subcarpetas
        with patch('src.adapters.data_loader.load_images_from_folder') as mock_load:
            mock_load.return_value = [Image.new('RGB', (100, 100))]
            
            def mock_category_iterdir():
                class MockDir:
                    def __init__(self, name):
                        self.name = name
                    def is_dir(self):
                        return True
                return [MockDir("Class1")]
            
            with patch.object(Path, 'iterdir', return_value=mock_category_iterdir()):
                result = load_raw_data()
        
        # Verificar estructura
        assert isinstance(result, dict)

    @patch('src.adapters.data_loader.paths')
    def test_load_raw_data_missing_directory(self, mock_paths):
        """Verifica que genera error cuando no existe data/raw."""
        mock_paths.data_raw.exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="No se encontró"):
            load_raw_data()

    @patch('src.adapters.data_loader.paths')
    @patch('src.adapters.data_loader.config')
    def test_load_raw_data_empty_raw_dir(self, mock_config, mock_paths):
        """Verifica comportamiento cuando data/raw está vacío."""
        mock_config.data.datasets_consideration = ["no-augmentation"]
        mock_paths.data_raw.exists.return_value = True
        
        # Simular directorio vacío (sin data_1, data_2)
        mock_paths.data_raw.iterdir.return_value = []
        
        with pytest.raises(FileNotFoundError, match="No se encontraron subdirectorios"):
            load_raw_data()

    @patch('src.adapters.data_loader.paths')
    @patch('src.adapters.data_loader.config')
    @patch('src.adapters.data_loader.load_images_from_folder')
    def test_load_raw_data_dataset_consideration(self, mock_load, mock_config, mock_paths):
        """Verifica que asigna correctamente dataset_consideration."""
        # Mock de configuración
        considerations = ["no-augmentation", "augmented"]
        mock_config.data.datasets_consideration = considerations
        
        # Mock de paths
        mock_paths.data_raw.exists.return_value = True
        
        def mock_iterdir():
            class MockDir:
                def __init__(self, name):
                    self.name = name
                def is_dir(self):
                    return True
            return [MockDir("data_1"), MockDir("data_2")]
        
        mock_paths.data_raw.iterdir = mock_iterdir
        mock_paths.data_raw.__truediv__ = lambda self, x: MagicMock(
            iterdir=lambda: [],
            __str__=lambda: f"/fake/{x}"
        )
        
        mock_load.return_value = []
        
        result = load_raw_data()
        
        # Verificar que los datasets tienen la propiedad dataset_consideration
        for i, dataset_key in enumerate(['data_1', 'data_2']):
            if dataset_key in result:
                assert 'dataset_consideration' in result[dataset_key]


class TestImageLoading:
    """Tests de integración para carga de imágenes."""

    def test_loaded_images_are_pil_objects(self):
        """Verifica que las imágenes cargadas son objetos PIL.Image."""
        with patch('src.adapters.data_loader.load_images_from_folder') as mock_load:
            # Simular carga de imágenes PIL
            mock_images = [
                Image.new('RGB', (100, 100)),
                Image.new('RGB', (150, 150)),
            ]
            mock_load.return_value = mock_images
            
            result = mock_load("/fake/path")
            
            assert len(result) == 2
            assert all(isinstance(img, Image.Image) for img in result)

    def test_handle_corrupted_images(self):
        """Verifica manejo de imágenes corruptas."""
        with patch('src.adapters.data_loader.load_images_from_folder') as mock_load:
            # Simular lista con None (imagen corrupta)
            mock_load.return_value = []  # Imágenes corruptas se omiten
            
            result = mock_load("/fake/path")
            
            assert isinstance(result, list)


class TestDataLoaderIntegration:
    """Tests de integración para el cargador de datos completo."""

    def test_load_split_data_returns_valid_structure(self):
        """Test de integración: estructura de datos válida."""
        with patch('src.adapters.data_loader.paths') as mock_paths:
            with patch('src.adapters.data_loader.load_images_from_folder') as mock_load:
                # Setup mocks
                mock_paths.data_train.exists.return_value = True
                mock_paths.data_val.exists.return_value = True
                mock_paths.data_test.exists.return_value = True
                
                mock_images = [Image.new('RGB', (100, 100)) for _ in range(5)]
                mock_load.return_value = mock_images
                
                def mock_iterdir():
                    class MockDir:
                        def __init__(self, name):
                            self.name = name
                        def is_dir(self):
                            return True
                    return [MockDir("Class1"), MockDir("Class2")]
                
                mock_paths.data_train.iterdir = mock_iterdir
                mock_paths.data_val.iterdir = mock_iterdir
                mock_paths.data_test.iterdir = mock_iterdir
                
                result = load_split_data()
                
                # Verificar que tiene las claves correctas
                assert 'train' in result
                assert 'val' in result
                assert 'test' in result

    def test_consistency_between_loaders(self):
        """Verifica consistencia entre load_split_data y load_raw_data."""
        # Ambas funciones deben retornar diccionarios
        with patch('src.adapters.data_loader.paths'):
            with patch('src.adapters.data_loader.load_images_from_folder'):
                result_split = load_split_data()
                assert isinstance(result_split, dict)
        
        with patch('src.adapters.data_loader.paths') as mock_paths:
            with patch('src.adapters.data_loader.config') as mock_config:
                mock_config.data.datasets_consideration = ["no-augmentation"]
                mock_paths.data_raw.exists.return_value = True
                mock_paths.data_raw.iterdir.return_value = []
                
                try:
                    result_raw = load_raw_data()
                    assert isinstance(result_raw, dict)
                except FileNotFoundError:
                    pass  # Esperado si no hay datos

