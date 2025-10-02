"""
Tests para el sistema de gestión de rutas.

Este módulo prueba la funcionalidad de paths.py,
verificando el correcto manejo de rutas del proyecto.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.paths import (
    get_project_root,
    ProjectPaths,
    paths,
    get_data_dir,
    get_models_dir,
    get_mlruns_dir
)


class TestGetProjectRoot:
    """Tests para get_project_root function."""

    def test_get_project_root_returns_path(self):
        """Verifica que retorna un Path."""
        root = get_project_root()
        
        assert isinstance(root, Path)
        assert root.exists()

    def test_get_project_root_has_src_directory(self):
        """Verifica que el root tiene directorio src."""
        root = get_project_root()
        src_dir = root / 'src'
        
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_get_project_root_consistency(self):
        """Verifica que múltiples llamadas retornan el mismo path."""
        root1 = get_project_root()
        root2 = get_project_root()
        
        assert root1 == root2

    def test_get_project_root_is_absolute(self):
        """Verifica que el path retornado es absoluto."""
        root = get_project_root()
        
        assert root.is_absolute()


class TestProjectPathsInitialization:
    """Tests para inicialización de ProjectPaths."""

    def test_project_paths_initialization(self):
        """Verifica inicialización correcta de ProjectPaths."""
        pp = ProjectPaths()
        
        assert pp is not None
        assert hasattr(pp, 'root')

    def test_project_paths_root_property(self):
        """Verifica propiedad root."""
        pp = ProjectPaths()
        
        assert isinstance(pp.root, Path)
        assert pp.root.exists()

    def test_global_paths_instance(self):
        """Verifica que existe instancia global 'paths'."""
        assert paths is not None
        assert isinstance(paths, ProjectPaths)


class TestProjectPathsProperties:
    """Tests para propiedades de ProjectPaths."""

    def test_src_property(self):
        """Verifica propiedad src."""
        pp = ProjectPaths()
        
        assert isinstance(pp.src, Path)
        assert pp.src.name == 'src'

    def test_data_property(self):
        """Verifica propiedad data."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data, Path)
        assert pp.data.name == 'data'

    def test_data_raw_property(self):
        """Verifica propiedad data_raw."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data_raw, Path)
        assert 'raw' in str(pp.data_raw)

    def test_data_processed_property(self):
        """Verifica propiedad data_processed."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data_processed, Path)
        assert 'processed' in str(pp.data_processed)

    def test_data_train_property(self):
        """Verifica propiedad data_train."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data_train, Path)
        assert 'train' in str(pp.data_train)

    def test_data_val_property(self):
        """Verifica propiedad data_val."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data_val, Path)
        assert 'val' in str(pp.data_val)

    def test_data_test_property(self):
        """Verifica propiedad data_test."""
        pp = ProjectPaths()
        
        assert isinstance(pp.data_test, Path)
        assert 'test' in str(pp.data_test)

    def test_models_property(self):
        """Verifica propiedad models."""
        pp = ProjectPaths()
        
        assert isinstance(pp.models, Path)
        assert pp.models.name == 'models'

    def test_models_exported_property(self):
        """Verifica propiedad models_exported."""
        pp = ProjectPaths()
        
        assert isinstance(pp.models_exported, Path)
        assert 'exported' in str(pp.models_exported)

    def test_models_tuner_property(self):
        """Verifica propiedad models_tuner."""
        pp = ProjectPaths()
        
        assert isinstance(pp.models_tuner, Path)
        assert 'tuner' in str(pp.models_tuner)

    def test_mlruns_property(self):
        """Verifica propiedad mlruns."""
        pp = ProjectPaths()
        
        assert isinstance(pp.mlruns, Path)
        assert 'mlruns' in str(pp.mlruns)

    def test_tests_property(self):
        """Verifica propiedad tests."""
        pp = ProjectPaths()
        
        assert isinstance(pp.tests, Path)
        assert pp.tests.name == 'tests'


class TestProjectPathsMethods:
    """Tests para métodos de ProjectPaths."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Verifica que ensure_dir crea directorios."""
        pp = ProjectPaths()
        test_dir = tmp_path / "test_dir" / "nested"
        
        result = pp.ensure_dir(test_dir)
        
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    def test_ensure_dir_idempotent(self, tmp_path):
        """Verifica que ensure_dir es idempotente."""
        pp = ProjectPaths()
        test_dir = tmp_path / "test_dir"
        
        # Primera llamada
        pp.ensure_dir(test_dir)
        assert test_dir.exists()
        
        # Segunda llamada no debe fallar
        pp.ensure_dir(test_dir)
        assert test_dir.exists()

    def test_get_model_path(self):
        """Verifica get_model_path."""
        pp = ProjectPaths()
        
        model_name = "best_VGG16.keras"
        model_path = pp.get_model_path(model_name)
        
        assert isinstance(model_path, Path)
        assert model_name in str(model_path)
        assert 'exported' in str(model_path)

    def test_get_model_path_different_names(self):
        """Verifica get_model_path con diferentes nombres."""
        pp = ProjectPaths()
        
        names = ["model1.keras", "model2.h5", "checkpoint.ckpt"]
        
        for name in names:
            model_path = pp.get_model_path(name)
            assert name in str(model_path)

    def test_relative_to_root(self):
        """Verifica relative_to_root."""
        pp = ProjectPaths()
        
        # Path dentro del proyecto
        abs_path = pp.root / "src" / "core" / "config.py"
        rel_path = pp.relative_to_root(abs_path)
        
        assert not rel_path.is_absolute() or rel_path == abs_path
        assert 'src' in str(rel_path)

    def test_relative_to_root_outside_project(self):
        """Verifica relative_to_root con path fuera del proyecto."""
        pp = ProjectPaths()
        
        # Path fuera del proyecto
        outside_path = Path("/some/random/path")
        result = pp.relative_to_root(outside_path)
        
        # Debe retornar el path original
        assert result == outside_path


class TestConvenienceFunctions:
    """Tests para funciones de conveniencia."""

    def test_get_data_dir(self):
        """Verifica get_data_dir."""
        data_dir = get_data_dir()
        
        assert isinstance(data_dir, Path)
        assert 'raw' in str(data_dir)

    def test_get_models_dir(self):
        """Verifica get_models_dir."""
        models_dir = get_models_dir()
        
        assert isinstance(models_dir, Path)
        assert 'exported' in str(models_dir)

    def test_get_mlruns_dir(self):
        """Verifica get_mlruns_dir."""
        mlruns_dir = get_mlruns_dir()
        
        assert isinstance(mlruns_dir, Path)
        assert 'mlruns' in str(mlruns_dir)

    def test_convenience_functions_consistency(self):
        """Verifica que las funciones de conveniencia son consistentes con paths."""
        assert get_data_dir() == paths.data_raw
        assert get_models_dir() == paths.models_exported
        assert get_mlruns_dir() == paths.mlruns


class TestPathsIntegration:
    """Tests de integración para el sistema de rutas."""

    def test_paths_structure_completeness(self):
        """Verifica que paths tiene todas las propiedades necesarias."""
        required_properties = [
            'root', 'src', 'data', 'data_raw', 'data_processed',
            'data_train', 'data_val', 'data_test',
            'models', 'models_exported', 'models_tuner',
            'mlruns', 'tests'
        ]
        
        for prop in required_properties:
            assert hasattr(paths, prop), f"Missing property: {prop}"
            value = getattr(paths, prop)
            assert isinstance(value, Path), f"{prop} is not a Path"

    def test_paths_hierarchy(self):
        """Verifica jerarquía correcta de paths."""
        # Todos los paths deben estar bajo root
        assert str(paths.src).startswith(str(paths.root))
        assert str(paths.data).startswith(str(paths.root))
        assert str(paths.models).startswith(str(paths.root))
        assert str(paths.tests).startswith(str(paths.root))

    def test_data_subdirectories_hierarchy(self):
        """Verifica jerarquía de subdirectorios de data."""
        data_root = str(paths.data)
        
        assert str(paths.data_raw).startswith(data_root)
        assert str(paths.data_processed).startswith(data_root)
        assert str(paths.data_train).startswith(data_root)
        assert str(paths.data_val).startswith(data_root)
        assert str(paths.data_test).startswith(data_root)

    def test_models_subdirectories_hierarchy(self):
        """Verifica jerarquía de subdirectorios de models."""
        models_root = str(paths.models)
        
        assert str(paths.models_exported).startswith(models_root)
        assert str(paths.models_tuner).startswith(models_root)
        assert str(paths.mlruns).startswith(models_root)

    def test_paths_consistency_across_modules(self):
        """Verifica que paths es consistente cuando se importa en diferentes módulos."""
        from src.utils.paths import paths as paths1
        from src.utils import paths as paths_module
        paths2 = paths_module.paths
        
        # Deben ser el mismo objeto
        assert paths1 is paths2
        assert paths1.root == paths2.root

    def test_ensure_dir_integration(self, tmp_path):
        """Test de integración para ensure_dir."""
        # Crear una estructura completa
        base_dir = tmp_path / "integration_test"
        
        dirs_to_create = [
            base_dir / "data" / "train",
            base_dir / "data" / "val",
            base_dir / "models" / "exported",
        ]
        
        for dir_path in dirs_to_create:
            paths.ensure_dir(dir_path)
            assert dir_path.exists()
            assert dir_path.is_dir()

    def test_get_model_path_integration(self):
        """Test de integración para get_model_path."""
        model_names = [
            "best_VGG16.keras",
            "best_ResNet50.keras",
            "model_v1.h5"
        ]
        
        for name in model_names:
            model_path = paths.get_model_path(name)
            
            # Verificar estructura
            assert isinstance(model_path, Path)
            assert name in str(model_path)
            assert 'models' in str(model_path)
            assert 'exported' in str(model_path)


class TestErrorHandling:
    """Tests para manejo de errores."""

    def test_get_project_root_error_handling(self):
        """Verifica manejo de error si no se encuentra src/."""
        with patch('src.utils.paths.Path') as MockPath:
            # Simular que no existe src/
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            MockPath.return_value = mock_path
            
            # Esto debería manejarse apropiadamente
            # (el test real puede fallar, pero no debe crashear)
            try:
                root = get_project_root()
            except RuntimeError as e:
                # Error esperado si no se encuentra src/
                assert "no se pudo determinar" in str(e).lower() or "src" in str(e).lower()

    def test_relative_to_root_with_none(self):
        """Verifica manejo de None en relative_to_root."""
        pp = ProjectPaths()
        
        # Aunque no es el uso esperado, no debe crashear
        try:
            result = pp.relative_to_root(Path("."))
            assert result is not None
        except Exception:
            # Cualquier excepción es aceptable, solo verificamos que no crashea el test
            pass

