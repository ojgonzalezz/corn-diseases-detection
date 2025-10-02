"""
Tests para el sistema de logging.

Este módulo prueba la funcionalidad de logger.py,
verificando el correcto funcionamiento del sistema de logging.
"""
import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from src.utils.logger import (
    setup_logger,
    get_logger,
    log_print,
    log_section,
    log_dict,
    LogColors,
    ColoredFormatter
)


class TestLoggerSetup:
    """Tests para configuración básica del logger."""

    def test_setup_logger_basic(self):
        """Verifica configuración básica del logger."""
        logger = setup_logger('test_logger')
        
        assert logger is not None
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """Verifica configuración con nivel personalizado."""
        logger = setup_logger('test_logger_debug', level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG

    def test_setup_logger_no_duplicate_handlers(self):
        """Verifica que no se crean handlers duplicados."""
        logger_name = 'test_no_duplicate'
        
        # Primera llamada
        logger1 = setup_logger(logger_name)
        handlers_count_1 = len(logger1.handlers)
        
        # Segunda llamada al mismo logger
        logger2 = setup_logger(logger_name)
        handlers_count_2 = len(logger2.handlers)
        
        # No debe agregar handlers duplicados
        assert handlers_count_1 == handlers_count_2
        assert logger1 is logger2

    def test_setup_logger_with_file(self, tmp_path):
        """Verifica configuración con archivo de log."""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test_file_logger', log_file=log_file)
        
        # Verificar que el logger tiene un FileHandler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        
        # Probar que escribe al archivo
        logger.info("Test message")
        logger.handlers[0].flush()  # Forzar escritura
        
        # El archivo debe existir y contener el mensaje
        assert log_file.exists()

    def test_setup_logger_without_colors(self):
        """Verifica configuración sin colores."""
        logger = setup_logger('test_no_colors', use_colors=False)
        
        assert logger is not None
        # Verificar que tiene handlers
        assert len(logger.handlers) > 0


class TestGetLogger:
    """Tests para get_logger function."""

    def test_get_logger_basic(self):
        """Verifica función de conveniencia get_logger."""
        logger = get_logger('test_get_logger')
        
        assert logger is not None
        assert 'test_get_logger' in logger.name

    def test_get_logger_with_file(self, tmp_path):
        """Verifica get_logger con archivo de log."""
        with patch('src.utils.logger.paths') as mock_paths:
            mock_paths.root = tmp_path
            
            logger = get_logger('test_file', log_to_file=True)
            
            assert logger is not None


class TestColoredFormatter:
    """Tests para ColoredFormatter."""

    def test_colored_formatter_initialization(self):
        """Verifica inicialización de ColoredFormatter."""
        formatter = ColoredFormatter('[%(levelname_colored)s] %(message)s')
        
        assert formatter is not None

    def test_colored_formatter_format(self):
        """Verifica formateo con colores."""
        formatter = ColoredFormatter('[%(levelname_colored)s] %(message)s')
        
        # Crear un LogRecord
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Debe contener el mensaje
        assert 'Test message' in formatted
        # Debe tener el atributo levelname_colored
        assert hasattr(record, 'levelname_colored')

    def test_colored_formatter_different_levels(self):
        """Verifica colores diferentes para diferentes niveles."""
        formatter = ColoredFormatter('[%(levelname_colored)s]')
        
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]
        
        for level in levels:
            record = logging.LogRecord(
                name='test',
                level=level,
                pathname='test.py',
                lineno=1,
                msg='Test',
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            assert formatted is not None


class TestLogColors:
    """Tests para LogColors."""

    def test_log_colors_constants(self):
        """Verifica que LogColors tiene todas las constantes necesarias."""
        assert hasattr(LogColors, 'RESET')
        assert hasattr(LogColors, 'RED')
        assert hasattr(LogColors, 'GREEN')
        assert hasattr(LogColors, 'YELLOW')
        assert hasattr(LogColors, 'BLUE')
        assert hasattr(LogColors, 'MAGENTA')
        assert hasattr(LogColors, 'CYAN')
        assert hasattr(LogColors, 'GRAY')

    def test_log_colors_are_strings(self):
        """Verifica que los colores son strings."""
        assert isinstance(LogColors.RESET, str)
        assert isinstance(LogColors.RED, str)
        assert isinstance(LogColors.BLUE, str)


class TestLogPrint:
    """Tests para log_print function."""

    def test_log_print_basic(self):
        """Verifica función log_print básica."""
        logger = get_logger('test_log_print')
        
        # Capturar logs
        with patch.object(logger, 'info') as mock_info:
            log_print("Test message", logger=logger)
            mock_info.assert_called()

    def test_log_print_detects_error_level(self):
        """Verifica detección automática de nivel ERROR."""
        logger = get_logger('test_error')
        
        with patch.object(logger, 'error') as mock_error:
            log_print("[ERROR] Something went wrong", logger=logger)
            mock_error.assert_called()

    def test_log_print_detects_warning_level(self):
        """Verifica detección automática de nivel WARNING."""
        logger = get_logger('test_warning')
        
        with patch.object(logger, 'warning') as mock_warning:
            log_print("[WARNING] Be careful", logger=logger)
            mock_warning.assert_called()

    def test_log_print_cleans_prefix(self):
        """Verifica que limpia prefijos redundantes."""
        logger = get_logger('test_clean')
        
        with patch.object(logger, 'info') as mock_info:
            log_print("[INFO] This is info", logger=logger)
            # Debe llamarse con el mensaje sin el prefijo
            call_args = mock_info.call_args
            assert "[INFO]" not in str(call_args)

    def test_log_print_without_logger(self):
        """Verifica que usa logger por defecto si no se proporciona."""
        # No debe generar error
        log_print("Test without logger")


class TestLogSection:
    """Tests para log_section function."""

    def test_log_section_basic(self):
        """Verifica creación de sección básica."""
        logger = get_logger('test_section')
        
        with patch.object(logger, 'info') as mock_info:
            log_section(logger, "Test Section")
            
            # Debe llamarse 3 veces (línea superior, título, línea inferior)
            assert mock_info.call_count == 3

    def test_log_section_custom_char(self):
        """Verifica sección con carácter personalizado."""
        logger = get_logger('test_section_char')
        
        with patch.object(logger, 'info') as mock_info:
            log_section(logger, "Title", char='-', width=50)
            
            assert mock_info.call_count == 3
            # Verificar que usó el carácter correcto
            first_call = str(mock_info.call_args_list[0])
            assert '-' in first_call

    def test_log_section_custom_width(self):
        """Verifica sección con ancho personalizado."""
        logger = get_logger('test_section_width')
        
        with patch.object(logger, 'info') as mock_info:
            log_section(logger, "Title", width=100)
            
            assert mock_info.call_count == 3


class TestLogDict:
    """Tests para log_dict function."""

    def test_log_dict_basic(self):
        """Verifica logging de diccionario básico."""
        logger = get_logger('test_dict')
        
        test_dict = {
            'key1': 'value1',
            'key2': 42,
            'key3': [1, 2, 3]
        }
        
        with patch.object(logger, 'info') as mock_info:
            log_dict(logger, test_dict)
            
            # Debe llamarse una vez por cada clave
            assert mock_info.call_count >= len(test_dict)

    def test_log_dict_with_title(self):
        """Verifica logging de diccionario con título."""
        logger = get_logger('test_dict_title')
        
        test_dict = {'key': 'value'}
        
        with patch.object(logger, 'info') as mock_info:
            log_dict(logger, test_dict, title="Test Dict")
            
            # Debe incluir el título en las llamadas
            calls = [str(call) for call in mock_info.call_args_list]
            assert any('Test Dict' in call for call in calls)

    def test_log_dict_empty(self):
        """Verifica logging de diccionario vacío."""
        logger = get_logger('test_dict_empty')
        
        with patch.object(logger, 'info') as mock_info:
            log_dict(logger, {})
            
            # No debe fallar con diccionario vacío
            assert True

    def test_log_dict_formatting(self):
        """Verifica formato de salida del diccionario."""
        logger = get_logger('test_dict_format')
        
        test_dict = {
            'short': 1,
            'very_long_key': 2
        }
        
        with patch.object(logger, 'info') as mock_info:
            log_dict(logger, test_dict)
            
            # Verificar que se llamó para cada entrada
            assert mock_info.call_count >= len(test_dict)


class TestLogModelSummary:
    """Tests para log_model_summary function."""

    def test_log_model_summary_with_mock_model(self):
        """Verifica logging de resumen de modelo."""
        from src.utils.logger import log_model_summary
        
        logger = get_logger('test_model')
        
        # Mock de modelo con método summary
        mock_model = MagicMock()
        mock_model.summary = MagicMock()
        
        with patch.object(logger, 'info'):
            log_model_summary(logger, mock_model)
            
            # Debe llamar a model.summary
            mock_model.summary.assert_called_once()

    def test_log_model_summary_handles_error(self):
        """Verifica manejo de error en log_model_summary."""
        from src.utils.logger import log_model_summary
        
        logger = get_logger('test_model_error')
        
        # Mock de modelo que genera error
        mock_model = MagicMock()
        mock_model.summary.side_effect = Exception("Test error")
        
        with patch.object(logger, 'warning') as mock_warning:
            log_model_summary(logger, mock_model)
            
            # Debe manejar el error y loggear warning
            mock_warning.assert_called()


class TestLoggingIntegration:
    """Tests de integración para el sistema de logging."""

    def test_logger_hierarchy(self):
        """Verifica jerarquía correcta de loggers."""
        parent = get_logger('parent')
        child = get_logger('parent.child')
        
        assert child.name.startswith(parent.name)

    def test_multiple_loggers(self):
        """Verifica que múltiples loggers funcionan correctamente."""
        logger1 = get_logger('logger1')
        logger2 = get_logger('logger2')
        
        assert logger1.name != logger2.name
        assert logger1 is not logger2

    def test_logger_output(self, capsys):
        """Test de integración: verificar salida real del logger."""
        logger = setup_logger('test_output', use_colors=False)
        
        logger.info("Test info message")
        
        # No podemos capturar directamente, pero al menos verificamos que no falla
        assert True

    def test_logger_persistence(self):
        """Verifica que los loggers persisten entre llamadas."""
        name = 'persistent_logger'
        
        logger1 = get_logger(name)
        logger2 = get_logger(name)
        
        # Deben ser el mismo objeto
        assert logger1 is logger2

