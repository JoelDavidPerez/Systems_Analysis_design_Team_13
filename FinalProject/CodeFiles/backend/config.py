import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración base"""
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', True)
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    TORCH_DEVICE = os.getenv('TORCH_DEVICE', 'cpu')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'models'

class DevelopmentConfig(Config):
    """Configuración de desarrollo"""
    DEBUG = True

class ProductionConfig(Config):
    """Configuración de producción"""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}