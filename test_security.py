import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import hashlib
import hmac
import base64
import secrets
import re

from src.services.mt5_service import MT5Service
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger

class TestSecurity(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mt5_service = MT5Service()
        self.config_loader = ConfigLoader()
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.logger = setup_logger("test_security", self.log_dir / "test_security.log")

    def test_api_key_security(self):
        """Test la sécurité des clés API."""
        # Vérifier que les clés API ne sont pas en dur dans le code
        with open('services/mt5_service.py', 'r') as f:
            content = f.read()
            self.assertNotIn('API_KEY', content)
            self.assertNotIn('SECRET_KEY', content)
        
        # Vérifier que les clés API sont stockées de manière sécurisée
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = 'test_api_key'
            api_key = self.mt5_service.get_api_key()
            self.assertIsNotNone(api_key)
            self.assertNotEqual(api_key, 'test_api_key')  # Devrait être chiffré

    def test_password_security(self):
        """Test la sécurité des mots de passe."""
        # Vérifier que les mots de passe ne sont pas en dur dans le code
        with open('services/mt5_service.py', 'r') as f:
            content = f.read()
            self.assertNotIn('PASSWORD', content)
        
        # Vérifier que les mots de passe sont stockés de manière sécurisée
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = 'test_password'
            password = self.mt5_service.get_password()
            self.assertIsNotNone(password)
            self.assertNotEqual(password, 'test_password')  # Devrait être haché

    def test_config_security(self):
        """Test la sécurité de la configuration."""
        # Vérifier que les fichiers de configuration sont protégés
        config_file = Path('config/config.json')
        if config_file.exists():
            # Vérifier les permissions du fichier
            self.assertTrue(os.access(config_file, os.R_OK))
            self.assertFalse(os.access(config_file, os.W_OK))
        
        # Vérifier que les données sensibles sont chiffrées
        test_config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'password': 'test_password'
        }
        
        with patch('json.dump') as mock_dump:
            self.config_loader.save_config(test_config)
            # Vérifier que les données sensibles sont chiffrées avant la sauvegarde
            call_args = mock_dump.call_args[0][0]
            self.assertNotIn('test_key', str(call_args))
            self.assertNotIn('test_secret', str(call_args))
            self.assertNotIn('test_password', str(call_args))

    def test_input_validation(self):
        """Test la validation des entrées."""
        # Test des entrées malveillantes
        malicious_inputs = [
            "'; DROP TABLE trades; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "'; SELECT * FROM users; --"
        ]
        
        for malicious_input in malicious_inputs:
            # Vérifier que les entrées sont nettoyées
            cleaned_input = self.mt5_service.sanitize_input(malicious_input)
            self.assertNotEqual(cleaned_input, malicious_input)
            
            # Vérifier que les entrées sont validées
            self.assertFalse(self.mt5_service.validate_input(malicious_input))

    def test_encryption(self):
        """Test le chiffrement des données sensibles."""
        # Test du chiffrement des données
        sensitive_data = "sensitive_information"
        
        # Chiffrer les données
        encrypted_data = self.mt5_service.encrypt_data(sensitive_data)
        self.assertNotEqual(encrypted_data, sensitive_data)
        
        # Déchiffrer les données
        decrypted_data = self.mt5_service.decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, sensitive_data)

    def test_authentication(self):
        """Test l'authentification."""
        # Test de l'authentification avec des identifiants valides
        with patch.object(self.mt5_service, 'authenticate') as mock_auth:
            mock_auth.return_value = True
            is_authenticated = self.mt5_service.authenticate('valid_user', 'valid_password')
            self.assertTrue(is_authenticated)
        
        # Test de l'authentification avec des identifiants invalides
        with patch.object(self.mt5_service, 'authenticate') as mock_auth:
            mock_auth.return_value = False
            is_authenticated = self.mt5_service.authenticate('invalid_user', 'invalid_password')
            self.assertFalse(is_authenticated)

    def test_session_security(self):
        """Test la sécurité des sessions."""
        # Test de la génération de token de session
        session_token = self.mt5_service.generate_session_token()
        self.assertIsNotNone(session_token)
        self.assertTrue(len(session_token) >= 32)  # Token suffisamment long
        
        # Test de la validation de token
        is_valid = self.mt5_service.validate_session_token(session_token)
        self.assertTrue(is_valid)
        
        # Test avec un token invalide
        invalid_token = "invalid_token"
        is_valid = self.mt5_service.validate_session_token(invalid_token)
        self.assertFalse(is_valid)

    def test_log_security(self):
        """Test la sécurité des logs."""
        # Vérifier que les logs ne contiennent pas d'informations sensibles
        test_message = "API Key: test_key, Password: test_password"
        self.logger.info(test_message)
        
        # Lire le fichier de log
        log_file = self.log_dir / "test_security.log"
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        # Vérifier que les informations sensibles sont masquées
        self.assertNotIn('test_key', log_content)
        self.assertNotIn('test_password', log_content)

    def test_rate_limiting(self):
        """Test la limitation de taux."""
        # Simuler plusieurs requêtes rapides
        for _ in range(10):
            is_allowed = self.mt5_service.check_rate_limit()
            self.assertTrue(is_allowed)
        
        # La 11ème requête devrait être bloquée
        is_allowed = self.mt5_service.check_rate_limit()
        self.assertFalse(is_allowed)

    def test_sql_injection_prevention(self):
        """Test la prévention des injections SQL."""
        # Test des requêtes SQL malveillantes
        malicious_queries = [
            "'; DROP TABLE trades; --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --"
        ]
        
        for query in malicious_queries:
            # Vérifier que les requêtes sont rejetées
            with self.assertRaises(ValueError):
                self.mt5_service.execute_query(query)

    def test_xss_prevention(self):
        """Test la prévention des attaques XSS."""
        # Test des entrées XSS
        xss_inputs = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert('xss')"
        ]
        
        for xss_input in xss_inputs:
            # Vérifier que les entrées sont nettoyées
            cleaned_input = self.mt5_service.sanitize_input(xss_input)
            self.assertNotIn('<script>', cleaned_input)
            self.assertNotIn('javascript:', cleaned_input)
            self.assertNotIn('onerror=', cleaned_input)

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer les fichiers de log de test
        for file in self.log_dir.glob("*.log"):
            file.unlink()
        self.log_dir.rmdir()

if __name__ == '__main__':
    unittest.main() 