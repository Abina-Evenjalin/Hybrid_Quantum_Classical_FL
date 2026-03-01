"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 1/10: Core Imports & Configuration
═══════════════════════════════════════════════════════════════
Research Project: Hybrid Quantum-Classical Federated Learning
Author: [Abina Evenjalin - s16420]
University: [University of Colombo]

This block contains all necessary imports, global configurations,
and color scheme definitions for the professional purple theme.

ENHANCED VERSION: Supports Numerical, Image, and Text Data
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# STANDARD LIBRARY IMPORTS
# ═══════════════════════════════════════════════════════════════
import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import traceback

# ═══════════════════════════════════════════════════════════════
# PYQT5 IMPORTS (GUI Framework)
# ═══════════════════════════════════════════════════════════════
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QRadioButton, QCheckBox, QProgressBar, QTabWidget,
    QGroupBox, QScrollArea, QFileDialog, QMessageBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QStackedWidget,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QGridLayout,
    QToolButton, QMenu, QAction, QStatusBar, QMenuBar, QSizePolicy, QButtonGroup
)

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QRect, QPropertyAnimation,
    QEasingCurve, QPoint, QRectF, pyqtSlot, QObject, QEvent
)

from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QPen, QBrush,
    QLinearGradient, QRadialGradient, QPainterPath, QImage, QDrag,
    QCursor, QMovie, QPaintEvent, QMouseEvent, QDragEnterEvent, QDropEvent
)

# ═══════════════════════════════════════════════════════════════
# SCIENTIFIC COMPUTING IMPORTS
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Backend for PyQt5 integration
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# ═══════════════════════════════════════════════════════════════
# MACHINE LEARNING IMPORTS
# ═══════════════════════════════════════════════════════════════
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ═══════════════════════════════════════════════════════════════
# DEEP LEARNING IMPORTS (for Image & Text)
# ═══════════════════════════════════════════════════════════════
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
    print("✅ PyTorch successfully imported - Image processing available")
except (ImportError, OSError) as e:
    print(f"⚠️  PyTorch not available: {e}")
    print("💡 For image support: pip install torch torchvision pillow")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers available - Text processing enabled")
except ImportError as e:
    print(f"⚠️  Sentence Transformers not available: {e}")
    print("💡 For text support: pip install sentence-transformers")

# ═══════════════════════════════════════════════════════════════
# QUANTUM COMPUTING IMPORTS (with fallback)
# ═══════════════════════════════════════════════════════════════
QISKIT_AVAILABLE = False
QISKIT_IBM_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    print("✅ Qiskit successfully imported - Real quantum circuits available")
except ImportError as e:
    print(f"⚠️  Qiskit not available: {e}")
    print("💡 Running in SIMULATION mode with classical fallback")
    print("💡 To use real quantum circuits: pip install qiskit qiskit-aer")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QISKIT_IBM_AVAILABLE = True
    print("✅ IBM Quantum Runtime available")
except ImportError:
    print("⚠️  IBM Quantum Runtime not available")
    print("💡 To use IBM Cloud: pip install qiskit-ibm-runtime")

# ═══════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Config:
    """Global application configuration"""
    
    # Application metadata
    APP_NAME = "Quantum Federated Learning Platform"
    APP_VERSION = "2.0.0"  # Updated to v2.0 for multi-modal support
    APP_AUTHOR = "R.A.Evenjalin - S16420"
    APP_UNIVERSITY = "University of Colombo"
    
    # Window settings
    WINDOW_MIN_WIDTH = 1200
    WINDOW_MIN_HEIGHT = 700
    WINDOW_DEFAULT_WIDTH = 1400
    WINDOW_DEFAULT_HEIGHT = 900
    
    # Quantum settings
    MIN_QUBITS = 2
    MAX_QUBITS = 10
    DEFAULT_QUBITS = 4
    
    # Federated learning settings
    MIN_CLIENTS = 2
    MAX_CLIENTS = 10
    DEFAULT_CLIENTS = 3
    
    MIN_ROUNDS = 1
    MAX_ROUNDS = 50
    DEFAULT_ROUNDS = 10
    
    MIN_LOCAL_EPOCHS = 1
    MAX_LOCAL_EPOCHS = 20
    DEFAULT_LOCAL_EPOCHS = 5
    
    DEFAULT_LEARNING_RATE = 0.01
    
    # ═══════════════════════════════════════════════════════════
    # MULTI-MODAL DATA SETTINGS (NEW)
    # ═══════════════════════════════════════════════════════════
    
    # Data types supported
    DATA_TYPES = ['numerical', 'image', 'text']
    
    # Numerical data settings
    SUPPORTED_NUMERICAL_FORMATS = ['.csv', '.xlsx', '.xls', '.json']
    MAX_FILE_SIZE_MB = 100
    MIN_SAMPLES = 10
    MAX_FEATURES = 50
    MIN_CLASSES = 2
    MAX_CLASSES = 100
    
    # Image data settings
    IMAGE_SIZE = 224  # Standard for ResNet/MobileNet
    IMAGE_CHANNELS = 3  # RGB
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    MAX_IMAGES_PER_CLASS = 10000
    IMAGE_FEATURE_EXTRACTORS = {
        'resnet18': {'dim': 512, 'name': 'ResNet-18 (Fast)'},
        'resnet50': {'dim': 2048, 'name': 'ResNet-50 (Accurate)'},
        'mobilenet': {'dim': 1280, 'name': 'MobileNet (Efficient)'}
    }
    DEFAULT_IMAGE_EXTRACTOR = 'resnet18'
    
    # Text data settings
    MAX_TEXT_LENGTH = 512  # BERT standard
    SUPPORTED_TEXT_FORMATS = ['.txt', '.csv', '.json']
    TEXT_EMBEDDING_MODELS = {
        'minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dim': 384,
            'display': 'MiniLM (Fast, 384d)'
        },
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dim': 768,
            'display': 'MPNet (Accurate, 768d)'
        },
        'distilbert': {
            'name': 'sentence-transformers/distilbert-base-nli-mean-tokens',
            'dim': 768,
            'display': 'DistilBERT (Balanced, 768d)'
        }
    }
    DEFAULT_TEXT_EMBEDDER = 'minilm'

# ═══════════════════════════════════════════════════════════════
# COLOR SCHEME - PURPLE QUANTUM THEME
# ═══════════════════════════════════════════════════════════════

class Colors:
    """Professional purple quantum color palette"""
    
    # Primary purple gradient
    PRIMARY_DARK = "#6366f1"      # Deep indigo
    PRIMARY = "#8b5cf6"            # Rich purple
    PRIMARY_LIGHT = "#a855f7"      # Bright purple
    PRIMARY_LIGHTER = "#c084fc"    # Light purple
    
    # Background colors
    BG_DARKEST = "#0f0a1e"        # Deep space
    BG_DARK = "#1e1b4b"           # Dark indigo
    BG_MEDIUM = "#2d2654"         # Medium purple-gray
    BG_LIGHT = "#3d3567"          # Light purple-gray
    
    # Accent colors
    ACCENT_CYAN = "#06b6d4"       # Quantum circuit lines
    ACCENT_PINK = "#ec4899"       # Quantum states
    ACCENT_PURPLE = "#a855f7"     # Highlights
    
    # Status colors
    SUCCESS = "#10b981"           # Green
    WARNING = "#f59e0b"           # Amber
    ERROR = "#ef4444"             # Red
    INFO = "#3b82f6"              # Blue
    
    # Text colors
    TEXT_PRIMARY = "#f8fafc"      # Almost white
    TEXT_SECONDARY = "#cbd5e1"    # Light gray
    TEXT_MUTED = "#94a3b8"        # Dim gray
    TEXT_DISABLED = "#64748b"     # Very dim
    
    # Border colors
    BORDER_DEFAULT = "#8b5cf6"    # Purple
    BORDER_LIGHT = "#a855f7"      # Light purple
    BORDER_DARK = "#6366f1"       # Dark purple
    
    # Hover states
    HOVER_BG = "#3d3567"
    HOVER_BORDER = "#c084fc"
    
    # Card colors
    CARD_BG = "rgba(30, 27, 75, 0.6)"  # Semi-transparent
    CARD_BORDER = "#8b5cf6"
    
    @staticmethod
    def gradient_string(color1, color2, direction="x1:0, y1:0, x2:1, y2:0"):
        """Generate PyQt gradient string"""
        return f"qlineargradient({direction}, stop:0 {color1}, stop:1 {color2})"
    
    @staticmethod
    def rgba(hex_color, alpha=1.0):
        """Convert hex to rgba with alpha"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

# ═══════════════════════════════════════════════════════════════
# FONT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Fonts:
    """Typography configuration"""
    
    # Font families
    PRIMARY_FAMILY = "Segoe UI, Arial, sans-serif"
    MONO_FAMILY = "Consolas, Monaco, monospace"
    
    # Font sizes
    TITLE_SIZE = 24
    HEADING1_SIZE = 20
    HEADING2_SIZE = 16
    HEADING3_SIZE = 14
    BODY_SIZE = 13
    SMALL_SIZE = 11
    TINY_SIZE = 9
    
    # Font weights
    WEIGHT_LIGHT = 300
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700
    
    @staticmethod
    def get_font(size=13, weight=400, family=None):
        """Create QFont with specified properties"""
        font = QFont(family or Fonts.PRIMARY_FAMILY, size)
        font.setWeight(weight)
        return font

# ═══════════════════════════════════════════════════════════════
# ICON PATHS (Unicode symbols for cross-platform compatibility)
# ═══════════════════════════════════════════════════════════════

class Icons:
    """Unicode icons for UI elements"""
    
    # Navigation
    QUANTUM = "⚛️"
    SETUP = "⚙️"
    DATASET = "📊"
    TRAIN = "🔬"
    RESULTS = "📈"
    CONFIG = "🔧"
    
    # Actions
    UPLOAD = "📁"
    DOWNLOAD = "💾"
    PLAY = "▶️"
    STOP = "⏸️"
    REFRESH = "🔄"
    DELETE = "🗑️"
    
    # Status
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    LOADING = "⏳"
    
    # Data types (NEW)
    NUMERICAL = "📊"
    IMAGE = "🖼️"
    TEXT = "📝"
    
    # Other
    HELP = "❓"
    CLOSE = "✖️"
    SEARCH = "🔍"
    SETTINGS = "⚙️"

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def log_message(message, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def format_time(seconds):
    """Format seconds to MM:SS or HH:MM:SS"""
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_number(num, decimals=2):
    """Format number with specified decimals"""
    if isinstance(num, (int, float)):
        return f"{num:.{decimals}f}"
    return str(num)

def validate_file_size(file_path, max_size_mb=100):
    """Validate file size in MB"""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb <= max_size_mb, size_mb

def get_file_extension(file_path):
    """Get file extension in lowercase"""
    return Path(file_path).suffix.lower()

# ═══════════════════════════════════════════════════════════════
# APPLICATION PATHS
# ═══════════════════════════════════════════════════════════════

class Paths:
    """Application directory paths"""
    
    BASE_DIR = Path(__file__).parent
    RESOURCES_DIR = BASE_DIR / "resources"
    IMAGES_DIR = RESOURCES_DIR / "images"
    EXPORTS_DIR = BASE_DIR / "exports"
    CACHE_DIR = BASE_DIR / ".cache"
    
    # Model cache directories (NEW)
    MODELS_DIR = CACHE_DIR / "models"
    IMAGE_MODELS_DIR = MODELS_DIR / "image"
    TEXT_MODELS_DIR = MODELS_DIR / "text"
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist"""
        for directory in [Paths.RESOURCES_DIR, Paths.IMAGES_DIR, 
                         Paths.EXPORTS_DIR, Paths.CACHE_DIR,
                         Paths.MODELS_DIR, Paths.IMAGE_MODELS_DIR,
                         Paths.TEXT_MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class DatasetValidationError(Exception):
    """Raised when dataset validation fails"""
    pass

class QuantumBackendError(Exception):
    """Raised when quantum backend connection fails"""
    pass

class TrainingError(Exception):
    """Raised when training process fails"""
    pass

class UnsupportedDataTypeError(Exception):
    """Raised when data type is not supported"""
    pass

class FeatureExtractionError(Exception):
    """Raised when feature extraction fails"""
    pass

# ═══════════════════════════════════════════════════════════════
# CAPABILITY CHECKER 
# ═══════════════════════════════════════════════════════════════

class CapabilityChecker:
    """Check which data modalities are supported based on installed packages"""
    
    @staticmethod
    def get_available_modalities():
        """
        Check which data types can be processed.
        
        Returns:
            dict: Available modalities and their status
        """
        return {
            'numerical': {
                'available': True,  # Always available
                'reason': 'Core functionality'
            },
            'image': {
                'available': TORCH_AVAILABLE,
                'reason': 'PyTorch installed' if TORCH_AVAILABLE else 'Install: pip install torch torchvision pillow'
            },
            'text': {
                'available': TRANSFORMERS_AVAILABLE,
                'reason': 'Transformers installed' if TRANSFORMERS_AVAILABLE else 'Install: pip install sentence-transformers'
            }
        }
    
    @staticmethod
    def get_capability_summary():
        """Get human-readable capability summary"""
        caps = CapabilityChecker.get_available_modalities()
        
        summary = "Platform Capabilities:\n"
        summary += "=" * 60 + "\n"
        
        for modality, info in caps.items():
            icon = Icons.SUCCESS if info['available'] else Icons.WARNING
            status = "Available" if info['available'] else "Not Available"
            summary += f"{icon} {modality.upper()}: {status} - {info['reason']}\n"
        
        return summary
    
    @staticmethod
    def check_modality_support(data_type):
        """
        Check if a specific modality is supported.
        
        Args:
            data_type (str): 'numerical', 'image', or 'text'
            
        Returns:
            bool: True if supported
        """
        caps = CapabilityChecker.get_available_modalities()
        return caps.get(data_type, {}).get('available', False)

# ═══════════════════════════════════════════════════════════════
# INITIALIZE PATHS ON IMPORT
# ═══════════════════════════════════════════════════════════════

Paths.ensure_directories()

# ═══════════════════════════════════════════════════════════════
# STARTUP CAPABILITY CHECK
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(CapabilityChecker.get_capability_summary())
print("="*60 + "\n")

# ═══════════════════════════════════════════════════════════════
# BLOCK 1 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 1/10: Core Imports & Configuration - LOADED ✅")
print("MULTI-MODAL SUPPORT: Numerical + Image + Text")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 2/10: Quantum Circuit Implementation
═══════════════════════════════════════════════════════════════
This block contains:
- Fallback quantum circuit (classical simulation)
- Real quantum circuit (Qiskit implementation)
- IBM Quantum Cloud integration
- Quantum circuit factory
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# FALLBACK QUANTUM CIRCUIT (Classical Simulation)
# ═══════════════════════════════════════════════════════════════

class FallbackQuantumCircuit:
    """
    Classical simulation of quantum circuit when Qiskit unavailable.
    Uses neural network-style computations to approximate quantum behavior.
    """
    
    def __init__(self, n_qubits):
        """
        Initialize fallback quantum circuit.
        
        Args:
            n_qubits (int): Number of qubits to simulate
        """
        self.n_qubits = n_qubits
        self.n_params = n_qubits * 3  # RX, RY, RZ for each qubit
        self.param_values = np.random.uniform(-np.pi, np.pi, self.n_params)
        
        log_message(f"Fallback quantum simulation initialized with {n_qubits} qubits", "WARNING")
    
    def forward(self, features):
        """
        Simulate quantum forward pass using classical computation.
        
        Args:
            features (np.array): Input feature vector
            
        Returns:
            float: Simulated quantum output in range [-1, 1]
        """
        # Pad features if needed
        if len(features) < self.n_qubits:
            features_padded = np.zeros(self.n_qubits)
            features_padded[:len(features)] = features
            features = features_padded
        
        # Classical approximation of quantum encoding + variational layers
        # Layer 1: Feature encoding (simulates RY gates)
        encoded = np.tanh(features[:self.n_qubits] * self.param_values[:self.n_qubits])
        
        # Layer 2: Entanglement simulation (simulates CNOT gates)
        entangled = np.zeros_like(encoded)
        for i in range(len(encoded)):
            # Simulate entanglement by mixing adjacent qubits
            prev_idx = (i - 1) % len(encoded)
            next_idx = (i + 1) % len(encoded)
            entangled[i] = 0.6 * encoded[i] + 0.2 * encoded[prev_idx] + 0.2 * encoded[next_idx]
        
        # Layer 3: Variational rotations (simulates RX, RZ gates)
        rx_params = self.param_values[self.n_qubits:2*self.n_qubits]
        rz_params = self.param_values[2*self.n_qubits:3*self.n_qubits]
        
        output = np.tanh(np.sum(entangled * np.cos(rx_params) * np.sin(rz_params)))
        
        return np.clip(output, -1, 1)
    
    def compute_quantum_gradients(self, features, error):
        """
        Simulate quantum gradient computation using finite differences.
        
        Args:
            features (np.array): Input features
            error (float): Prediction error
            
        Returns:
            np.array: Simulated gradients for all parameters
        """
        gradients = np.zeros_like(self.param_values)
        epsilon = 0.01
        
        for i in range(len(gradients)):
            # Forward difference approximation
            self.param_values[i] += epsilon
            output_plus = self.forward(features)
            
            self.param_values[i] -= 2 * epsilon
            output_minus = self.forward(features)
            
            # Restore original value
            self.param_values[i] += epsilon
            
            # Compute gradient
            gradients[i] = (output_plus - output_minus) / (2 * epsilon) * error
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update circuit parameters using gradients.
        
        Args:
            gradients (np.array): Computed gradients
            learning_rate (float): Learning rate for updates
        """
        self.param_values -= learning_rate * gradients
        self.param_values = np.clip(self.param_values, -2*np.pi, 2*np.pi)
    
    def get_circuit_info(self):
        """
        Get information about the simulated circuit.
        
        Returns:
            dict: Circuit information
        """
        return {
            'n_qubits': self.n_qubits,
            'n_parameters': len(self.param_values),
            'circuit_depth': 4,  # Encoding + Entanglement + 2 Variational layers
            'gate_count': self.n_qubits * 6,  # Approximate
            'has_entanglement': True,
            'parameter_names': [f'θ_{i}' for i in range(len(self.param_values))],
            'backend': 'Classical Simulation (Fallback)'
        }
    
    def get_circuit_diagram(self):
        """
        Get text representation of the circuit.
        
        Returns:
            str: Circuit diagram as text
        """
        diagram = f"""

   FALLBACK QUANTUM CIRCUIT (Classical Simulation)         

                                                           
   Qubits: {self.n_qubits}                                              
   Parameters: {len(self.param_values)}                                         
   Layers: 4 (Encoding + Entanglement + Variational)      
                                                           
   Circuit Structure:                                      
   ┌──────────┐  ┌──────────┐  ┌──────────┐              
   │ Encoding │→ │Entangle  │→ │Variational│→ Measure    
   │  (RY)    │  │ (CNOT)   │  │ (RX, RZ)  │              
   └──────────┘  └──────────┘  └──────────┘              
                                                           
   Note: This is a classical approximation                
         Install Qiskit for real quantum circuits         

"""
        return diagram
    
    def run_circuit(self, param_values=None, shots=1024):
        """
        Simulate circuit execution (for API compatibility).
        
        Args:
            param_values (np.array): Optional parameter values
            shots (int): Number of measurements (ignored in simulation)
            
        Returns:
            dict: Simulated measurement counts
        """
        # Simulate binary measurement outcomes
        output = self.forward(np.random.randn(self.n_qubits))
        
        # Convert to bitstring counts (simulated)
        prob_zero = (1 + output) / 2  # Map [-1, 1] to [0, 1]
        
        counts = {
            '0' * self.n_qubits: int(prob_zero * shots),
            '1' * self.n_qubits: int((1 - prob_zero) * shots)
        }
        
        return counts

# ═══════════════════════════════════════════════════════════════
# REAL QUANTUM CIRCUIT (Qiskit Implementation)
# ═══════════════════════════════════════════════════════════════

if QISKIT_AVAILABLE:
    
    class RealQuantumCircuit:
        """
        Real quantum circuit implementation using IBM Qiskit.
        Supports both local simulation (Aer) and IBM Quantum Cloud.
        """
        
        def __init__(self, n_qubits, use_ibm=False):
            """
            Initialize real quantum circuit.
            
            Args:
                n_qubits (int): Number of qubits
                use_ibm (bool): Whether to use IBM Quantum Cloud
            """
            self.n_qubits = n_qubits
            self.n_params = n_qubits * 3  # RX, RY, RZ for each qubit
            
            # Create parameterized circuit
            self.parameters = [Parameter(f'θ_{i}') for i in range(self.n_params)]
            self.param_values = np.random.uniform(-np.pi, np.pi, self.n_params)
            
            # Build the variational quantum circuit
            self.circuit = self._build_variational_circuit()
            
            # Initialize backend
            self.use_ibm = use_ibm and QISKIT_IBM_AVAILABLE
            self.backend_name = "Unknown"
            
            if self.use_ibm:
                self._initialize_ibm_backend()
            else:
                self._initialize_aer_backend()
        
        def _initialize_aer_backend(self):
            """Initialize local Aer simulator"""
            self.simulator = AerSimulator()
            self.backend_name = "AerSimulator (Local)"
            self.use_ibm = False
            log_message(f"Quantum circuit using {self.backend_name}", "INFO")
        
        def _initialize_ibm_backend(self):
            """Initialize IBM Quantum Cloud backend"""
            try:
                # Try to connect to IBM Quantum
                self.service = QiskitRuntimeService(channel="ibm_cloud")
                
                # Get available backends
                available_backends = self.service.backends(
                    operational=True,
                    simulator=False
                )
                
                if available_backends:
                    # Use least busy backend
                    self.backend = self.service.least_busy(
                        operational=True,
                        simulator=False
                    )
                    self.backend_name = f"IBM {self.backend.name}"
                    log_message(f"Connected to {self.backend_name}", "SUCCESS")
                else:
                    # No real hardware available, fall back to cloud simulator
                    log_message("No real quantum hardware available, using cloud simulator", "WARNING")
                    self.backend = self.service.least_busy(
                        operational=True,
                        simulator=True
                    )
                    self.backend_name = f"IBM {self.backend.name} (Simulator)"
                
            except Exception as e:
                log_message(f"IBM Quantum connection failed: {e}", "ERROR")
                log_message("Falling back to local AerSimulator", "INFO")
                self._initialize_aer_backend()
        
        def _build_variational_circuit(self):
            """
            Build parameterized variational quantum circuit.
            
            Returns:
                QuantumCircuit: Parameterized circuit
            """
            qreg = QuantumRegister(self.n_qubits, 'q')
            creg = ClassicalRegister(self.n_qubits, 'c')
            circuit = QuantumCircuit(qreg, creg)
            
            param_idx = 0
            
            # Layer 1: Feature encoding using RY gates
            for i in range(self.n_qubits):
                circuit.ry(self.parameters[param_idx], qreg[i])
                param_idx += 1
            
            # Layer 2: Entanglement using CNOT gates
            # Linear entanglement
            for i in range(self.n_qubits - 1):
                circuit.cx(qreg[i], qreg[i + 1])
            
            # Circular entanglement (connect last to first)
            if self.n_qubits > 2:
                circuit.cx(qreg[-1], qreg[0])
            
            # Layer 3: Variational layer with RX gates
            for i in range(self.n_qubits):
                circuit.rx(self.parameters[param_idx], qreg[i])
                param_idx += 1
            
            # Layer 4: Variational layer with RZ gates
            for i in range(self.n_qubits):
                circuit.rz(self.parameters[param_idx], qreg[i])
                param_idx += 1
            
            # Measurement
            circuit.measure(qreg, creg)
            
            return circuit
        
        def forward(self, features):
            """
            Forward pass through quantum circuit.
            
            Args:
                features (np.array): Input feature vector
                
            Returns:
                float: Expectation value from quantum circuit
            """
            try:
                # Encode features into parameters (angle encoding)
                encoded_params = self.param_values.copy()
                for i in range(min(len(features), self.n_qubits)):
                    encoded_params[i] = features[i] * np.pi
                
                # Run circuit and get measurement counts
                counts = self.run_circuit(encoded_params, shots=1024)
                
                # Convert counts to expectation value
                expectation = self._counts_to_expectation(counts)
                
                return expectation
                
            except Exception as e:
                log_message(f"Quantum forward pass error: {e}", "ERROR")
                return 0.0
        
        def run_circuit(self, param_values=None, shots=1024):
            """
            Execute quantum circuit with given parameters.
            
            Args:
                param_values (np.array): Parameter values for circuit
                shots (int): Number of measurement shots
                
            Returns:
                dict: Measurement counts
            """
            if param_values is not None:
                self.param_values = param_values
            
            # Bind parameters to circuit
            try:
                bound_circuit = self.circuit.assign_parameters(
                    {p: v for p, v in zip(self.parameters, self.param_values)}
                )
            except AttributeError:
                # Fallback for older Qiskit versions
                bound_circuit = self.circuit.bind_parameters(
                    {p: v for p, v in zip(self.parameters, self.param_values)}
                )
            
            # Execute on backend
            if self.use_ibm:
                try:
                    job = self.backend.run(bound_circuit, shots=shots)
                    result = job.result()
                    return result.get_counts(bound_circuit)
                except Exception as e:
                    log_message(f"IBM execution failed: {e}, falling back to Aer", "WARNING")
                    self._initialize_aer_backend()
            
            # Execute on local simulator
            job = self.simulator.run(bound_circuit, shots=shots)
            result = job.result()
            return result.get_counts(bound_circuit)
        
        def _counts_to_expectation(self, counts):
            """
            Convert measurement counts to expectation value.
            Uses parity-based measurement.
            
            Args:
                counts (dict): Measurement outcome counts
                
            Returns:
                float: Expectation value in range [-1, 1]
            """
            total_shots = sum(counts.values())
            expectation = 0.0
            
            for bitstring, count in counts.items():
                # Count number of 1s in bitstring
                ones = bitstring.count('1')
                # Parity: even number of 1s → +1, odd → -1
                parity = 1 if ones % 2 == 0 else -1
                expectation += parity * (count / total_shots)
            
            return expectation
        
        def compute_quantum_gradients(self, features, error):
            """
            Compute gradients using parameter shift rule.
            
            Parameter shift rule:
            ∂f/∂θᵢ = [f(θᵢ + π/2) - f(θᵢ - π/2)] / 2
            
            Args:
                features (np.array): Input features
                error (float): Prediction error
                
            Returns:
                np.array: Gradients for all parameters
            """
            gradients = np.zeros_like(self.param_values)
            shift = np.pi / 2
            
            for i in range(len(gradients)):
                # Shift parameter forward
                params_plus = self.param_values.copy()
                params_plus[i] += shift
                
                # Shift parameter backward
                params_minus = self.param_values.copy()
                params_minus[i] -= shift
                
                # Compute outputs with shifted parameters
                counts_plus = self.run_circuit(params_plus, shots=512)
                output_plus = self._counts_to_expectation(counts_plus)
                
                counts_minus = self.run_circuit(params_minus, shots=512)
                output_minus = self._counts_to_expectation(counts_minus)
                
                # Calculate gradient
                gradients[i] = (output_plus - output_minus) / 2 * error
            
            # Restore original parameters
            self.param_values = self.param_values.copy()
            
            return gradients
        
        def update_parameters(self, gradients, learning_rate):
            """
            Update quantum circuit parameters.
            
            Args:
                gradients (np.array): Computed gradients
                learning_rate (float): Learning rate
            """
            self.param_values -= learning_rate * gradients
            self.param_values = np.clip(self.param_values, -2*np.pi, 2*np.pi)
        
        def get_circuit_info(self):
            """
            Get comprehensive circuit information.
            
            Returns:
                dict: Circuit metadata
            """
            return {
                'n_qubits': self.n_qubits,
                'n_parameters': len(self.param_values),
                'circuit_depth': self.circuit.depth(),
                'gate_count': sum(self.circuit.count_ops().values()),
                'has_entanglement': True,
                'parameter_names': [f'θ_{i}' for i in range(len(self.param_values))],
                'backend': self.backend_name
            }
        
        def get_circuit_diagram(self):
            """
            Get visual representation of circuit.
            
            Returns:
                str: Circuit diagram
            """
            try:
                diagram_str = str(self.circuit.draw(output='text'))
                
                header = f"""

   REAL QUANTUM CIRCUIT (Qiskit)                           
   Backend: {self.backend_name:<43} 
   Qubits: {self.n_qubits:<48} 
   Parameters: {len(self.param_values):<44} 
   Circuit Depth: {self.circuit.depth():<41} 
"""

                footer = """

"""
                return header + diagram_str + footer
                
            except Exception as e:
                return f"""

   REAL QUANTUM CIRCUIT (Qiskit)                           
   Backend: {self.backend_name:<43} 
   Qubits: {self.n_qubits:<48} 
   Parameters: {len(self.param_values):<44} 
   Depth: {self.circuit.depth():<48} 
                                                           
   Circuit Layers:                                         
   1. Feature Encoding (RY gates)                         
   2. Entanglement (CNOT gates - linear + circular)       
   3. Variational Layer 1 (RX gates)                      
   4. Variational Layer 2 (RZ gates)                      
   5. Measurement (Z-basis)                               
"""

# ═══════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT FACTORY
# ═══════════════════════════════════════════════════════════════

def create_quantum_circuit(n_qubits, use_ibm_hardware=False):
    """
    Factory function to create appropriate quantum circuit.
    Automatically selects real or fallback implementation.
    
    Args:
        n_qubits (int): Number of qubits
        use_ibm_hardware (bool): Whether to use IBM Quantum Cloud
        
    Returns:
        QuantumCircuit: Either RealQuantumCircuit or FallbackQuantumCircuit
    """
    if QISKIT_AVAILABLE:
        log_message(f"Creating real quantum circuit with {n_qubits} qubits", "INFO")
        return RealQuantumCircuit(n_qubits, use_ibm=use_ibm_hardware)
    else:
        log_message(f"Creating fallback quantum circuit with {n_qubits} qubits", "WARNING")
        return FallbackQuantumCircuit(n_qubits)

# ═══════════════════════════════════════════════════════════════
# QUANTUM BACKEND MANAGER
# ═══════════════════════════════════════════════════════════════

class QuantumBackendManager:
    """
    Manages quantum backend connections and status.
    Provides unified interface for Aer, IBM Cloud, and future hardware.
    """
    
    def __init__(self):
        """Initialize backend manager"""
        self.current_backend = "aer"  # Default: local simulator
        self.ibm_token_saved = False
        self.available_backends = self._detect_available_backends()
    
    def _detect_available_backends(self):
        """
        Detect which quantum backends are available.
        
        Returns:
            dict: Available backends with status
        """
        backends = {
            'aer': {
                'name': 'Aer Simulator (Local)',
                'available': QISKIT_AVAILABLE,
                'type': 'simulator',
                'description': 'Local quantum simulator - Fast, ideal for development'
            },
            'ibm_cloud': {
                'name': 'IBM Quantum Cloud',
                'available': QISKIT_IBM_AVAILABLE,
                'type': 'cloud',
                'description': 'Real quantum hardware via IBM Cloud - Requires API token'
            },
            'university': {
                'name': 'University 1-Qubit Computer',
                'available': False,  # Placeholder for future integration
                'type': 'hardware',
                'description': 'University lab quantum device - Under development'
            }
        }
        
        return backends
    
    def check_ibm_token(self):
        """
        Check if IBM Quantum token is saved.
        
        Returns:
            bool: True if token is saved and valid
        """
        if not QISKIT_IBM_AVAILABLE:
            return False
        
        try:
            service = QiskitRuntimeService(channel="ibm_cloud")
            self.ibm_token_saved = True
            return True
        except Exception:
            self.ibm_token_saved = False
            return False
    
    def get_backend_status(self):
        """
        Get status of all backends.
        
        Returns:
            dict: Backend status information
        """
        status = {}
        
        for backend_id, info in self.available_backends.items():
            status[backend_id] = {
                'name': info['name'],
                'available': info['available'],
                'active': backend_id == self.current_backend,
                'description': info['description']
            }
        
        # Update IBM Cloud status if token is saved
        if self.check_ibm_token():
            status['ibm_cloud']['status'] = 'Connected'
        else:
            status['ibm_cloud']['status'] = 'No token saved'
        
        return status
    
    def set_backend(self, backend_id):
        """
        Set active quantum backend.
        
        Args:
            backend_id (str): Backend identifier ('aer', 'ibm_cloud', 'university')
            
        Returns:
            bool: True if successful
        """
        if backend_id not in self.available_backends:
            log_message(f"Unknown backend: {backend_id}", "ERROR")
            return False
        
        if not self.available_backends[backend_id]['available']:
            log_message(f"Backend not available: {backend_id}", "ERROR")
            return False
        
        self.current_backend = backend_id
        log_message(f"Switched to backend: {self.available_backends[backend_id]['name']}", "INFO")
        return True
# ═══════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT FACTORY 
# ═══════════════════════════════════════════════════════════════

    def create_quantum_circuit(n_qubits, use_ibm_hardware=False):
        """
        Factory function to create quantum circuits.
    
        Args:
            n_qubits: Number of qubits
            use_ibm_hardware: Whether to use real IBM quantum hardware
    
        Returns:
            Quantum circuit instance (Real or Fallback)
        """
        if QISKIT_AVAILABLE:
            return RealQuantumCircuit(n_qubits, use_ibm=use_ibm_hardware)
        else:
            return FallbackQuantumCircuit(n_qubits)

# ═══════════════════════════════════════════════════════════════
# BLOCK 2 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 2/10: Quantum Circuit Implementation - LOADED ✅")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 3/10: Hybrid Models (Quantum & Classical)
═══════════════════════════════════════════════════════════════
This block contains:
- Hybrid Quantum-Classical Model
- Pure Classical Model (baseline)
- Multi-class classification support
- Model comparison utilities
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# HYBRID QUANTUM-CLASSICAL MODEL
# ═══════════════════════════════════════════════════════════════

class HybridQuantumModel:
    """
    Hybrid model combining quantum circuits with classical neural networks.
    Supports both binary and multi-class classification.
    """
    
    def __init__(self, n_qubits, input_dim, n_classes=2, use_ibm_hardware=False):
        """
        Initialize hybrid quantum-classical model.
        
        Args:
            n_qubits (int): Number of qubits in quantum circuit
            input_dim (int): Number of input features
            n_classes (int): Number of output classes (2 for binary, 3+ for multi-class)
            use_ibm_hardware (bool): Whether to use IBM Quantum hardware
        """
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_ibm_hardware = use_ibm_hardware
        
        # Create quantum circuit
        self.quantum_circuit = create_quantum_circuit(n_qubits, use_ibm_hardware)
        
        # Classical neural network components
        if n_classes == 2:
            # Binary classification: single output
            self.classical_weights = np.random.randn(2, 1) * 0.1
            self.classical_bias = np.zeros(1)
        else:
            # Multi-class: one-vs-rest approach with multiple circuits
            self.quantum_circuits = [
                create_quantum_circuit(n_qubits, use_ibm_hardware) 
                for _ in range(n_classes)
            ]
            self.classical_weights = np.random.randn(n_classes + 1, n_classes) * 0.1
            self.classical_bias = np.zeros(n_classes)
        
        circuit_type = "Real Qiskit" if QISKIT_AVAILABLE else "Fallback Simulation"
        backend = "IBM Cloud" if use_ibm_hardware else "Local Aer"
        
        log_message(
            f"Hybrid model initialized: {n_qubits} qubits ({circuit_type}/{backend}) "
            f"+ classical NN, {n_classes} classes",
            "INFO"
        )
    
    def forward(self, x):
        """
        Forward pass through hybrid quantum-classical model.
        
        Args:
            x (np.array): Input feature vector
            
        Returns:
            float or np.array: Prediction (binary: single value, multi-class: probabilities)
        """
        try:
            # Ensure input has enough features
            x_padded = self._pad_input(x)
            
            if self.n_classes == 2:
                # Binary classification
                return self._forward_binary(x_padded)
            else:
                # Multi-class classification
                return self._forward_multiclass(x_padded)
                
        except Exception as e:
            log_message(f"Forward pass error: {e}", "ERROR")
            if self.n_classes == 2:
                return 0.5
            else:
                return np.ones(self.n_classes) / self.n_classes
    
    def _pad_input(self, x):
        """Pad input to match quantum circuit dimensions"""
        if len(x) < self.n_qubits:
            x_padded = np.zeros(self.n_qubits)
            x_padded[:len(x)] = x
            return x_padded
        return x[:self.n_qubits]
    
    def _forward_binary(self, x):
        """
        Binary classification forward pass.
        
        Args:
            x (np.array): Padded input
            
        Returns:
            float: Prediction probability for class 1
        """
        # Quantum computation
        quantum_output = self.quantum_circuit.forward(x)
        
        # Classical feature extraction
        classical_features = np.mean(x)
        
        # Combine quantum and classical outputs
        combined_input = np.array([quantum_output, classical_features])
        
        # Classical neural network final layer
        linear_output = np.dot(combined_input, self.classical_weights.flatten()) + self.classical_bias[0]
        
        # Sigmoid activation with numerical stability
        return self._sigmoid(linear_output)
    
    def _forward_multiclass(self, x):
        """
        Multi-class classification forward pass using one-vs-rest.
        
        Args:
            x (np.array): Padded input
            
        Returns:
            np.array: Class probabilities (sums to 1)
        """
        # Get quantum outputs from all circuits (one per class)
        quantum_outputs = np.array([
            circuit.forward(x) for circuit in self.quantum_circuits
        ])
        
        # Classical feature
        classical_features = np.mean(x)
        
        # Combine all outputs
        combined_input = np.concatenate([quantum_outputs, [classical_features]])
        
        # Classical neural network
        logits = np.dot(combined_input, self.classical_weights) + self.classical_bias
        
        # Softmax for probabilities
        return self._softmax(logits)
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict(self, x):
        """
        Make prediction for input.
        
        Args:
            x (np.array): Input features
            
        Returns:
            int: Predicted class label
        """
        output = self.forward(x)
        
        if self.n_classes == 2:
            return 1 if output > 0.5 else 0
        else:
            return np.argmax(output)
    
    def get_parameters(self):
        """
        Get all model parameters.
        
        Returns:
            dict: Model parameters
        """
        if self.n_classes == 2:
            return {
                'quantum': self.quantum_circuit.param_values.copy(),
                'classical_weights': self.classical_weights.copy(),
                'classical_bias': self.classical_bias.copy()
            }
        else:
            return {
                'quantum': [circuit.param_values.copy() for circuit in self.quantum_circuits],
                'classical_weights': self.classical_weights.copy(),
                'classical_bias': self.classical_bias.copy()
            }
    
    def set_parameters(self, params):
        """
        Set model parameters.
        
        Args:
            params (dict): Model parameters
        """
        if self.n_classes == 2:
            self.quantum_circuit.param_values = params['quantum'].copy()
        else:
            for i, circuit in enumerate(self.quantum_circuits):
                circuit.param_values = params['quantum'][i].copy()
        
        self.classical_weights = params['classical_weights'].copy()
        self.classical_bias = params['classical_bias'].copy()
    
    def compute_gradients(self, x, y_true, prediction):
        """
        Compute gradients for both quantum and classical parts.
        
        Args:
            x (np.array): Input features
            y_true (int or np.array): True label
            prediction (float or np.array): Model prediction
            
        Returns:
            tuple: (quantum_gradients, classical_gradients)
        """
        x_padded = self._pad_input(x)
        
        if self.n_classes == 2:
            return self._compute_gradients_binary(x_padded, y_true, prediction)
        else:
            return self._compute_gradients_multiclass(x_padded, y_true, prediction)
    
    def _compute_gradients_binary(self, x, y_true, prediction):
        """Compute gradients for binary classification"""
        error = prediction - y_true
        
        # Quantum gradients using parameter shift rule
        quantum_gradients = self.quantum_circuit.compute_quantum_gradients(x, error)
        
        # Classical gradients
        quantum_out = self.quantum_circuit.forward(x)
        classical_features = np.mean(x)
        combined_input = np.array([quantum_out, classical_features])
        classical_gradients = error * combined_input
        
        return quantum_gradients, classical_gradients
    
    def _compute_gradients_multiclass(self, x, y_true, prediction):
        """Compute gradients for multi-class classification"""
        # One-hot encode true label
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[y_true] = 1
        
        # Error for each class
        error = prediction - y_one_hot
        
        # Quantum gradients for each circuit
        quantum_gradients = []
        for i, circuit in enumerate(self.quantum_circuits):
            grads = circuit.compute_quantum_gradients(x, error[i])
            quantum_gradients.append(grads)
        
        # Classical gradients
        quantum_outputs = np.array([circuit.forward(x) for circuit in self.quantum_circuits])
        classical_features = np.mean(x)
        combined_input = np.concatenate([quantum_outputs, [classical_features]])
        
        classical_gradients = np.outer(combined_input, error)
        
        return quantum_gradients, classical_gradients
    
    def update(self, quantum_grads, classical_grads, learning_rate):
        """
        Update model parameters.
        
        Args:
            quantum_grads: Quantum circuit gradients
            classical_grads: Classical network gradients
            learning_rate (float): Learning rate
        """
        if self.n_classes == 2:
            # Update single quantum circuit
            self.quantum_circuit.update_parameters(quantum_grads, learning_rate)
            
            # Update classical weights
            self.classical_weights -= learning_rate * classical_grads.reshape(-1, 1)
            self.classical_weights = np.clip(self.classical_weights, -10, 10)
        else:
            # Update multiple quantum circuits
            for i, circuit in enumerate(self.quantum_circuits):
                circuit.update_parameters(quantum_grads[i], learning_rate)
            
            # Update classical weights
            self.classical_weights -= learning_rate * classical_grads
            self.classical_weights = np.clip(self.classical_weights, -10, 10)
    
    def get_model_info(self):
        """
        Get comprehensive model information.
        
        Returns:
            dict: Model metadata
        """
        if self.n_classes == 2:
            circuit_info = self.quantum_circuit.get_circuit_info()
            quantum_params = circuit_info['n_parameters']
        else:
            quantum_params = sum(
                circuit.get_circuit_info()['n_parameters'] 
                for circuit in self.quantum_circuits
            )
        
        classical_params = self.classical_weights.size + self.classical_bias.size
        
        return {
            'model_type': 'Hybrid Quantum-Classical',
            'n_classes': self.n_classes,
            'quantum_qubits': self.n_qubits,
            'quantum_parameters': quantum_params,
            'classical_parameters': classical_params,
            'total_parameters': quantum_params + classical_params,
            'backend': self.quantum_circuit.get_circuit_info()['backend'] if self.n_classes == 2 
                      else self.quantum_circuits[0].get_circuit_info()['backend'],
            'has_entanglement': True
        }

# ═══════════════════════════════════════════════════════════════
# PURE CLASSICAL MODEL (Baseline)
# ═══════════════════════════════════════════════════════════════

class ClassicalModel:
    """
    Pure classical neural network for baseline comparison.
    Supports both binary and multi-class classification.
    """
    
    def __init__(self, input_dim, n_classes=2, hidden_units=8):
        """
        Initialize classical model.
        
        Args:
            input_dim (int): Number of input features
            n_classes (int): Number of output classes
            hidden_units (int): Number of hidden layer neurons
        """
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        
        # Network architecture: input → hidden → output
        self.hidden_weights = np.random.randn(input_dim, hidden_units) * 0.1
        self.hidden_bias = np.zeros(hidden_units)
        
        if n_classes == 2:
            self.output_weights = np.random.randn(hidden_units, 1) * 0.1
            self.output_bias = np.zeros(1)
        else:
            self.output_weights = np.random.randn(hidden_units, n_classes) * 0.1
            self.output_bias = np.zeros(n_classes)
        
        log_message(
            f"Classical model initialized: {input_dim} → {hidden_units} → {n_classes} "
            f"({self._count_parameters()} parameters)",
            "INFO"
        )
    
    def _count_parameters(self):
        """Count total trainable parameters"""
        return (self.hidden_weights.size + self.hidden_bias.size + 
                self.output_weights.size + self.output_bias.size)
    
    def forward(self, x):
        """
        Forward pass through classical network.
        
        Args:
            x (np.array): Input features
            
        Returns:
            float or np.array: Prediction
        """
        try:
            # Ensure correct input shape
            x = self._prepare_input(x)
            
            # Hidden layer with tanh activation
            hidden = np.tanh(np.dot(x, self.hidden_weights) + self.hidden_bias)
            
            # Output layer
            if self.n_classes == 2:
                # Binary: sigmoid activation
                output = np.dot(hidden, self.output_weights.flatten()) + self.output_bias[0]
                return self._sigmoid(output)
            else:
                # Multi-class: softmax activation
                logits = np.dot(hidden, self.output_weights) + self.output_bias
                return self._softmax(logits)
                
        except Exception as e:
            log_message(f"Forward pass error: {e}", "ERROR")
            if self.n_classes == 2:
                return 0.5
            else:
                return np.ones(self.n_classes) / self.n_classes
    
    def _prepare_input(self, x):
        """Ensure input has correct dimensions"""
        if len(x) < self.input_dim:
            x_padded = np.zeros(self.input_dim)
            x_padded[:len(x)] = x
            return x_padded
        elif len(x) > self.input_dim:
            return x[:self.input_dim]
        return x
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict(self, x):
        """
        Make prediction for input.
        
        Args:
            x (np.array): Input features
            
        Returns:
            int: Predicted class label
        """
        output = self.forward(x)
        
        if self.n_classes == 2:
            return 1 if output > 0.5 else 0
        else:
            return np.argmax(output)
    
    def get_parameters(self):
        """
        Get all model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            'hidden_weights': self.hidden_weights.copy(),
            'hidden_bias': self.hidden_bias.copy(),
            'output_weights': self.output_weights.copy(),
            'output_bias': self.output_bias.copy()
        }
    
    def set_parameters(self, params):
        """
        Set model parameters.
        
        Args:
            params (dict): Model parameters
        """
        self.hidden_weights = params['hidden_weights'].copy()
        self.hidden_bias = params['hidden_bias'].copy()
        self.output_weights = params['output_weights'].copy()
        self.output_bias = params['output_bias'].copy()
    
    def compute_gradients(self, x, y_true, prediction):
        """
        Compute gradients using backpropagation.
        
        Args:
            x (np.array): Input features
            y_true (int or np.array): True label
            prediction (float or np.array): Model prediction
            
        Returns:
            dict: Gradients for all parameters
        """
        x = self._prepare_input(x)
        
        if self.n_classes == 2:
            return self._compute_gradients_binary(x, y_true, prediction)
        else:
            return self._compute_gradients_multiclass(x, y_true, prediction)
    
    def _compute_gradients_binary(self, x, y_true, prediction):
        """Compute gradients for binary classification"""
        error = prediction - y_true
        
        # Forward pass to get hidden activations
        hidden = np.tanh(np.dot(x, self.hidden_weights) + self.hidden_bias)
        
        # Output layer gradients
        output_grad_weights = error * hidden
        output_grad_bias = error
        
        # Hidden layer gradients (backpropagation)
        hidden_error = error * self.output_weights.flatten()
        hidden_grad = hidden_error * (1 - hidden ** 2)  # tanh derivative
        hidden_grad_weights = np.outer(x, hidden_grad)
        hidden_grad_bias = hidden_grad
        
        return {
            'hidden_weights': hidden_grad_weights,
            'hidden_bias': hidden_grad_bias,
            'output_weights': output_grad_weights.reshape(-1, 1),
            'output_bias': np.array([output_grad_bias])
        }
    
    def _compute_gradients_multiclass(self, x, y_true, prediction):
        """Compute gradients for multi-class classification"""
        # One-hot encode true label
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[y_true] = 1
        
        # Error
        error = prediction - y_one_hot
        
        # Forward pass to get hidden activations
        hidden = np.tanh(np.dot(x, self.hidden_weights) + self.hidden_bias)
        
        # Output layer gradients
        output_grad_weights = np.outer(hidden, error)
        output_grad_bias = error
        
        # Hidden layer gradients
        hidden_error = np.dot(self.output_weights, error)
        hidden_grad = hidden_error * (1 - hidden ** 2)
        hidden_grad_weights = np.outer(x, hidden_grad)
        hidden_grad_bias = hidden_grad
        
        return {
            'hidden_weights': hidden_grad_weights,
            'hidden_bias': hidden_grad_bias,
            'output_weights': output_grad_weights,
            'output_bias': output_grad_bias
        }
    
    def update(self, gradients, learning_rate):
        """
        Update model parameters.
        
        Args:
            gradients (dict): Computed gradients
            learning_rate (float): Learning rate
        """
        self.hidden_weights -= learning_rate * gradients['hidden_weights']
        self.hidden_bias -= learning_rate * gradients['hidden_bias']
        self.output_weights -= learning_rate * gradients['output_weights']
        self.output_bias -= learning_rate * gradients['output_bias']
        
        # Clip to prevent explosion
        self.hidden_weights = np.clip(self.hidden_weights, -10, 10)
        self.output_weights = np.clip(self.output_weights, -10, 10)
    
    def get_model_info(self):
        """
        Get model information.
        
        Returns:
            dict: Model metadata
        """
        return {
            'model_type': 'Classical Neural Network',
            'n_classes': self.n_classes,
            'quantum_qubits': 0,
            'quantum_parameters': 0,
            'classical_parameters': self._count_parameters(),
            'total_parameters': self._count_parameters(),
            'backend': 'CPU (NumPy)',
            'has_entanglement': False
        }

# ═══════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════

def create_model(model_type, input_dim, n_classes=2, n_qubits=4, use_ibm=False):
    """
    Factory function to create models.
    
    Args:
        model_type (str): 'quantum' or 'classical'
        input_dim (int): Number of input features
        n_classes (int): Number of output classes
        n_qubits (int): Number of qubits (for quantum models)
        use_ibm (bool): Use IBM Quantum hardware
        
    Returns:
        Model instance
    """
    if model_type.lower() == 'quantum':
        return HybridQuantumModel(n_qubits, input_dim, n_classes, use_ibm)
    elif model_type.lower() == 'classical':
        return ClassicalModel(input_dim, n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════════

class ModelComparator:
    """Utility class for comparing quantum and classical models"""
    
    @staticmethod
    def compare_parameters(quantum_model, classical_model):
        """
        Compare parameter counts between models.
        
        Args:
            quantum_model: HybridQuantumModel instance
            classical_model: ClassicalModel instance
            
        Returns:
            dict: Comparison statistics
        """
        q_info = quantum_model.get_model_info()
        c_info = classical_model.get_model_info()
        
        return {
            'quantum_total': q_info['total_parameters'],
            'quantum_breakdown': {
                'quantum': q_info['quantum_parameters'],
                'classical': q_info['classical_parameters']
            },
            'classical_total': c_info['total_parameters'],
            'quantum_advantage': q_info['quantum_parameters'] > 0,
            'parameter_ratio': q_info['total_parameters'] / c_info['total_parameters']
        }
    
    @staticmethod
    def evaluate_both(quantum_model, classical_model, X_test, y_test):
        """
        Evaluate both models on test data.
        
        Args:
            quantum_model: HybridQuantumModel instance
            classical_model: ClassicalModel instance
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        results = {
            'quantum': {'correct': 0, 'total': len(X_test), 'predictions': []},
            'classical': {'correct': 0, 'total': len(X_test), 'predictions': []}
        }
        
        for i in range(len(X_test)):
            # Quantum prediction
            q_pred = quantum_model.predict(X_test[i])
            if q_pred == y_test[i]:
                results['quantum']['correct'] += 1
            results['quantum']['predictions'].append(q_pred)
            
            # Classical prediction
            c_pred = classical_model.predict(X_test[i])
            if c_pred == y_test[i]:
                results['classical']['correct'] += 1
            results['classical']['predictions'].append(c_pred)
        
        # Calculate accuracies
        results['quantum']['accuracy'] = results['quantum']['correct'] / results['quantum']['total']
        results['classical']['accuracy'] = results['classical']['correct'] / results['classical']['total']
        
        # Quantum advantage
        results['quantum_advantage'] = results['quantum']['accuracy'] > results['classical']['accuracy']
        results['accuracy_difference'] = results['quantum']['accuracy'] - results['classical']['accuracy']
        
        return results

# ═══════════════════════════════════════════════════════════════
# BLOCK 3 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 3/10: Hybrid Models (Quantum & Classical) - LOADED ✅")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 4/10: Federated Learning Core
═══════════════════════════════════════════════════════════════
This block contains:
- Federated Learning Client
- Federated Learning Server
- Federated averaging algorithms
- Training coordination
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# FEDERATED LEARNING CLIENT
# ═══════════════════════════════════════════════════════════════

class FederatedClient:
    """
    Federated learning client that trains on local data.
    Supports both quantum and classical models.
    """
    
    def __init__(self, client_id, model, data, client_name=None):
        """
        Initialize federated client.
        
        Args:
            client_id (int): Unique client identifier
            model: Model instance (HybridQuantumModel or ClassicalModel)
            data (tuple): (X_train, y_train) local training data
            client_name (str): Optional human-readable name
        """
        self.client_id = client_id
        self.client_name = client_name or f"Client_{client_id}"
        self.model = model
        self.x_train, self.y_train = data
        
        # Training history
        self.training_history = {
            'rounds': [],
            'losses': [],
            'accuracies': [],
            'training_times': []
        }
        
        # Client statistics
        self.total_samples = len(self.x_train)
        self.n_features = self.x_train.shape[1] if len(self.x_train.shape) > 1 else 1
        self.n_classes = len(np.unique(self.y_train))
        
        log_message(
            f"{self.client_name} initialized: {self.total_samples} samples, "
            f"{self.n_features} features, {self.n_classes} classes",
            "INFO"
        )
    
    def local_training(self, epochs, learning_rate, round_number=None):
        """
        Perform local training on client's data.
        
        Args:
            epochs (int): Number of local training epochs
            learning_rate (float): Learning rate
            round_number (int): Current federated round number
            
        Returns:
            dict: Training results (loss, accuracy, time)
        """
        start_time = time.time()
        
        model_type = "Quantum" if isinstance(self.model, HybridQuantumModel) else "Classical"
        log_message(
            f"{self.client_name} starting local {model_type} training "
            f"(Round {round_number}, {epochs} epochs)",
            "INFO"
        )
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(self.x_train))
            
            for idx in indices:
                x, y = self.x_train[idx], self.y_train[idx]
                
                try:
                    # Forward pass
                    prediction = self.model.forward(x)
                    
                    # Compute loss
                    loss = self._compute_loss(prediction, y)
                    epoch_loss += loss
                    
                    # Check accuracy
                    predicted_class = self.model.predict(x)
                    if predicted_class == y:
                        correct_predictions += 1
                    
                    # Compute and apply gradients
                    if isinstance(self.model, HybridQuantumModel):
                        # Quantum model
                        quantum_grads, classical_grads = self.model.compute_gradients(
                            x, y, prediction
                        )
                        self.model.update(quantum_grads, classical_grads, learning_rate)
                    else:
                        # Classical model
                        gradients = self.model.compute_gradients(x, y, prediction)
                        self.model.update(gradients, learning_rate)
                    
                except Exception as e:
                    log_message(
                        f"Training error in {self.client_name}, sample {idx}: {e}",
                        "ERROR"
                    )
                    continue
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(self.x_train) if len(self.x_train) > 0 else 0
            accuracy = correct_predictions / len(self.x_train) if len(self.x_train) > 0 else 0
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            # Log progress every few epochs
            if epoch % max(1, epochs // 3) == 0 or epoch == epochs - 1:
                log_message(
                    f"  {self.client_name} Epoch {epoch+1}/{epochs}: "
                    f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}",
                    "INFO"
                )
        
        training_time = time.time() - start_time
        
        # Store in history
        if round_number is not None:
            self.training_history['rounds'].append(round_number)
            self.training_history['losses'].append(losses[-1])
            self.training_history['accuracies'].append(accuracies[-1])
            self.training_history['training_times'].append(training_time)
        
        final_results = {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'training_time': training_time,
            'epoch_losses': losses,
            'epoch_accuracies': accuracies
        }
        
        log_message(
            f"{self.client_name} completed: "
            f"Loss={final_results['final_loss']:.4f}, "
            f"Acc={final_results['final_accuracy']:.4f}, "
            f"Time={training_time:.2f}s",
            "SUCCESS"
        )
        
        return final_results
    
    def _compute_loss(self, prediction, y_true):
        """
        Compute loss (cross-entropy for classification).
        
        Args:
            prediction (float or np.array): Model prediction
            y_true (int): True label
            
        Returns:
            float: Loss value
        """
        if isinstance(prediction, np.ndarray):
            # Multi-class: cross-entropy loss
            # Clip to avoid log(0)
            prediction = np.clip(prediction, 1e-8, 1 - 1e-8)
            return -np.log(prediction[int(y_true)])
        else:
            # Binary: binary cross-entropy
            prediction = np.clip(prediction, 1e-8, 1 - 1e-8)
            return -(y_true * np.log(prediction) + (1 - y_true) * np.log(1 - prediction))
    
    def get_client_stats(self):
        """
        Get client statistics.
        
        Returns:
            dict: Client information and statistics
        """
        model_info = self.model.get_model_info()
        
        return {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'total_samples': self.total_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'model_type': model_info['model_type'],
            'training_rounds': len(self.training_history['rounds']),
            'best_accuracy': max(self.training_history['accuracies']) 
                           if self.training_history['accuracies'] else 0,
            'total_training_time': sum(self.training_history['training_times'])
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate client's model on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        correct = 0
        predictions = []
        
        for i in range(len(X_test)):
            pred = self.model.predict(X_test[i])
            predictions.append(pred)
            if pred == y_test[i]:
                correct += 1
        
        accuracy = correct / len(X_test) if len(X_test) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(X_test),
            'predictions': predictions
        }

# ═══════════════════════════════════════════════════════════════
# FEDERATED LEARNING SERVER
# ═══════════════════════════════════════════════════════════════

class FederatedServer:
    """
    Federated learning server that coordinates training and aggregates models.
    """
    
    def __init__(self, model_type, input_dim, n_classes=2, n_qubits=4, use_ibm=False):
        """
        Initialize federated server.
        
        Args:
            model_type (str): 'quantum' or 'classical'
            input_dim (int): Number of input features
            n_classes (int): Number of output classes
            n_qubits (int): Number of qubits (for quantum)
            use_ibm (bool): Use IBM Quantum hardware
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.use_ibm = use_ibm
        
        # Create global model
        self.global_model = create_model(
            model_type, input_dim, n_classes, n_qubits, use_ibm
        )
        
        # Federated learning components
        self.clients = []
        self.training_history = {
            'rounds': [],
            'global_accuracies': [],
            'avg_client_accuracies': [],
            'avg_client_losses': [],
            'round_times': []
        }
        
        log_message(
            f"Federated server initialized: {model_type.upper()} mode, "
            f"{n_classes} classes, {input_dim} features",
            "INFO"
        )
    
    def add_client(self, client):
        """
        Add client to federated system.
        
        Args:
            client (FederatedClient): Client instance
        """
        self.clients.append(client)
        log_message(f"Added {client.client_name} to federated system", "INFO")
    
    def distribute_global_model(self):
        """
        Distribute global model parameters to all clients.
        """
        global_params = self.global_model.get_parameters()
        
        for client in self.clients:
            client.model.set_parameters(global_params)
        
        log_message(f"Distributed global model to {len(self.clients)} clients", "INFO")
    
    def federated_averaging(self, weighted=True):
        """
        Perform federated averaging to update global model.
        
        Args:
            weighted (bool): Whether to weight by client data size
            
        Returns:
            dict: Aggregation statistics
        """
        if not self.clients:
            log_message("No clients available for federated averaging", "ERROR")
            return None
        
        log_message(f"Starting federated averaging ({len(self.clients)} clients)...", "INFO")
        start_time = time.time()
        
        try:
            # Collect parameters from all clients
            client_params = []
            client_weights = []
            
            for client in self.clients:
                params = client.model.get_parameters()
                client_params.append(params)
                
                # Weight by number of samples
                if weighted:
                    client_weights.append(client.total_samples)
                else:
                    client_weights.append(1.0)
            
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            # Aggregate parameters
            if self.model_type == 'quantum':
                aggregated_params = self._aggregate_quantum_params(
                    client_params, client_weights
                )
            else:
                aggregated_params = self._aggregate_classical_params(
                    client_params, client_weights
                )
            
            # Update global model
            self.global_model.set_parameters(aggregated_params)
            
            aggregation_time = time.time() - start_time
            
            log_message(
                f"Federated averaging completed in {aggregation_time:.2f}s",
                "SUCCESS"
            )
            
            return {
                'n_clients': len(self.clients),
                'weighted': weighted,
                'aggregation_time': aggregation_time,
                'success': True
            }
            
        except Exception as e:
            log_message(f"Federated averaging failed: {e}", "ERROR")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _aggregate_quantum_params(self, client_params, weights):
        """
        Aggregate quantum model parameters.
        
        Args:
            client_params (list): List of parameter dictionaries
            weights (list): Client weights
            
        Returns:
            dict: Aggregated parameters
        """
        if self.n_classes == 2:
            # Binary classification: single quantum circuit
            avg_quantum = np.zeros_like(client_params[0]['quantum'])
            for params, weight in zip(client_params, weights):
                avg_quantum += weight * params['quantum']
        else:
            # Multi-class: multiple quantum circuits
            n_circuits = len(client_params[0]['quantum'])
            avg_quantum = [np.zeros_like(client_params[0]['quantum'][i]) 
                          for i in range(n_circuits)]
            
            for params, weight in zip(client_params, weights):
                for i in range(n_circuits):
                    avg_quantum[i] += weight * params['quantum'][i]
        
        # Aggregate classical parameters
        avg_classical_weights = np.zeros_like(client_params[0]['classical_weights'])
        avg_classical_bias = np.zeros_like(client_params[0]['classical_bias'])
        
        for params, weight in zip(client_params, weights):
            avg_classical_weights += weight * params['classical_weights']
            avg_classical_bias += weight * params['classical_bias']
        
        return {
            'quantum': avg_quantum,
            'classical_weights': avg_classical_weights,
            'classical_bias': avg_classical_bias
        }
    
    def _aggregate_classical_params(self, client_params, weights):
        """
        Aggregate classical model parameters.
        
        Args:
            client_params (list): List of parameter dictionaries
            weights (list): Client weights
            
        Returns:
            dict: Aggregated parameters
        """
        avg_params = {}
        
        for key in client_params[0].keys():
            avg_params[key] = np.zeros_like(client_params[0][key])
            
            for params, weight in zip(client_params, weights):
                avg_params[key] += weight * params[key]
        
        return avg_params
    
    def evaluate_global_model(self, X_test, y_test):
        """
        Evaluate global model on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        log_message(f"Evaluating global model on {len(X_test)} test samples...", "INFO")
        
        correct = 0
        predictions = []
        
        for i in range(len(X_test)):
            try:
                pred = self.global_model.predict(X_test[i])
                predictions.append(pred)
                
                if pred == y_test[i]:
                    correct += 1
                    
            except Exception as e:
                log_message(f"Evaluation error on sample {i}: {e}", "ERROR")
                continue
        
        accuracy = correct / len(X_test) if len(X_test) > 0 else 0
        
        log_message(f"Global model accuracy: {accuracy:.4f}", "SUCCESS")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(X_test),
            'predictions': predictions
        }
    
    def federated_round(self, round_number, local_epochs, learning_rate, X_test=None, y_test=None):
        """
        Execute one complete federated learning round.
        
        Args:
            round_number (int): Current round number
            local_epochs (int): Epochs for local training
            learning_rate (float): Learning rate
            X_test (np.array): Optional test data for evaluation
            y_test (np.array): Optional test labels
            
        Returns:
            dict: Round results
        """
        round_start_time = time.time()
        
        log_message(f"\n{'='*60}", "INFO")
        log_message(f"FEDERATED ROUND {round_number}", "INFO")
        log_message(f"{'='*60}", "INFO")
        
        # Step 1: Distribute global model to clients
        self.distribute_global_model()
        
        # Step 2: Local training on all clients
        client_results = []
        for client in self.clients:
            result = client.local_training(local_epochs, learning_rate, round_number)
            client_results.append(result)
        
        # Step 3: Federated averaging
        aggregation_result = self.federated_averaging(weighted=True)
        
        # Step 4: Evaluate global model
        global_accuracy = 0
        if X_test is not None and y_test is not None:
            eval_result = self.evaluate_global_model(X_test, y_test)
            global_accuracy = eval_result['accuracy']
        
        # Calculate round statistics
        avg_client_loss = np.mean([r['final_loss'] for r in client_results])
        avg_client_accuracy = np.mean([r['final_accuracy'] for r in client_results])
        round_time = time.time() - round_start_time
        
        # Store in history
        self.training_history['rounds'].append(round_number)
        self.training_history['global_accuracies'].append(global_accuracy)
        self.training_history['avg_client_accuracies'].append(avg_client_accuracy)
        self.training_history['avg_client_losses'].append(avg_client_loss)
        self.training_history['round_times'].append(round_time)
        
        # Summary
        round_summary = {
            'round_number': round_number,
            'global_accuracy': global_accuracy,
            'avg_client_accuracy': avg_client_accuracy,
            'avg_client_loss': avg_client_loss,
            'round_time': round_time,
            'client_results': client_results,
            'aggregation_result': aggregation_result
        }
        
        log_message(f"\nRound {round_number} Summary:", "INFO")
        log_message(f"  Global Accuracy: {global_accuracy:.4f}", "INFO")
        log_message(f"  Avg Client Accuracy: {avg_client_accuracy:.4f}", "INFO")
        log_message(f"  Avg Client Loss: {avg_client_loss:.4f}", "INFO")
        log_message(f"  Round Time: {round_time:.2f}s", "INFO")
        
        return round_summary
    
    def get_server_stats(self):
        """
        Get server statistics.
        
        Returns:
            dict: Server information and statistics
        """
        model_info = self.global_model.get_model_info()
        
        return {
            'model_type': self.model_type,
            'n_clients': len(self.clients),
            'n_classes': self.n_classes,
            'input_dim': self.input_dim,
            'quantum_qubits': model_info['quantum_qubits'],
            'total_parameters': model_info['total_parameters'],
            'training_rounds_completed': len(self.training_history['rounds']),
            'best_global_accuracy': max(self.training_history['global_accuracies']) 
                                   if self.training_history['global_accuracies'] else 0,
            'total_training_time': sum(self.training_history['round_times'])
        }
    
    def get_training_summary(self):
        """
        Get comprehensive training summary.
        
        Returns:
            dict: Detailed training summary
        """
        if not self.training_history['rounds']:
            return {'status': 'No training completed'}
        
        return {
            'total_rounds': len(self.training_history['rounds']),
            'final_global_accuracy': self.training_history['global_accuracies'][-1],
            'best_global_accuracy': max(self.training_history['global_accuracies']),
            'initial_accuracy': self.training_history['global_accuracies'][0],
            'improvement': self.training_history['global_accuracies'][-1] - 
                          self.training_history['global_accuracies'][0],
            'avg_round_time': np.mean(self.training_history['round_times']),
            'total_time': sum(self.training_history['round_times']),
            'convergence_round': self._find_convergence_round(),
            'client_summary': [client.get_client_stats() for client in self.clients]
        }
    
    def _find_convergence_round(self, threshold=0.01):
        """
        Find round where model converged (change < threshold).
        
        Args:
            threshold (float): Convergence threshold
            
        Returns:
            int: Round number where convergence occurred, or -1
        """
        if len(self.training_history['global_accuracies']) < 2:
            return -1
        
        for i in range(1, len(self.training_history['global_accuracies'])):
            change = abs(self.training_history['global_accuracies'][i] - 
                        self.training_history['global_accuracies'][i-1])
            if change < threshold:
                return i + 1
        
        return -1

# ═══════════════════════════════════════════════════════════════
# BLOCK 4 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 4/10: Federated Learning Core - LOADED ✅")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 5/10: Data Handler & Validator
═══════════════════════════════════════════════════════════════
This block contains:
- Dataset validation and preprocessing (numerical, image, text)
- Automatic configuration suggestions
- Data distribution for federated learning
- Dataset analysis utilities

ENHANCED VERSION: Multi-Modal Support
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# DATA TYPE DETECTOR (NEW)
# ═══════════════════════════════════════════════════════════════

class DataTypeDetector:
    """
    Automatically detect data type from uploaded files or folders.
    """
    
    @staticmethod
    def detect_data_type(file_or_folder_path):
        """
        Detect if data is numerical, image, or text.
        
        Args:
            file_or_folder_path (str or Path): Path to file or folder
            
        Returns:
            str: 'numerical', 'image', or 'text'
        """
        path = Path(file_or_folder_path)
        
        # Check if folder (likely images)
        if path.is_dir():
            # Look for image files
            image_files = []
            for ext in Config.SUPPORTED_IMAGE_FORMATS:
                image_files.extend(list(path.glob(f'**/*{ext}')))
            
            if len(image_files) > 0:
                log_message(f"Detected {len(image_files)} image files in folder", "INFO")
                return 'image'
            
            # Look for text files
            text_files = list(path.glob('**/*.txt'))
            if len(text_files) > 0:
                log_message(f"Detected {len(text_files)} text files in folder", "INFO")
                return 'text'
            
            raise UnsupportedDataTypeError("Folder contains no recognized image or text files")
        
        # Check file
        if path.is_file():
            ext = path.suffix.lower()
            
            # Image file
            if ext in Config.SUPPORTED_IMAGE_FORMATS:
                log_message(f"Detected single image file", "INFO")
                return 'image'
            
            # Text file
            if ext == '.txt':
                log_message(f"Detected text file", "INFO")
                return 'text'
            
            # CSV - need to check content
            if ext == '.csv':
                try:
                    df = pd.read_csv(path, nrows=10)
                    
                    # Check if contains text columns with substantial text
                    text_cols = df.select_dtypes(include=['object']).columns
                    if len(text_cols) > 0:
                        # Sample first text column
                        for col in text_cols:
                            sample = df[col].iloc[0]
                            if isinstance(sample, str) and len(sample.split()) > 5:
                                log_message(f"Detected CSV with text data in column '{col}'", "INFO")
                                return 'text'
                    
                    # Otherwise assume numerical
                    log_message("Detected numerical CSV", "INFO")
                    return 'numerical'
                    
                except Exception as e:
                    log_message(f"Error reading CSV: {e}", "ERROR")
                    raise
            
            # Excel
            if ext in ['.xlsx', '.xls']:
                log_message("Detected Excel file (assuming numerical)", "INFO")
                return 'numerical'
            
            # JSON
            if ext == '.json':
                log_message("Detected JSON file (assuming numerical)", "INFO")
                return 'numerical'
        
        raise UnsupportedDataTypeError(f"Cannot detect data type for: {path}")

# ═══════════════════════════════════════════════════════════════
# DATASET VALIDATOR (Enhanced for multi-modal)
# ═══════════════════════════════════════════════════════════════

class DatasetValidator:
    """
    Validates and analyzes uploaded datasets for quantum federated learning.
    Now supports numerical, image, and text data.
    """
    
    def __init__(self):
        """Initialize dataset validator"""
        self.validation_report = {}
    
    def validate(self, file_path):
        """
        Validate uploaded dataset file.
        
        Args:
            file_path (str): Path to dataset file
            
        Returns:
            dict: Validation results
        """
        log_message(f"Validating dataset: {file_path}", "INFO")
        
        try:
            # Check file existence
            if not os.path.exists(file_path):
                raise DatasetValidationError(f"File not found: {file_path}")
            
            # Check file size
            is_valid_size, size_mb = validate_file_size(
                file_path, 
                Config.MAX_FILE_SIZE_MB
            )
            if not is_valid_size:
                raise DatasetValidationError(
                    f"File too large: {size_mb:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)"
                )
            
            # Load dataset based on extension
            extension = get_file_extension(file_path)
            
            if extension == '.csv':
                data = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            elif extension == '.json':
                data = pd.read_json(file_path)
            else:
                raise DatasetValidationError(
                    f"Unsupported file format: {extension}. "
                    f"Supported: {', '.join(Config.SUPPORTED_NUMERICAL_FORMATS)}"
                )
            
            # Validate dataset structure
            validation_result = self._validate_structure(data)
            validation_result['file_path'] = file_path
            validation_result['file_size_mb'] = size_mb
            
            self.validation_report = validation_result
            
            log_message("Dataset validation successful", "SUCCESS")
            return validation_result
            
        except Exception as e:
            log_message(f"Dataset validation failed: {e}", "ERROR")
            raise DatasetValidationError(str(e))
    
    def _validate_structure(self, data):
        """
        Validate internal dataset structure.
        
        Args:
            data (pd.DataFrame): Loaded dataset
            
        Returns:
            dict: Validation results with metadata
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check minimum samples
        if len(data) < Config.MIN_SAMPLES:
            validation['errors'].append(
                f"Insufficient samples: {len(data)} (minimum: {Config.MIN_SAMPLES})"
            )
            validation['valid'] = False
        
        # Check columns
        if len(data.columns) < 2:
            validation['errors'].append(
                "Dataset must have at least 2 columns (features + target)"
            )
            validation['valid'] = False
            return validation
        
        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Analyze features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_features) == 0 and len(categorical_features) == 0:
            validation['errors'].append("No valid features detected")
            validation['valid'] = False
            return validation
        
        # Check for too many features
        total_features = len(numeric_features) + len(categorical_features)
        if total_features > Config.MAX_FEATURES:
            validation['warnings'].append(
                f"Many features detected ({total_features}). "
                f"Consider dimensionality reduction."
            )
        
        # Detect problem type
        unique_targets = y.nunique()
        
        if y.dtype in [np.float64, np.float32] and unique_targets > 20:
            problem_type = 'regression'
            validation['warnings'].append(
                "Detected regression problem. Platform supports classification. "
                "Will convert to classification by binning."
            )
        else:
            problem_type = 'classification'
        
        # Check number of classes
        if problem_type == 'classification':
            if unique_targets < Config.MIN_CLASSES:
                validation['errors'].append(
                    f"Too few classes: {unique_targets} (minimum: {Config.MIN_CLASSES})"
                )
                validation['valid'] = False
            
            if unique_targets > Config.MAX_CLASSES:
                validation['warnings'].append(
                    f"Many classes detected: {unique_targets}. "
                    f"Training may be slow."
                )
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            missing_percent = (missing_count / data.size) * 100
            validation['warnings'].append(
                f"Missing values detected: {missing_count} ({missing_percent:.1f}%). "
                f"Will be imputed automatically."
            )
        
        # Check class balance (for classification)
        if problem_type == 'classification':
            class_counts = y.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()
            
            if min_class_ratio < 0.3:
                validation['warnings'].append(
                    f"Imbalanced classes detected (ratio: {min_class_ratio:.2f}). "
                    f"Consider data balancing techniques."
                )
        
        # Store metadata
        validation['metadata'] = {
            'n_samples': len(data),
            'n_features': len(numeric_features) + len(categorical_features),
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'n_numeric': len(numeric_features),
            'n_categorical': len(categorical_features),
            'problem_type': problem_type,
            'n_classes': unique_targets if problem_type == 'classification' else None,
            'class_distribution': y.value_counts().to_dict() if problem_type == 'classification' else None,
            'missing_values': missing_count,
            'feature_names': data.columns[:-1].tolist(),
            'target_name': data.columns[-1]
        }
        
        return validation

# ═══════════════════════════════════════════════════════════════
# NUMERICAL DATA PREPROCESSOR 
# ═══════════════════════════════════════════════════════════════

class DatasetPreprocessor:
    """
    Preprocesses numerical datasets for quantum federated learning.
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = None
        self.label_encoder = None
        self.preprocessing_steps = []
    
    def preprocess(self, file_path, validation_result):
        """
        Preprocess dataset for quantum ML.
        
        Args:
            file_path (str): Path to dataset file
            validation_result (dict): Validation results from DatasetValidator
            
        Returns:
            tuple: (X, y, preprocessing_info)
        """
        log_message("Starting numerical dataset preprocessing...", "INFO")
        
        # Load data
        extension = get_file_extension(file_path)
        if extension == '.csv':
            data = pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif extension == '.json':
            data = pd.read_json(file_path)
        
        metadata = validation_result['metadata']
        
        # Separate features and target
        X = data.iloc[:, :-1].copy()
        y = data.iloc[:, -1].copy()
        
        # Step 1: Handle missing values
        if metadata['missing_values'] > 0:
            X, y = self._handle_missing_values(X, y)
        
        # Step 2: Encode categorical features
        if metadata['n_categorical'] > 0:
            X = self._encode_categorical(X, metadata['categorical_features'])
        
        # Step 3: Convert regression to classification if needed
        if metadata['problem_type'] == 'regression':
            y = self._convert_regression_to_classification(y)
            self.preprocessing_steps.append("Converted regression to classification (3 bins)")
        
        # Step 4: Encode target labels
        y = self._encode_labels(y)
        
        # Step 5: Normalize features
        X = self._normalize_features(X)
        
        # Convert to numpy arrays
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        
        # Final validation
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        preprocessing_info = {
            'data_type': 'numerical',
            'steps': self.preprocessing_steps,
            'final_shape': X.shape,
            'n_classes': len(np.unique(y)),
            'feature_range': (X.min(), X.max()),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        log_message(
            f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features, "
            f"{preprocessing_info['n_classes']} classes",
            "SUCCESS"
        )
        
        return X, y, preprocessing_info
    
    def _handle_missing_values(self, X, y):
        """Handle missing values in dataset"""
        log_message("Handling missing values...", "INFO")
        
        # For features: impute with mean (numeric) or mode (categorical)
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    X[col].fillna(X[col].mean(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        # For target: drop rows with missing targets
        if y.isnull().any():
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
            log_message(f"Dropped {(~valid_indices).sum()} rows with missing targets", "WARNING")
        
        self.preprocessing_steps.append("Imputed missing values")
        return X, y
    
    def _encode_categorical(self, X, categorical_features):
        """Encode categorical features using one-hot encoding"""
        log_message(f"Encoding {len(categorical_features)} categorical features...", "INFO")
        
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        self.preprocessing_steps.append(
            f"One-hot encoded {len(categorical_features)} categorical features"
        )
        return X
    
    def _convert_regression_to_classification(self, y, n_bins=3):
        """Convert regression target to classification"""
        log_message(f"Converting regression to {n_bins}-class classification...", "INFO")
        
        y = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        return y
    
    def _encode_labels(self, y):
        """Encode target labels to integers"""
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.preprocessing_steps.append(
            f"Encoded labels: {dict(enumerate(self.label_encoder.classes_))}"
        )
        return y_encoded
    
    def _normalize_features(self, X):
        """Normalize features to standard scale"""
        log_message("Normalizing features...", "INFO")
        
        self.scaler = StandardScaler()
        
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X = self.scaler.fit_transform(X)
        
        self.preprocessing_steps.append("Standardized features (mean=0, std=1)")
        return X

# ═══════════════════════════════════════════════════════════════
# IMAGE PREPROCESSOR 
# ═══════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """
    Preprocess images for quantum federated learning using CNN feature extraction.
    """
    
    def __init__(self, model_name='resnet18'):
        """
        Initialize with pretrained CNN for feature extraction.
        
        Args:
            model_name (str): CNN model name ('resnet18', 'resnet50', 'mobilenet')
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not available! Install with: pip install torch torchvision pillow"
            )
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        log_message(f"Using device: {self.device}", "INFO")
        
        # Load pretrained model
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = model.features
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        log_message(f"Image preprocessor initialized with {model_name}", "SUCCESS")
    
    def preprocess(self, folder_path):
        """
        Process image folder into feature vectors.
        
        Expected folder structure:
        folder_path/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                img4.jpg
        
        Args:
            folder_path (str or Path): Path to folder with class subfolders
            
        Returns:
            tuple: (X, y, preprocessing_info)
        """
        folder_path = Path(folder_path)
        
        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")
        
        # Find all class folders
        class_folders = sorted([f for f in folder_path.iterdir() if f.is_dir()])
        
        if len(class_folders) == 0:
            raise ValueError(
                "No class folders found!\n"
                "Expected structure: folder/class_name/images.jpg"
            )
        
        class_names = [f.name for f in class_folders]
        log_message(f"Found {len(class_folders)} classes: {class_names}", "INFO")
        
        # Extract features from all images
        all_features = []
        all_labels = []
        total_images = 0
        
        for class_idx, class_folder in enumerate(class_folders):
            # Find all image files in this class
            image_files = []
            for ext in Config.SUPPORTED_IMAGE_FORMATS:
                image_files.extend(list(class_folder.glob(f'*{ext}')))
            
            log_message(
                f"Processing class '{class_folder.name}': {len(image_files)} images",
                "INFO"
            )
            
            class_features = []
            
            for img_path in image_files:
                try:
                    # Load and transform image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    with torch.no_grad():
                        features = self.feature_extractor(img_tensor)
                        
                        # Flatten features
                        if self.model_name == 'mobilenet':
                            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        
                        features = features.squeeze().cpu().numpy()
                    
                    class_features.append(features)
                    all_labels.append(class_idx)
                    total_images += 1
                    
                except Exception as e:
                    log_message(f"Error processing {img_path.name}: {e}", "WARNING")
                    continue
            
            if class_features:
                all_features.extend(class_features)
        
        if len(all_features) == 0:
            raise FeatureExtractionError("No images were successfully processed!")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        preprocessing_info = {
            'data_type': 'image',
            'feature_extractor': self.model_name,
            'feature_dim': self.feature_dim,
            'n_classes': len(class_folders),
            'class_names': class_names,
            'total_images': len(X),
            'images_per_class': {name: sum(y == i) for i, name in enumerate(class_names)},
            'scaler': scaler,
            'device': str(self.device)
        }
        
        log_message(
            f"Image preprocessing complete: {len(X)} images → {self.feature_dim}D features",
            "SUCCESS"
        )
        
        return X, y, preprocessing_info

# ═══════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR 
# ═══════════════════════════════════════════════════════════════

class TextPreprocessor:
    """
    Preprocess text data using sentence embeddings.
    """
    
    def __init__(self, model_name='minilm'):
        """
        Initialize with pretrained text embedding model.
        
        Args:
            model_name (str): Model key from Config.TEXT_EMBEDDING_MODELS
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Sentence Transformers not available! "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_key = model_name
        model_config = Config.TEXT_EMBEDDING_MODELS.get(model_name)
        
        if model_config is None:
            raise ValueError(f"Unknown text model: {model_name}")
        
        self.model_name = model_config['name']
        self.embedding_dim = model_config['dim']
        self.model = None  # Loaded lazily in worker thread
        
        log_message(f"TextPreprocessor configured: {model_config['display']}", "INFO")
    
    def _ensure_model_loaded(self):
        """Load model lazily in the worker thread that will use it."""
        if self.model is None:
            log_message(f"Loading text model: {self.model_name}", "INFO")
            self.model = SentenceTransformer(self.model_name)
            log_message(f"Text preprocessor ready: {self.embedding_dim}D embeddings", "SUCCESS")
    
    def preprocess(self, file_or_folder_path):
        """
        Process text data into embeddings.
        
        Supports:
        1. CSV with 'text' and 'label' columns
        2. Folder with class subfolders containing .txt files
        
        Args:
            file_or_folder_path (str or Path): Path to CSV or folder
            
        Returns:
            tuple: (X, y, preprocessing_info)
        """
        path = Path(file_or_folder_path)
        
        if path.is_file() and path.suffix == '.csv':
            return self._preprocess_csv(path)
        elif path.is_dir():
            return self._preprocess_folder(path)
        else:
            raise ValueError(
                "Text data must be:\n"
                "1. CSV file with 'text' and 'label' columns, OR\n"
                "2. Folder with class subfolders containing .txt files"
            )
    
    def _preprocess_csv(self, csv_path):
        """Process CSV with text and labels"""
        self._ensure_model_loaded()
        log_message(f"Processing text CSV: {csv_path}", "INFO")
        
        df = pd.read_csv(csv_path)
        
        # Detect text and label columns
        if 'text' in df.columns:
            text_col = 'text'
        else:
            # Find first text column
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) == 0:
                raise ValueError("No text column found in CSV")
            text_col = text_cols[0]
            log_message(f"Using column '{text_col}' as text", "INFO")
        
        if 'label' in df.columns:
            label_col = 'label'
        else:
            # Use last column
            label_col = df.columns[-1]
            log_message(f"Using column '{label_col}' as label", "INFO")
        
        texts = df[text_col].fillna('').tolist()
        labels = df[label_col].tolist()
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Generate embeddings
        log_message(f"Generating embeddings for {len(texts)} texts...", "INFO")
        X = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        preprocessing_info = {
            'data_type': 'text',
            'embedding_model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'n_classes': len(np.unique(y)),
            'class_names': label_encoder.classes_.tolist(),
            'total_texts': len(X),
            'label_encoder': label_encoder,
            'text_column': text_col,
            'label_column': label_col
        }
        
        log_message(f"Text preprocessing complete: {len(X)} texts → {self.embedding_dim}D", "SUCCESS")
        
        return X, y, preprocessing_info
    
    def _preprocess_folder(self, folder_path):
        """Process folder of text files organized by class"""
        self._ensure_model_loaded()
        log_message(f"Processing text folder: {folder_path}", "INFO")
        
        class_folders = sorted([f for f in folder_path.iterdir() if f.is_dir()])
        
        if len(class_folders) == 0:
            raise ValueError(
                "No class folders found!\n"
                "Expected structure: folder/class_name/text_files.txt"
            )
        
        all_texts = []
        all_labels = []
        class_names = []
        
        for class_idx, class_folder in enumerate(class_folders):
            class_names.append(class_folder.name)
            text_files = list(class_folder.glob('*.txt'))
            
            log_message(
                f"Processing class '{class_folder.name}': {len(text_files)} files",
                "INFO"
            )
            
            for txt_file in text_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:  # Only add non-empty texts
                        all_texts.append(text)
                        all_labels.append(class_idx)
                        
                except Exception as e:
                    log_message(f"Error reading {txt_file.name}: {e}", "WARNING")
                    continue
        
        if len(all_texts) == 0:
            raise FeatureExtractionError("No text files were successfully processed!")
        
        # Generate embeddings
        log_message(f"Generating embeddings for {len(all_texts)} texts...", "INFO")
        X = self.model.encode(
            all_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        y = np.array(all_labels)
        
        preprocessing_info = {
            'data_type': 'text',
            'embedding_model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'n_classes': len(class_folders),
            'class_names': class_names,
            'total_texts': len(X),
            'texts_per_class': {name: sum(y == i) for i, name in enumerate(class_names)}
        }
        
        log_message(f"Text preprocessing complete: {len(X)} texts → {self.embedding_ratio}D", "SUCCESS")
        
        return X, y, preprocessing_info
# ═══════════════════════════════════════════════════════════════
# UNIVERSAL PREPROCESSOR (Router) 
# ═══════════════════════════════════════════════════════════════

class UniversalPreprocessor:
    """
    Routes to appropriate preprocessor based on data type.
    Automatically detects and processes numerical, image, or text data.
    """
    
    def __init__(self):
        self.image_preprocessor = None
        self.text_preprocessor = None
        self.numeric_preprocessor = DatasetPreprocessor()
    
    def preprocess(self, file_or_folder_path):
        """
        Automatically detect type and preprocess.
        
        Args:
            file_or_folder_path (str or Path): Path to data
            
        Returns:
            tuple: (X, y, preprocessing_info)
        """
        # Detect data type
        data_type = DataTypeDetector.detect_data_type(file_or_folder_path)
        
        log_message(f"📊 Detected data type: {data_type.upper()}", "INFO")
        
        # Check if modality is supported
        if not CapabilityChecker.check_modality_support(data_type):
            caps = CapabilityChecker.get_available_modalities()
            raise UnsupportedDataTypeError(
                f"{data_type.upper()} data not supported!\n"
                f"Reason: {caps[data_type]['reason']}"
            )
        
        # Route to appropriate preprocessor
        if data_type == 'numerical':
            log_message("Using numerical preprocessor", "INFO")
            validator = DatasetValidator()
            validation = validator.validate(file_or_folder_path)
            return self.numeric_preprocessor.preprocess(file_or_folder_path, validation)
        
        elif data_type == 'image':
            log_message("Using image preprocessor (CNN features)", "INFO")
            # Initialize image preprocessor if needed
            if self.image_preprocessor is None:
                model_name = Config.DEFAULT_IMAGE_EXTRACTOR
                self.image_preprocessor = ImagePreprocessor(model_name)
            return self.image_preprocessor.preprocess(file_or_folder_path)
        
        elif data_type == 'text':
            log_message("Using text preprocessor (embeddings)", "INFO")
            # Initialize text preprocessor if needed
            if self.text_preprocessor is None:
                model_name = Config.DEFAULT_TEXT_EMBEDDER
                self.text_preprocessor = TextPreprocessor(model_name)
            return self.text_preprocessor.preprocess(file_or_folder_path)
        
        else:
            raise UnsupportedDataTypeError(f"Unknown data type: {data_type}")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION SUGGESTER (Updated for multi-modal)
# ═══════════════════════════════════════════════════════════════

class ConfigurationSuggester:
    """
    Suggests optimal quantum and federated learning configurations.
    """
    
    @staticmethod
    def suggest_configuration(X, y, data_type='numerical', backend_type='aer'):
        """
        Suggest optimal configuration based on dataset.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
            data_type (str): 'numerical', 'image', or 'text'
            backend_type (str): Quantum backend type
            
        Returns:
            dict: Suggested configuration
        """
        n_samples = len(X)
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        n_classes = len(np.unique(y))
        
        # Suggest number of qubits (based on features, but capped)
        suggested_qubits = min(max(int(np.log2(n_features)) + 1, Config.MIN_QUBITS), Config.MAX_QUBITS)
        
        # Suggest number of clients (based on sample size)
        if n_samples < 50:
            suggested_clients = 2
        elif n_samples < 200:
            suggested_clients = 3
        elif n_samples < 500:
            suggested_clients = 4
        else:
            suggested_clients = 5
        
        # Suggest number of rounds (based on complexity)
        complexity = n_features * n_classes
        if complexity < 20:
            suggested_rounds = 8
        elif complexity < 50:
            suggested_rounds = 10
        elif complexity < 100:
            suggested_rounds = 15
        else:
            suggested_rounds = 20
        
        # Suggest local epochs
        samples_per_client = n_samples // suggested_clients
        if samples_per_client < 20:
            suggested_epochs = 5
        elif samples_per_client < 50:
            suggested_epochs = 3
        else:
            suggested_epochs = 3
        
        # Suggest learning rate (adjust for data type)
        if data_type == 'image':
            suggested_lr = 0.005  # Lower for pre-extracted features
        elif data_type == 'text':
            suggested_lr = 0.008  # Slightly lower for embeddings
        else:
            suggested_lr = 0.01  # Standard for numerical
        
        # Assess quantum advantage potential
        quantum_advantage_score = ConfigurationSuggester._assess_quantum_advantage(
            n_features, n_classes, n_samples, data_type
        )
        
        suggestions = {
            'qubits': suggested_qubits,
            'n_clients': suggested_clients,
            'n_rounds': suggested_rounds,
            'local_epochs': suggested_epochs,
            'learning_rate': suggested_lr,
            'quantum_advantage_score': quantum_advantage_score,
            'data_type': data_type,
            'reasoning': {
                'qubits': f"Based on {n_features} features (log₂ scale, capped at {Config.MAX_QUBITS})",
                'clients': f"Based on {n_samples} samples",
                'rounds': f"Based on problem complexity ({n_features}×{n_classes})",
                'epochs': f"Based on samples per client (~{samples_per_client})",
                'learning_rate': f"Optimized for {data_type} data"
            }
        }
        
        return suggestions
    
    @staticmethod
    def _assess_quantum_advantage(n_features, n_classes, n_samples, data_type):
        """
        Assess potential for quantum advantage.
        
        Returns:
            float: Score from 0-100
        """
        score = 0
        
        # Feature complexity (more features = more quantum advantage potential)
        if n_features >= 100:  # High-dimensional (images/text)
            score += 40
        elif n_features >= 10:
            score += 30
        elif n_features >= 4:
            score += 20
        else:
            score += 10
        
        # Multi-class complexity
        if n_classes > 5:
            score += 25
        elif n_classes > 2:
            score += 15
        else:
            score += 10
        
        # Sample size (quantum works well with moderate data)
        if 50 <= n_samples <= 500:
            score += 25
        elif n_samples < 50:
            score += 10
        else:
            score += 15
        
        # Data type bonus
        if data_type in ['image', 'text']:
            score += 10  # Non-linear patterns in image/text benefit from quantum
        
        return min(score, 100)

# ═══════════════════════════════════════════════════════════════
# DATA DISTRIBUTOR - works with any feature vectors
# ═══════════════════════════════════════════════════════════════

class DataDistributor:
    """
    Distributes data among federated learning clients.
    """
    
    @staticmethod
    def distribute_iid(X, y, n_clients):
        """
        Distribute data in IID (Independent and Identically Distributed) manner.
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            n_clients (int): Number of clients
            
        Returns:
            list: List of (X_client, y_client) tuples
        """
        log_message(f"Distributing data IID to {n_clients} clients...", "INFO")
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split evenly
        client_data = []
        samples_per_client = len(X) // n_clients
        
        for i in range(n_clients):
            start_idx = i * samples_per_client
            if i == n_clients - 1:
                # Last client gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * samples_per_client
            
            X_client = X_shuffled[start_idx:end_idx]
            y_client = y_shuffled[start_idx:end_idx]
            
            client_data.append((X_client, y_client))
            log_message(
                f"  Client {i+1}: {len(X_client)} samples",
                "INFO"
            )
        
        return client_data
    
    @staticmethod
    def distribute_non_iid(X, y, n_clients, alpha=0.5):
        """
        Distribute data in non-IID manner (class imbalance per client).
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            n_clients (int): Number of clients
            alpha (float): Dirichlet concentration parameter (lower = more non-IID)
            
        Returns:
            list: List of (X_client, y_client) tuples
        """
        log_message(
            f"Distributing data non-IID to {n_clients} clients (alpha={alpha})...",
            "INFO"
        )
        
        n_classes = len(np.unique(y))
        client_data = [[] for _ in range(n_clients)]
        
        # For each class, distribute samples using Dirichlet distribution
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            np.random.shuffle(class_indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * n_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            
            # Adjust to ensure all samples are assigned
            proportions[-1] = len(class_indices) - np.sum(proportions[:-1])
            
            # Distribute samples
            start_idx = 0
            for client_idx, proportion in enumerate(proportions):
                end_idx = start_idx + proportion
                client_indices = class_indices[start_idx:end_idx]
                client_data[client_idx].extend(client_indices)
                start_idx = end_idx
        
        # Convert indices to actual data
        result = []
        for client_idx, indices in enumerate(client_data):
            if len(indices) > 0:
                X_client = X[indices]
                y_client = y[indices]
                result.append((X_client, y_client))
                
                # Log class distribution
                unique, counts = np.unique(y_client, return_counts=True)
                dist_str = ", ".join([f"Class {u}: {c}" for u, c in zip(unique, counts)])
                log_message(f"  Client {client_idx+1}: {len(indices)} samples ({dist_str})", "INFO")
        
        return result

# ═══════════════════════════════════════════════════════════════
# DATASET ANALYZER - works with any feature vectors
# ═══════════════════════════════════════════════════════════════

class DatasetAnalyzer:
    """
    Provides detailed analysis of datasets.
    """
    
    @staticmethod
    def analyze(X, y):
        """
        Perform comprehensive dataset analysis.
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            
        Returns:
            dict: Analysis results
        """
        n_samples, n_features = X.shape if len(X.shape) > 1 else (len(X), 1)
        
        analysis = {
            'basic_stats': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': len(np.unique(y)),
                'class_labels': np.unique(y).tolist()
            },
            'feature_stats': {
                'mean': np.mean(X, axis=0).tolist() if n_features > 1 else [np.mean(X)],
                'std': np.std(X, axis=0).tolist() if n_features > 1 else [np.std(X)],
                'min': np.min(X, axis=0).tolist() if n_features > 1 else [np.min(X)],
                'max': np.max(X, axis=0).tolist() if n_features > 1 else [np.max(X)]
            },
            'class_distribution': {},
            'data_quality': {}
        }
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            analysis['class_distribution'][int(label)] = {
                'count': int(count),
                'percentage': float(count / n_samples * 100)
            }
        
        # Balance ratio
        balance_ratio = counts.min() / counts.max()
        analysis['data_quality']['balance_ratio'] = float(balance_ratio)
        analysis['data_quality']['is_balanced'] = balance_ratio > 0.7
        
        # Feature correlations (if not too many features and not too high-dimensional)
        if 2 <= n_features <= 20:
            try:
                corr_matrix = np.corrcoef(X.T)
                analysis['feature_correlations'] = {
                    'max_correlation': float(np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))),
                    'avg_correlation': float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
                }
            except:
                pass
        
        return analysis

# ═══════════════════════════════════════════════════════════════
# BLOCK 5 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 5/10: Data Handler & Validator - LOADED ✅")
print("MULTI-MODAL SUPPORT: Numerical + Image + Text")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 6/10: PyQt5 Stylesheet & Theme
═══════════════════════════════════════════════════════════════
This block contains:
- Complete PyQt5 stylesheet (QSS)
- Professional purple quantum theme
- Custom widget styles
- Gradient definitions
- Hover and animation effects
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION STYLESHEET
# ═══════════════════════════════════════════════════════════════

def get_main_stylesheet():
    """
    Get the complete application stylesheet.
    
    Returns:
        str: QSS stylesheet string
    """
    
    return f"""
    /* ═══════════════════════════════════════════════════════════
       GLOBAL STYLES
       ═══════════════════════════════════════════════════════════ */
    
    * {{
        font-family: {Fonts.PRIMARY_FAMILY};
        font-size: {Fonts.BODY_SIZE}px;
    }}
    
    QMainWindow {{
        background-color: {Colors.BG_DARKEST};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QWidget {{
        background-color: transparent;
        color: {Colors.TEXT_PRIMARY};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       LABELS
       ═══════════════════════════════════════════════════════════ */
    
    QLabel {{
        background-color: transparent;
        color: {Colors.TEXT_PRIMARY};
        padding: 2px;
    }}
    
    QLabel[heading="true"] {{
        font-size: {Fonts.HEADING2_SIZE}px;
        font-weight: {Fonts.WEIGHT_BOLD};
        color: {Colors.PRIMARY_LIGHT};
    }}
    
    QLabel[subheading="true"] {{
        font-size: {Fonts.HEADING3_SIZE}px;
        font-weight: {Fonts.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_SECONDARY};
    }}
    
    QLabel[muted="true"] {{
        color: {Colors.TEXT_MUTED};
        font-size: {Fonts.SMALL_SIZE}px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════════════════════════ */
    
    QPushButton {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY_DARK}, stop:1 {Colors.PRIMARY}
        );
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: {Fonts.BODY_SIZE}px;
        font-weight: {Fonts.WEIGHT_SEMIBOLD};
        min-height: 35px;
    }}
    
    QPushButton:hover {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_LIGHT}
        );
    }}
    
    QPushButton:pressed {{
        background: {Colors.PRIMARY_DARK};
    }}
    
    QPushButton:disabled {{
        background: {Colors.BG_MEDIUM};
        color: {Colors.TEXT_DISABLED};
    }}
    
    QPushButton[primary="true"] {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_LIGHT}
        );
        font-weight: {Fonts.WEIGHT_BOLD};
        min-height: 40px;
        font-size: {Fonts.HEADING3_SIZE}px;
    }}
    
    QPushButton[secondary="true"] {{
        background: {Colors.BG_MEDIUM};
        border: 2px solid {Colors.PRIMARY};
    }}
    
    QPushButton[danger="true"] {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.ERROR}, stop:1 #dc2626
        );
    }}
    
    QPushButton[success="true"] {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.SUCCESS}, stop:1 #059669
        );
    }}
    
    /* ═══════════════════════════════════════════════════════════
       TEXT INPUT FIELDS
       ═══════════════════════════════════════════════════════════ */
    
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.BG_MEDIUM};
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {Colors.PRIMARY};
    }}
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 2px solid {Colors.PRIMARY};
    }}
    
    QLineEdit:disabled, QTextEdit:disabled {{
        background-color: {Colors.BG_MEDIUM};
        color: {Colors.TEXT_DISABLED};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       SPIN BOXES
       ═══════════════════════════════════════════════════════════ */
    
    QSpinBox, QDoubleSpinBox {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.BG_MEDIUM};
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 30px;
    }}
    
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {Colors.PRIMARY};
    }}
    
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        background-color: {Colors.PRIMARY};
        border-top-right-radius: 4px;
        width: 20px;
    }}
    
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {Colors.PRIMARY_DARK};
        border-bottom-right-radius: 4px;
        width: 20px;
    }}
    
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {Colors.PRIMARY};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       COMBO BOXES (DROPDOWNS)
       ═══════════════════════════════════════════════════════════ */
    
    QComboBox {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.BG_MEDIUM};
        border-radius: 6px;
        padding: 8px 12px;
        min-height: 30px;
    }}
    
    QComboBox:focus {{
        border: 2px solid {Colors.PRIMARY};
    }}
    
    QComboBox::drop-down {{
        border: none;
        background: {Colors.PRIMARY};
        width: 30px;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid white;
        width: 0;
        height: 0;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.PRIMARY};
        selection-background-color: {Colors.PRIMARY};
        selection-color: white;
        outline: none;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       RADIO BUTTONS & CHECKBOXES
       ═══════════════════════════════════════════════════════════ */
    
    QRadioButton, QCheckBox {{
        color: {Colors.TEXT_PRIMARY};
        spacing: 8px;
        padding: 4px;
    }}
    
    QRadioButton::indicator, QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border: 2px solid {Colors.PRIMARY};
        border-radius: 4px;
        background-color: {Colors.BG_DARK};
    }}
    
    QRadioButton::indicator:checked, QCheckBox::indicator:checked {{
        background-color: {Colors.PRIMARY};
        border: 2px solid {Colors.PRIMARY_LIGHT};
    }}
    
    QRadioButton::indicator:hover, QCheckBox::indicator:hover {{
        border: 2px solid {Colors.PRIMARY_LIGHT};
    }}
    
    QRadioButton::indicator {{
        border-radius: 10px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       PROGRESS BARS
       ═══════════════════════════════════════════════════════════ */
    
    QProgressBar {{
        background-color: {Colors.BG_DARK};
        border: 2px solid {Colors.BG_MEDIUM};
        border-radius: 6px;
        text-align: center;
        color: white;
        font-weight: {Fonts.WEIGHT_SEMIBOLD};
        min-height: 25px;
    }}
    
    QProgressBar::chunk {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY_DARK}, 
            stop:0.5 {Colors.PRIMARY}, 
            stop:1 {Colors.PRIMARY_LIGHT}
        );
        border-radius: 4px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       GROUP BOXES (CARDS)
       ═══════════════════════════════════════════════════════════ */
    
    QGroupBox {{
        background-color: {Colors.rgba(Colors.BG_DARK, 0.6)};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 10px;
        margin-top: 12px;
        padding-top: 20px;
        font-weight: {Fonts.WEIGHT_SEMIBOLD};
        font-size: {Fonts.HEADING3_SIZE}px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 15px;
        padding: 0 8px;
        color: {Colors.PRIMARY_LIGHT};
        background-color: {Colors.BG_DARKEST};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       SCROLL AREAS
       ═══════════════════════════════════════════════════════════ */
    
    QScrollArea {{
        background-color: transparent;
        border: none;
    }}
    
    QScrollBar:vertical {{
        background-color: {Colors.BG_DARK};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {Colors.PRIMARY};
        border-radius: 6px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background-color: {Colors.BG_DARK};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background-color: {Colors.PRIMARY};
        border-radius: 6px;
        min-width: 30px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════════════════════ */
    
    QTabWidget::pane {{
        border: 2px solid {Colors.PRIMARY};
        border-radius: 8px;
        background-color: {Colors.BG_DARK};
        top: -2px;
    }}
    
    QTabBar::tab {{
        background-color: {Colors.BG_MEDIUM};
        color: {Colors.TEXT_SECONDARY};
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: {Fonts.WEIGHT_SEMIBOLD};
    }}
    
    QTabBar::tab:selected {{
        background: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_DARK}
        );
        color: white;
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {Colors.HOVER_BG};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       TABLES
       ═══════════════════════════════════════════════════════════ */
    
    QTableWidget {{
        background-color: {Colors.BG_DARK};
        alternate-background-color: {Colors.BG_MEDIUM};
        color: {Colors.TEXT_PRIMARY};
        gridline-color: {Colors.BG_MEDIUM};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 8px;
        selection-background-color: {Colors.PRIMARY};
    }}
    
    QTableWidget::item {{
        padding: 8px;
    }}
    
    QHeaderView::section {{
        background: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 {Colors.PRIMARY_LIGHT}, stop:1 {Colors.PRIMARY}
        );
        color: white;
        padding: 8px;
        border: none;
        font-weight: {Fonts.WEIGHT_BOLD};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       LIST WIDGETS
       ═══════════════════════════════════════════════════════════ */
    
    QListWidget {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 8px;
        padding: 5px;
        outline: none;
    }}
    
    QListWidget::item {{
        padding: 10px;
        border-radius: 4px;
        margin: 2px;
    }}
    
    QListWidget::item:selected {{
        background-color: {Colors.PRIMARY};
        color: white;
    }}
    
    QListWidget::item:hover:!selected {{
        background-color: {Colors.HOVER_BG};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       MENU BAR
       ═══════════════════════════════════════════════════════════ */
    
    QMenuBar {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        padding: 4px;
    }}
    
    QMenuBar::item {{
        padding: 8px 12px;
        background-color: transparent;
        border-radius: 4px;
    }}
    
    QMenuBar::item:selected {{
        background-color: {Colors.PRIMARY};
    }}
    
    QMenu {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 6px;
        padding: 5px;
    }}
    
    QMenu::item {{
        padding: 8px 25px;
        border-radius: 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {Colors.PRIMARY};
    }}
    
    /* ═══════════════════════════════════════════════════════════
       STATUS BAR
       ═══════════════════════════════════════════════════════════ */
    
    QStatusBar {{
        background-color: {Colors.BG_DARK};
        color: {Colors.TEXT_SECONDARY};
        border-top: 2px solid {Colors.PRIMARY};
        padding: 5px;
        font-size: {Fonts.SMALL_SIZE}px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       TOOL TIPS
       ═══════════════════════════════════════════════════════════ */
    
    QToolTip {{
        background-color: {Colors.PRIMARY};
        color: white;
        border: 1px solid {Colors.PRIMARY_LIGHT};
        border-radius: 4px;
        padding: 8px;
        font-size: {Fonts.SMALL_SIZE}px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       DIALOG BOXES
       ═══════════════════════════════════════════════════════════ */
    
    QDialog {{
        background-color: {Colors.BG_DARKEST};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QDialogButtonBox QPushButton {{
        min-width: 80px;
    }}
    
    /* ═══════════════════════════════════════════════════════════
       SPLITTER
       ═══════════════════════════════════════════════════════════ */
    
    QSplitter::handle {{
        background-color: {Colors.PRIMARY};
    }}
    
    QSplitter::handle:horizontal {{
        width: 3px;
    }}
    
    QSplitter::handle:vertical {{
        height: 3px;
    }}
    
    QSplitter::handle:hover {{
        background-color: {Colors.PRIMARY_LIGHT};
    }}
    """

# ═══════════════════════════════════════════════════════════════
# CUSTOM WIDGET STYLES
# ═══════════════════════════════════════════════════════════════

def get_header_stylesheet():
    """
    Stylesheet for application header.
    
    Returns:
        str: QSS for header
    """
    return f"""
    QWidget {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY_DARK}, 
            stop:0.5 {Colors.PRIMARY},
            stop:1 {Colors.PRIMARY_LIGHT}
        );
        border-radius: 10px;
    }}
    
    QLabel {{
        color: white;
        background: transparent;
        font-weight: {Fonts.WEIGHT_BOLD};
    }}
    """

def get_sidebar_stylesheet():
    """
    Stylesheet for navigation sidebar.
    
    Returns:
        str: QSS for sidebar
    """
    return f"""
    QWidget {{
        background-color: {Colors.BG_DARK};
        border-right: 2px solid {Colors.PRIMARY};
    }}
    
    QPushButton {{
        background-color: transparent;
        color: {Colors.TEXT_SECONDARY};
        border: none;
        border-radius: 8px;
        padding: 15px 10px;
        text-align: left;
        font-size: {Fonts.HEADING3_SIZE}px;
        font-weight: {Fonts.WEIGHT_MEDIUM};
    }}
    
    QPushButton:hover {{
        background-color: {Colors.HOVER_BG};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QPushButton:checked {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.PRIMARY_DARK}, stop:1 {Colors.PRIMARY}
        );
        color: white;
        font-weight: {Fonts.WEIGHT_BOLD};
    }}
    """

def get_upload_zone_stylesheet():
    """Stylesheet for drag-and-drop upload zone."""
    return f"""
    DropZoneWidget {{
        background-color: transparent;
        border: 3px dashed {Colors.PRIMARY};
        border-radius: 15px;
    }}
    
    DropZoneWidget:hover {{
        border-color: {Colors.PRIMARY_LIGHT};
        background-color: {Colors.rgba(Colors.PRIMARY, 0.05)};
    }}
    
    DropZoneWidget[active="true"] {{
        border: 3px solid {Colors.PRIMARY_LIGHT};
        background-color: {Colors.rgba(Colors.PRIMARY, 0.15)};
    }}
    
    DropZoneWidget QLabel {{
        border: none;
        background: transparent;
    }}
    
    DropZoneWidget QPushButton {{
        border: none;
    }}
    """

def get_card_stylesheet():
    """
    Stylesheet for content cards.
    
    Returns:
        str: QSS for cards
    """
    return f"""
    QFrame {{
        background-color: {Colors.rgba(Colors.BG_DARK, 0.8)};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 10px;
        padding: 15px;
    }}
    
    QFrame:hover {{
        border: 2px solid {Colors.PRIMARY_LIGHT};
    }}
    """

def get_metric_card_stylesheet(metric_type='default'):
    """
    Stylesheet for metric display cards.
    
    Args:
        metric_type (str): Type of metric (default, success, warning, error)
    
    Returns:
        str: QSS for metric card
    """
    color_map = {
        'default': Colors.PRIMARY,
        'success': Colors.SUCCESS,
        'warning': Colors.WARNING,
        'error': Colors.ERROR,
        'info': Colors.INFO
    }
    
    border_color = color_map.get(metric_type, Colors.PRIMARY)
    
    return f"""
    QFrame {{
        background-color: {Colors.BG_DARK};
        border-left: 5px solid {border_color};
        border-radius: 8px;
        padding: 15px;
    }}
    
    QLabel[value="true"] {{
        font-size: {Fonts.TITLE_SIZE}px;
        font-weight: {Fonts.WEIGHT_BOLD};
        color: {border_color};
    }}
    
    QLabel[label="true"] {{
        font-size: {Fonts.SMALL_SIZE}px;
        color: {Colors.TEXT_MUTED};
        text-transform: uppercase;
    }}
    """

def get_log_viewer_stylesheet():
    """
    Stylesheet for log/console viewer.
    
    Returns:
        str: QSS for log viewer
    """
    return f"""
    QTextEdit {{
        background-color: {Colors.BG_DARKEST};
        color: {Colors.TEXT_PRIMARY};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 8px;
        padding: 10px;
        font-family: {Fonts.MONO_FAMILY};
        font-size: {Fonts.SMALL_SIZE}px;
    }}
    """

def get_quantum_circuit_display_stylesheet():
    """
    Stylesheet for quantum circuit visualization area.
    
    Returns:
        str: QSS for circuit display
    """
    return f"""
    QTextEdit {{
        background-color: {Colors.BG_DARKEST};
        color: {Colors.ACCENT_CYAN};
        border: 2px solid {Colors.PRIMARY};
        border-radius: 8px;
        padding: 15px;
        font-family: {Fonts.MONO_FAMILY};
        font-size: {Fonts.BODY_SIZE}px;
    }}
    """

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR DYNAMIC STYLING
# ═══════════════════════════════════════════════════════════════

def set_button_style(button, style_type='primary'):
    """
    Apply specific style to a button.
    
    Args:
        button (QPushButton): Button widget
        style_type (str): Style type (primary, secondary, danger, success)
    """
    button.setProperty(style_type, True)
    button.style().unpolish(button)
    button.style().polish(button)

def set_label_style(label, style_type='default'):
    """
    Apply specific style to a label.
    
    Args:
        label (QLabel): Label widget
        style_type (str): Style type (heading, subheading, muted)
    """
    if style_type != 'default':
        label.setProperty(style_type, True)
        label.style().unpolish(label)
        label.style().polish(label)

def create_gradient_background(widget, color1, color2, direction='horizontal'):
    """
    Create gradient background for widget.
    
    Args:
        widget (QWidget): Widget to style
        color1 (str): Start color
        color2 (str): End color
        direction (str): 'horizontal' or 'vertical'
    """
    if direction == 'horizontal':
        gradient = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2})"
    else:
        gradient = f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {color1}, stop:1 {color2})"
    
    widget.setStyleSheet(f"background: {gradient}; border-radius: 8px;")

def apply_glow_effect(widget, color=None):
    """
    Apply glow effect to widget (using QGraphicsDropShadowEffect).
    
    Args:
        widget (QWidget): Widget to add glow to
        color (QColor): Glow color (defaults to primary purple)
    """
    from PyQt5.QtWidgets import QGraphicsDropShadowEffect
    
    if color is None:
        color = QColor(Colors.PRIMARY_LIGHT)
    
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setColor(color)
    shadow.setOffset(0, 0)
    widget.setGraphicsEffect(shadow)

def apply_hover_effect(widget):
    """
    Enable hover effects for a widget.
    
    Args:
        widget (QWidget): Widget to enable hover for
    """
    widget.setAttribute(Qt.WA_Hover, True)

# ═══════════════════════════════════════════════════════════════
# ICON AND IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════

def create_colored_icon(text, color, size=24):
    """
    Create a colored text icon.
    
    Args:
        text (str): Icon text (emoji or character)
        color (str): Text color
        size (int): Icon size
        
    Returns:
        QIcon: Icon object
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setPen(QColor(color))
    font = QFont(Fonts.PRIMARY_FAMILY, size // 2)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
    painter.end()
    
    return QIcon(pixmap)

def create_gradient_pixmap(width, height, color1, color2, direction='horizontal'):
    """
    Create a gradient pixmap.
    
    Args:
        width (int): Pixmap width
        height (int): Pixmap height
        color1 (str): Start color
        color2 (str): End color
        direction (str): 'horizontal' or 'vertical'
        
    Returns:
        QPixmap: Gradient pixmap
    """
    pixmap = QPixmap(width, height)
    painter = QPainter(pixmap)
    
    if direction == 'horizontal':
        gradient = QLinearGradient(0, 0, width, 0)
    else:
        gradient = QLinearGradient(0, 0, 0, height)
    
    gradient.setColorAt(0, QColor(color1))
    gradient.setColorAt(1, QColor(color2))
    
    painter.fillRect(0, 0, width, height, gradient)
    painter.end()
    
    return pixmap

# ═══════════════════════════════════════════════════════════════
# QUANTUM BACKGROUND PATTERN GENERATOR
# ═══════════════════════════════════════════════════════════════

def create_quantum_background(width, height):
    """
    Create quantum circuit pattern background.
    
    Args:
        width (int): Background width
        height (int): Background height
        
    Returns:
        QPixmap: Quantum background pattern
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(Colors.BG_DARKEST))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Draw faint circuit lines
    pen = QPen(QColor(Colors.PRIMARY))
    pen.setWidth(1)
    pen.setStyle(Qt.DotLine)
    painter.setPen(pen)
    painter.setOpacity(0.1)
    
    # Horizontal lines (qubit wires)
    spacing = height // 6
    for i in range(1, 6):
        y = i * spacing
        painter.drawLine(0, y, width, y)
    
    # Vertical lines (gate positions)
    v_spacing = width // 10
    for i in range(1, 10):
        x = i * v_spacing
        painter.drawLine(x, spacing, x, height - spacing)
    
    # Draw circuit nodes
    painter.setOpacity(0.15)
    painter.setBrush(QColor(Colors.ACCENT_CYAN))
    painter.setPen(Qt.NoPen)
    
    for i in range(2, 6):
        for j in range(2, 9):
            x = j * v_spacing
            y = i * spacing
            painter.drawEllipse(x - 3, y - 3, 6, 6)
    
    painter.end()
    
    return pixmap

# ═══════════════════════════════════════════════════════════════
# BLOCK 6 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 6/10: PyQt5 Stylesheet & Theme - LOADED ✅")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 7/10: Main Window & Navigation
═══════════════════════════════════════════════════════════════
This block contains:
- Main application window structure
- Sidebar navigation
- Content area management
- Menu bar and status bar
- Window layout and organization
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════

class QuantumFederatedLearningApp(QMainWindow):
    """
    Main application window for Quantum Federated Learning Platform.
    Professional PyQt5 interface with sidebar navigation.
    """
    
    def __init__(self):
        """Initialize main application window"""
        super().__init__()
        
        # Application state
        self.current_dataset = None
        self.federated_server = None
        self.training_thread = None
        self.stop_training_flag = False
        
        # Browser-style navigation history
        self.page_history = [0]
        self.history_index = 0
        
        # Backend manager
        self.backend_manager = QuantumBackendManager()
        
        # Initialize UI
        self.init_ui()
        
        # Apply theme
        self.setStyleSheet(get_main_stylesheet())
        
        # Show startup message
        self.show_startup_status()
        
        log_message("Application initialized successfully", "SUCCESS")
    
    def init_ui(self):
        """Initialize user interface components"""
        
        # Window properties
        self.setWindowTitle(f"{Config.APP_NAME} v{Config.APP_VERSION}")
        self.setMinimumSize(Config.WINDOW_MIN_WIDTH, Config.WINDOW_MIN_HEIGHT)
        
        # Set window size (responsive)
        screen = QApplication.desktop().screenGeometry()
        window_width = int(screen.width() * 0.85)
        window_height = int(screen.height() * 0.85)
        self.resize(window_width, window_height)
        
        # Center window on screen
        self.center_on_screen()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar and content area
        self.create_sidebar()
        self.create_content_area()
        
        # Add to main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content_stack, stretch=1)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create browser-style navigation toolbar
        self.create_nav_toolbar()
        
        # Create status bar
        self.create_status_bar()
        
        # Set default page
        self.show_page(0)
    
    def center_on_screen(self):
        """Center window on screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    def create_sidebar(self):
        """Create navigation sidebar"""
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet(get_sidebar_stylesheet())
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(5)
        
        # Logo/Title area
        logo_frame = self.create_logo_section()
        sidebar_layout.addWidget(logo_frame)
        
        sidebar_layout.addSpacing(20)
        
        # Navigation buttons
        self.nav_buttons = []
        
        nav_items = [
            (Icons.HELP, "User Guide", "Platform overview and instructions"),
            (Icons.QUANTUM, "Quantum Setup", "Configure quantum backend"),
            (Icons.DATASET, "Dataset", "Upload and analyze data"),
            (Icons.CONFIG, "Configuration", "Set training parameters"),
            (Icons.TRAIN, "Training", "Execute federated learning"),
            (Icons.RESULTS, "Results", "View and export results")
        ]
        
        for icon, text, tooltip in nav_items:
            btn = self.create_nav_button(icon, text, tooltip)
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)
        
        # Spacer
        sidebar_layout.addStretch()
        
        # Bottom info
        info_label = QLabel(f"v{Config.APP_VERSION}")
        info_label.setProperty("muted", True)
        info_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(info_label)
    
    def create_logo_section(self):
        """Create logo/title section for sidebar"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.PRIMARY_DARK}, stop:1 {Colors.PRIMARY}
                );
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)
        
        # Icon
        icon_label = QLabel(Icons.QUANTUM)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"font-size: {Fonts.TITLE_SIZE + 10}px; background: transparent;")
        layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel("Quantum FL")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: {Fonts.HEADING2_SIZE}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: white;
            background: transparent;
        """)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Research Platform")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet(f"""
            font-size: {Fonts.SMALL_SIZE}px;
            color: {Colors.TEXT_SECONDARY};
            background: transparent;
        """)
        layout.addWidget(subtitle_label)
        
        return frame
    
    def create_nav_button(self, icon, text, tooltip):
        """
        Create navigation button.
        
        Args:
            icon (str): Icon character
            text (str): Button text
            tooltip (str): Tooltip text
            
        Returns:
            QPushButton: Configured navigation button
        """
        btn = QPushButton(f"{icon}  {text}")
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda: self.show_page(self.nav_buttons.index(btn)))
        return btn
    
    def create_content_area(self):
        """Create main content area with stacked pages"""
        self.content_stack = QStackedWidget()
        
        # Create pages - FIXED: All classes are in the same file, no imports needed
        self.user_guide_page = UserGuideWidget(self)
        self.quantum_setup_page = QuantumSetupWidget(self)
        self.dataset_page = DatasetWidget(self)
        self.config_page = ConfigurationWidget(self)
        self.training_page = TrainingWidget(self)
        self.results_page = ResultsWidget(self)
        
        self.content_stack.addWidget(self.user_guide_page)
        self.content_stack.addWidget(self.quantum_setup_page)
        self.content_stack.addWidget(self.dataset_page)
        self.content_stack.addWidget(self.config_page)
        self.content_stack.addWidget(self.training_page)
        self.content_stack.addWidget(self.results_page)
    
    def show_page(self, index, record_history=True):
        """
        Show specific page and update navigation.
        
        Args:
            index (int): Page index
            record_history (bool): Whether to push to history (False for back/forward)
        """
        # Record in history (truncate forward history on new navigation)
        if record_history:
            self.page_history = self.page_history[:self.history_index + 1]
            if not self.page_history or self.page_history[-1] != index:
                self.page_history.append(index)
                self.history_index = len(self.page_history) - 1
        
        # Update navigation buttons
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        
        # Switch page
        self.content_stack.setCurrentIndex(index)
        
        # Update status bar
        page_names = ["User Guide", "Quantum Setup", "Dataset Management", "Configuration", "Training", "Results"]
        self.status_label.setText(f"Current: {page_names[index]}")
        
        # Update back/forward button states
        if hasattr(self, 'back_btn'):
            self.back_btn.setEnabled(self.history_index > 0)
            self.forward_btn.setEnabled(self.history_index < len(self.page_history) - 1)
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        import_action = QAction("Import Dataset", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self.import_dataset)
        file_menu.addAction(import_action)
        
        export_action = QAction("Export Results", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        refresh_action = QAction("↻  Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.nav_refresh)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        user_guide_action = QAction(f"{Icons.HELP} User Guide", self)
        user_guide_action.triggered.connect(lambda: self.show_page(0))
        view_menu.addAction(user_guide_action)
        
        quantum_setup_action = QAction(f"{Icons.QUANTUM} Quantum Setup", self)
        quantum_setup_action.triggered.connect(lambda: self.show_page(1))
        view_menu.addAction(quantum_setup_action)
        
        dataset_action = QAction(f"{Icons.DATASET} Dataset", self)
        dataset_action.triggered.connect(lambda: self.show_page(2))
        view_menu.addAction(dataset_action)
        
        config_action = QAction(f"{Icons.CONFIG} Configuration", self)
        config_action.triggered.connect(lambda: self.show_page(3))
        view_menu.addAction(config_action)
        
        training_action = QAction(f"{Icons.TRAIN} Training", self)
        training_action.triggered.connect(lambda: self.show_page(4))
        view_menu.addAction(training_action)
        
        results_action = QAction(f"{Icons.RESULTS} Results", self)
        results_action.triggered.connect(lambda: self.show_page(5))
        view_menu.addAction(results_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        documentation_action = QAction("Documentation", self)
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_nav_toolbar(self):
        """Create browser-style back / forward / refresh toolbar"""
        toolbar = self.addToolBar("Navigation")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {Colors.BG_DARK};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
                padding: 4px 8px;
                spacing: 2px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 18px;
                color: {Colors.TEXT_PRIMARY};
                min-width: 32px;
                min-height: 32px;
            }}
            QToolButton:hover {{
                background-color: {Colors.HOVER_BG};
            }}
            QToolButton:pressed {{
                background-color: {Colors.BG_MEDIUM};
            }}
            QToolButton:disabled {{
                color: {Colors.TEXT_DISABLED};
            }}
        """)
        
        # Back button
        self.back_btn = QAction("←", self)
        self.back_btn.setToolTip("Go back  (Alt+Left)")
        self.back_btn.setShortcut("Alt+Left")
        self.back_btn.setEnabled(False)
        self.back_btn.triggered.connect(self.nav_back)
        toolbar.addAction(self.back_btn)
        
        # Forward button
        self.forward_btn = QAction("→", self)
        self.forward_btn.setToolTip("Go forward  (Alt+Right)")
        self.forward_btn.setShortcut("Alt+Right")
        self.forward_btn.setEnabled(False)
        self.forward_btn.triggered.connect(self.nav_forward)
        toolbar.addAction(self.forward_btn)
        
        # Refresh button
        refresh_btn = QAction("↻", self)
        refresh_btn.setToolTip("Go to User Guide  (F5)")
        refresh_btn.setShortcut("F5")
        refresh_btn.triggered.connect(self.nav_refresh)
        toolbar.addAction(refresh_btn)
    
    def nav_back(self):
        """Navigate to previous page in history"""
        if self.history_index > 0:
            self.history_index -= 1
            self.show_page(self.page_history[self.history_index], record_history=False)
    
    def nav_forward(self):
        """Navigate to next page in history"""
        if self.history_index < len(self.page_history) - 1:
            self.history_index += 1
            self.show_page(self.page_history[self.history_index], record_history=False)
    
    def nav_refresh(self):
        """Refresh — navigate back to User Guide (page 0)"""
        self.show_page(0)
    
    def create_status_bar(self):
        """Create application status bar"""
        statusbar = self.statusBar()
        
        # Status label
        self.status_label = QLabel("Ready")
        statusbar.addWidget(self.status_label)
        
        # Spacer
        statusbar.addWidget(QLabel(""), stretch=1)
        
        # Backend indicator
        self.backend_indicator = QLabel()
        self.update_backend_indicator()
        statusbar.addPermanentWidget(self.backend_indicator)
        
        # Quantum status indicator
        self.quantum_status_indicator = QLabel()
        self.update_quantum_status_indicator()
        statusbar.addPermanentWidget(self.quantum_status_indicator)
    
    def update_backend_indicator(self):
        """Update backend status indicator"""
        backend = self.backend_manager.current_backend
        backend_info = self.backend_manager.available_backends[backend]
        
        if backend_info['available']:
            icon = Icons.SUCCESS
            color = Colors.SUCCESS
        else:
            icon = Icons.WARNING
            color = Colors.WARNING
        
        if hasattr(self, 'backend_indicator'):
            self.backend_indicator.setText(f"{icon} Backend: {backend_info['name']}")
            self.backend_indicator.setStyleSheet(f"color: {color}; font-weight: {Fonts.WEIGHT_SEMIBOLD};")
    
    def update_quantum_status_indicator(self):
        """Update quantum availability indicator"""
        if QISKIT_AVAILABLE:
            icon = Icons.SUCCESS
            text = "Quantum: Available"
            color = Colors.SUCCESS
        else:
            icon = Icons.WARNING
            text = "Quantum: Simulation Mode"
            color = Colors.WARNING
        
        self.quantum_status_indicator.setText(f"{icon} {text}")
        self.quantum_status_indicator.setStyleSheet(f"color: {color}; font-weight: {Fonts.WEIGHT_SEMIBOLD};")
    
    def show_startup_status(self):
        """Show startup status in status bar"""
        if QISKIT_AVAILABLE:
            message = "✅ Real Quantum Mode - Qiskit Loaded"
        else:
            message = "⚠️ Simulation Mode - Install Qiskit for Quantum Features"
        
        self.status_label.setText(message)
    
    # ═══════════════════════════════════════════════════════════
    # MENU ACTION HANDLERS
    # ═══════════════════════════════════════════════════════════
    
    def import_dataset(self):
        """Handle dataset import"""
        self.show_page(1)  # Switch to dataset page
        self.dataset_page.trigger_file_dialog()
    
    def export_results(self):
        """Handle results export"""
        if self.federated_server is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No training results available to export.\nPlease complete a training session first."
            )
            return
        
        self.show_page(4)  # Switch to results page
        self.results_page.export_results()
    
    def show_preferences(self):
        """Show preferences dialog"""
        dialog = PreferencesDialog(self)
        dialog.exec_()
    
    def show_documentation(self):
        """Show documentation dialog"""
        dialog = DocumentationDialog(self)
        dialog.exec_()
    
    def show_about(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec_()
    
    # ═══════════════════════════════════════════════════════════
    # PUBLIC METHODS FOR CHILD WIDGETS
    # ═══════════════════════════════════════════════════════════
    
    def update_status(self, message, duration=0):
        """
        Update status bar message.
        
        Args:
            message (str): Status message
            duration (int): Display duration in ms (0 = permanent)
        """
        if duration > 0:
            self.statusBar().showMessage(message, duration)
        else:
            self.status_label.setText(message)
    
    def get_backend_manager(self):
        """Get backend manager instance"""
        return self.backend_manager
    
    def set_dataset(self, X, y, metadata):
        """
        Set current dataset.
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            metadata (dict): Dataset metadata
        """
        self.current_dataset = {
            'X': X,
            'y': y,
            'metadata': metadata
        }
        log_message("Dataset set in main application", "INFO")
    
    def get_dataset(self):
        """Get current dataset"""
        return self.current_dataset
    
    def set_federated_server(self, server):
        """
        Set federated server instance.
        
        Args:
            server (FederatedServer): Server instance
        """
        self.federated_server = server
        log_message("Federated server set in main application", "INFO")
    
    def get_federated_server(self):
        """Get federated server instance"""
        return self.federated_server

# Rest of Block 7 (Dialogs) remains the same...
# ═══════════════════════════════════════════════════════════════
# DIALOG WINDOWS
# ═══════════════════════════════════════════════════════════════

class PreferencesDialog(QDialog):
    """Preferences dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(500, 400)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Application Preferences")
        title.setProperty("heading", True)
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Settings (placeholder for now)
        settings_label = QLabel("Preferences panel - Coming soon")
        settings_label.setProperty("muted", True)
        layout.addWidget(settings_label)
        
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class DocumentationDialog(QDialog):
    """Documentation dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documentation")
        self.setMinimumSize(700, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"{Icons.HELP} Platform Documentation")
        title.setProperty("heading", True)
        layout.addWidget(title)
        
        layout.addSpacing(10)
        
        # Documentation content
        doc_text = QTextEdit()
        doc_text.setReadOnly(True)
        doc_text.setStyleSheet(get_log_viewer_stylesheet())
        
        doc_content = f"""
⚛️ QUANTUM FEDERATED LEARNING PLATFORM - USER GUIDE
{'='*70}

OVERVIEW
{'-'*70}
This platform enables quantum-enhanced federated learning research with
MULTI-MODAL support. Combine quantum circuits with classical neural networks
for distributed machine learning on numerical, image, and text data.

WORKFLOW
{'-'*70}
1. ⚛️ QUANTUM SETUP
   • Select quantum backend (Aer Simulator, IBM Cloud)
   • Configure quantum circuit parameters
   • Test quantum connectivity

2. 📊 DATASET MANAGEMENT (MULTI-MODAL)
   • Upload NUMERICAL data (CSV/Excel)
   • Upload IMAGE data (folders with class subfolders)
   • Upload TEXT data (documents or CSV with text column)
   • Automatic feature extraction and preprocessing
   • View data analysis and AI-suggested configurations

3. 🔧 CONFIGURATION
   • Auto-suggested parameters based on your data type
   • Customize qubits, clients, rounds, and learning rate
   • Choose quantum or classical model for comparison

4. 🔬 TRAINING
   • Execute federated learning across multiple clients
   • Monitor real-time training progress
   • View live accuracy and loss metrics

5. 📈 RESULTS
   • Analyze training results and visualizations
   • Compare quantum vs classical performance
   • Export reports and data (includes data type info)

FEATURES
{'-'*70}
✓ MULTI-MODAL: Numerical + Image + Text data
✓ Automatic feature extraction (CNN for images, BERT for text)
✓ Multi-class classification support (2-100+ classes)
✓ Intelligent configuration suggestions per data type
✓ Real quantum hardware integration (IBM Cloud)
✓ Comprehensive training analytics
✓ Publication-ready result exports

SUPPORTED DATASETS
{'-'*70}

📊 NUMERICAL DATA:
   - Format: CSV, Excel (.xlsx, .xls), JSON
   - Requirements: Labeled data (last column = target)
   - Size: 10+ samples, up to 50 features

🖼️ IMAGE DATA:
   - Format: JPG, PNG, BMP, TIFF
   - Structure: folder/class_name/images.jpg
   - Features: Automatic CNN extraction (ResNet/MobileNet)
   - Requirements: PyTorch installed

📝 TEXT DATA:
   - Format: TXT files or CSV with 'text' column
   - Structure: folder/class_name/docs.txt OR CSV
   - Features: Automatic BERT embeddings
   - Requirements: sentence-transformers installed

QUANTUM BACKENDS
{'-'*70}
- Aer Simulator: Local, fast, ideal for development
- IBM Quantum Cloud: Real quantum hardware (requires API token)

TIPS FOR BEST RESULTS
{'-'*70}
- Start with 3-5 clients for federated learning
- Use 8-15 rounds for most datasets
- Image/text data: Use lower learning rate (0.005-0.008)
- Numerical data: Standard learning rate (0.01)
- Quantum advantage is highest with high-dimensional data
- Compare quantum vs classical for scientific validation

INSTALLATION
{'-'*70}
For full multi-modal support:
   pip install torch torchvision sentence-transformers

For quantum features:
   pip install qiskit qiskit-aer

RESEARCH CITATION
{'-'*70}
{Config.APP_NAME}
Version {Config.APP_VERSION}
Author: {Config.APP_AUTHOR}
Institution: {Config.APP_UNIVERSITY}

Multi-modal quantum federated learning platform supporting
numerical, image, and text data with automatic feature extraction.

For more information, visit the Help menu or contact support.
"""        
        doc_text.setPlainText(doc_content)
        layout.addWidget(doc_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class AboutDialog(QDialog):
    """About dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setFixedSize(520, 560)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)
        
        # Icon
        icon_label = QLabel(Icons.QUANTUM)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 48px; color: {Colors.PRIMARY_LIGHT}; border: none; background: transparent;")
        layout.addWidget(icon_label)
        
        # App name
        name_label = QLabel(Config.APP_NAME)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet(f"""
            font-size: {Fonts.HEADING1_SIZE}px;
            font-weight: {Fonts.WEIGHT_BOLD};
            color: {Colors.PRIMARY_LIGHT};
            border: none;
            background: transparent;
        """)
        layout.addWidget(name_label)
        
        # Version
        version_label = QLabel(f"Version {Config.APP_VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setProperty("muted", True)
        layout.addWidget(version_label)
        
        layout.addSpacing(8)
        
        # Description
        desc_label = QLabel(
            "A professional research platform for exploring\n"
            "quantum-enhanced federated learning with\n"
            "multi-modal data support (Numerical + Image + Text)"
        )
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(desc_label)
        
        layout.addSpacing(8)
        
        # Credits card
        credits_frame = QGroupBox("Platform Information")
        credits_layout = QVBoxLayout(credits_frame)
        credits_layout.setContentsMargins(15, 15, 15, 15)
        credits_layout.setSpacing(8)
        
        info_items = [
            ("Author", Config.APP_AUTHOR),
            ("Institution", Config.APP_UNIVERSITY),
            ("Data Types", "Numerical • Image • Text"),
            ("Quantum Backend", "Qiskit + IBM Cloud"),
            ("Technology Stack", "Python • PyQt5 • PyTorch • Transformers"),
        ]
        
        for label, value in info_items:
            item_layout = QHBoxLayout()
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet(f"font-weight: {Fonts.WEIGHT_SEMIBOLD}; border: none; background: transparent;")
            label_widget.setFixedWidth(130)
            item_layout.addWidget(label_widget)
            
            value_widget = QLabel(value)
            value_widget.setProperty("muted", True)
            value_widget.setStyleSheet("border: none; background: transparent;")
            value_widget.setWordWrap(True)
            item_layout.addWidget(value_widget, stretch=1)
            
            credits_layout.addLayout(item_layout)
        
        layout.addWidget(credits_frame)
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setProperty("primary", True)
        close_btn.setFixedWidth(120)
        close_btn.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

# ═══════════════════════════════════════════════════════════════
# BLOCK 7 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 7/10: Main Window & Navigation - LOADED ✅")
print("="*60)
"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 8/10: Upload & Configuration Widgets
═══════════════════════════════════════════════════════════════
This block contains:
- Quantum Setup Widget (backend selection)
- Dataset Upload Widget (drag-and-drop) - MULTI-MODAL
- Configuration Widget (auto-suggestions)
- Helper widgets and components

ENHANCED VERSION: Supports Numerical, Image & Text Data
═══════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════
# USER GUIDE WIDGET
# ═══════════════════════════════════════════════════════════════

class UserGuideWidget(QWidget):
    """
    User Guide and Getting Started widget.
    First page users see - explains the platform.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)
        
        # Welcome section
        welcome_card = self.create_welcome_card()
        scroll_layout.addWidget(welcome_card)
        
        # Data types section
        data_types_card = self.create_data_types_card()
        scroll_layout.addWidget(data_types_card)
        
        # Workflow section
        workflow_card = self.create_workflow_card()
        scroll_layout.addWidget(workflow_card)
        
        # Benefits section
        benefits_card = self.create_benefits_card()
        scroll_layout.addWidget(benefits_card)
        
        # Quick start button
        quick_start_btn = QPushButton(f"{Icons.PLAY} Start Using Platform")
        quick_start_btn.setProperty("primary", True)
        quick_start_btn.setFixedHeight(50)
        quick_start_btn.clicked.connect(lambda: self.main_window.show_page(1))
        scroll_layout.addWidget(quick_start_btn)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(120)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.HELP} Welcome to Quantum Federated Learning Platform")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Multi-modal AI research platform with quantum enhancement")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        version = QLabel(f"Version {Config.APP_VERSION}")
        version.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.6);")
        layout.addWidget(version)
        
        return header_frame
    
    def create_welcome_card(self):
        """Create welcome section"""
        card = QGroupBox("What is This Platform?")
        layout = QVBoxLayout(card)
        
        intro = QLabel(
            "This platform enables cutting-edge research in <b>quantum-enhanced federated learning</b> "
            "with support for multiple data types. Combine the power of quantum computing with "
            "privacy-preserving federated learning to train models on distributed data."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        
        layout.addSpacing(10)
        
        features = QLabel(
            "✅ <b>Multi-Modal Support:</b> Numerical, Image, and Text data<br>"
            "✅ <b>Quantum Enhancement:</b> Real quantum circuits or simulation<br>"
            "✅ <b>Federated Learning:</b> Privacy-preserving distributed training<br>"
            "✅ <b>Research Ready:</b> Publication-quality results and exports"
        )
        features.setWordWrap(True)
        layout.addWidget(features)
        
        return card
    
    def create_data_types_card(self):
        """Create data types section"""
        card = QGroupBox("Supported Data Types & Requirements")
        layout = QVBoxLayout(card)
        
        # Check capabilities
        caps = CapabilityChecker.get_available_modalities()
        
        # Numerical
        numerical_frame = self.create_data_type_frame(
            Icons.NUMERICAL,
            "Numerical Data (CSV/Excel)",
            caps['numerical']['available'],
            "• Formats: CSV, Excel (.xlsx, .xls), JSON\n"
            "• Structure: Rows = samples, Last column = target/label\n"
            "• Requirements: Min 10 samples, 2-100 classes\n"
            "• Use case: Traditional ML datasets, tabular data",
            "Always available"
        )
        layout.addWidget(numerical_frame)
        
        # Images
        image_frame = self.create_data_type_frame(
            Icons.IMAGE,
            "Image Data (Folders)",
            caps['image']['available'],
            "• Formats: JPG, PNG, BMP, TIFF\n"
            "• Structure: folder/class_name/images.jpg\n"
            "• Processing: Automatic CNN feature extraction (ResNet/MobileNet)\n"
            "• Use case: Image classification, computer vision research",
            caps['image']['reason']
        )
        layout.addWidget(image_frame)
        
        # Text
        text_frame = self.create_data_type_frame(
            Icons.TEXT,
            "Text Data (Documents)",
            caps['text']['available'],
            "• Formats: TXT files or CSV with 'text' column\n"
            "• Structure: folder/class_name/docs.txt OR CSV\n"
            "• Processing: Automatic BERT embeddings\n"
            "• Use case: Sentiment analysis, document classification, NLP",
            caps['text']['reason']
        )
        layout.addWidget(text_frame)
        
        return card
    
    def create_data_type_frame(self, icon, title, available, details, status):
        """Create individual data type info frame"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.rgba(Colors.BG_DARK, 0.5)};
                border-left: 4px solid {Colors.SUCCESS if available else Colors.WARNING};
                border-radius: 6px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(8)
        
        # Title
        title_layout = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        title_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: {Fonts.HEADING3_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD};")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        status_icon = Icons.SUCCESS if available else Icons.WARNING
        status_label = QLabel(f"{status_icon} {status}")
        status_label.setStyleSheet(f"color: {Colors.SUCCESS if available else Colors.WARNING};")
        title_layout.addWidget(status_label)
        
        layout.addLayout(title_layout)
        
        # Details
        details_label = QLabel(details)
        details_label.setWordWrap(True)
        details_label.setProperty("muted", True)
        layout.addWidget(details_label)
        
        return frame
    
    def create_workflow_card(self):
        """Create workflow section"""
        card = QGroupBox("How to Use This Platform (5-Step Workflow)")
        layout = QVBoxLayout(card)
        
        steps = [
            ("1️⃣", "Quantum Setup", "Select your quantum backend (simulation or real hardware)"),
            ("2️⃣", "Upload Dataset", "Upload numerical CSV, image folders, or text documents"),
            ("3️⃣", "Configure Training", "Review AI-suggested parameters or customize"),
            ("4️⃣", "Start Training", "Execute federated learning with live monitoring"),
            ("5️⃣", "Analyze Results", "View performance, compare models, export data")
        ]
        
        for icon, step, description in steps:
            step_frame = QFrame()
            step_layout = QHBoxLayout(step_frame)
            
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 24px;")
            icon_label.setFixedWidth(40)
            step_layout.addWidget(icon_label)
            
            text_layout = QVBoxLayout()
            step_title = QLabel(f"<b>{step}</b>")
            text_layout.addWidget(step_title)
            
            step_desc = QLabel(description)
            step_desc.setProperty("muted", True)
            text_layout.addWidget(step_desc)
            
            step_layout.addLayout(text_layout)
            layout.addWidget(step_frame)
        
        return card
    
    def create_benefits_card(self):
        """Create benefits and target audience section"""
        card = QGroupBox("Who Should Use This Platform?")
        layout = QVBoxLayout(card)
        
        target_audience = QLabel(
            "<b>This platform is ideal for:</b><br><br>"
            "🎓 <b>Researchers:</b> Explore quantum advantage in federated learning<br>"
            "🏫 <b>Students:</b> Learn quantum ML and federated learning concepts<br>"
            "🔬 <b>Data Scientists:</b> Experiment with privacy-preserving ML<br>"
            "🏢 <b>Organizations:</b> Test distributed learning on sensitive data<br><br>"
            "<b>Key Benefits:</b><br><br>"
            "✨ No quantum programming required - automated circuit design<br>"
            "🔒 Privacy-preserving - data never leaves client devices<br>"
            "📊 Multi-modal - works with numbers, images, and text<br>"
            "🚀 Research-grade - publication-ready results<br>"
            "⚡ Flexible - compare quantum vs classical models<br>"
            "🎯 User-friendly - AI suggests optimal configurations"
        )
        target_audience.setWordWrap(True)
        layout.addWidget(target_audience)
        
        return card

# ═══════════════════════════════════════════════════════════════
# QUANTUM SETUP WIDGET
# ═══════════════════════════════════════════════════════════════

class QuantumSetupWidget(QWidget):
    """
    Quantum backend configuration widget.
    Allows selection and testing of quantum backends.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Scrollable area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
    
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)

        # Backend selection card
        backend_card = self.create_backend_selection()
        scroll_layout.addWidget(backend_card)
        
        # IBM Cloud token card
        self.ibm_card = self.create_ibm_token_card()
        scroll_layout.addWidget(self.ibm_card)
        
        # Circuit test card
        test_card = self.create_test_card()
        scroll_layout.addWidget(test_card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
       
        # Initialize backend status
        self.update_backend_display()
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(100)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.QUANTUM} Quantum Backend Configuration")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Select and configure your quantum computing backend")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        return header_frame
    
    def create_backend_selection(self):
        """Create backend selection card"""
        card = QGroupBox("Quantum Backend Selection")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(15)
        
        # Backend options
        self.backend_buttons = QButtonGroup()
        
        backends = [
            ('aer', Icons.SUCCESS + " Aer Simulator (Local)", 
             "Fast local quantum simulation - Ideal for development"),
            ('ibm_cloud', Icons.QUANTUM + " IBM Quantum Cloud", 
             "Real quantum hardware - Requires API token"),
            ('university', Icons.WARNING + " University 1-Qubit Computer", 
             " ")
        ]
        
        for backend_id, label, description in backends:
            radio = QRadioButton(label)
            radio.setProperty('backend_id', backend_id)
            radio.toggled.connect(self.on_backend_changed)
            
            # Enable/disable based on availability
            backend_info = self.main_window.backend_manager.available_backends[backend_id]
            radio.setEnabled(backend_info['available'])
            
            if backend_id == 'aer':
                radio.setChecked(True)
            
            self.backend_buttons.addButton(radio)
            card_layout.addWidget(radio)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setProperty("muted", True)
            desc_label.setContentsMargins(30, 0, 0, 10)
            card_layout.addWidget(desc_label)
        
        # Status display
        self.backend_status_label = QLabel()
        self.backend_status_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold; margin-top: 10px;")
        card_layout.addWidget(self.backend_status_label)
        
        return card
    
    def create_ibm_token_card(self):
        """Create IBM Quantum token configuration card"""
        card = QGroupBox("IBM Quantum Cloud Setup")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(10)
        
        # Info message
        info_label = QLabel(
            f"{Icons.INFO} Save your IBM Quantum API token to enable real quantum hardware access"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.INFO}; padding: 10px; background: {Colors.rgba(Colors.INFO, 0.1)}; border-radius: 6px;")
        card_layout.addWidget(info_label)
        
        # Token input
        token_layout = QHBoxLayout()
        
        token_label = QLabel("API Token:")
        token_label.setFixedWidth(80)
        token_layout.addWidget(token_label)
        
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("Enter your IBM Quantum API token...")
        token_layout.addWidget(self.token_input)
        
        show_btn = QPushButton("Show")
        show_btn.setFixedWidth(60)
        show_btn.clicked.connect(self.toggle_token_visibility)
        token_layout.addWidget(show_btn)
        
        card_layout.addLayout(token_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton(f"{Icons.DOWNLOAD} Save Token")
        save_btn.setProperty("success", True)
        save_btn.clicked.connect(self.save_ibm_token)
        button_layout.addWidget(save_btn)
        
        test_btn = QPushButton(f"{Icons.SEARCH} Test Connection")
        test_btn.setProperty("primary", True)
        test_btn.clicked.connect(self.test_ibm_connection)
        button_layout.addWidget(test_btn)
        
        remove_btn = QPushButton(f"{Icons.DELETE} Remove Token")
        remove_btn.setProperty("danger", True)
        remove_btn.clicked.connect(self.remove_ibm_token)
        button_layout.addWidget(remove_btn)
        
        card_layout.addLayout(button_layout)
        
        # Status
        self.ibm_status_label = QLabel("Status: No token saved")
        self.ibm_status_label.setProperty("muted", True)
        card_layout.addWidget(self.ibm_status_label)
        
        # Help text
        help_text = QLabel(
            "💡 Get your token at: https://quantum.ibm.com/ → Account Settings → API Token"
        )
        help_text.setProperty("muted", True)
        help_text.setWordWrap(True)
        card_layout.addWidget(help_text)
        
        # Initially hidden if not needed
        card.setVisible(False)
        
        return card
    
    def create_test_card(self):
        """Create circuit testing card"""
        card = QGroupBox("Quantum Circuit Testing")
        card_layout = QVBoxLayout(card)
        
        # Test buttons
        button_layout = QHBoxLayout()
        
        status_btn = QPushButton(f"{Icons.REFRESH} Update Status")
        status_btn.clicked.connect(self.update_backend_display)
        button_layout.addWidget(status_btn)
        
        test_btn = QPushButton(f"{Icons.PLAY} Test Circuit")
        test_btn.setProperty("primary", True)
        test_btn.clicked.connect(self.test_quantum_circuit)
        button_layout.addWidget(test_btn)
        
        diagram_btn = QPushButton(f"{Icons.SEARCH} Show Diagram")
        diagram_btn.clicked.connect(self.show_circuit_diagram)
        button_layout.addWidget(diagram_btn)
        
        card_layout.addLayout(button_layout)
        
        # Results display
        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setStyleSheet(get_quantum_circuit_display_stylesheet())
        self.test_results.setMinimumHeight(300)
        card_layout.addWidget(self.test_results)
        
        return card
    
    def on_backend_changed(self, checked):
        """Handle backend selection change"""
        if not checked:
            return
        
        sender = self.sender()
        backend_id = sender.property('backend_id')
        
        # Update backend manager
        self.main_window.backend_manager.set_backend(backend_id)
        
        # Show/hide IBM card
        if hasattr(self, 'ibm_card'):
            self.ibm_card.setVisible(backend_id == 'ibm_cloud')        
        # Update status
        self.update_backend_display()
        self.main_window.update_backend_indicator()
        
        log_message(f"Backend changed to: {backend_id}", "INFO")
    
    def update_backend_display(self):
        """Update backend status display"""
        backend = self.main_window.backend_manager.current_backend
        backend_info = self.main_window.backend_manager.available_backends[backend]
        
        status_text = f"{Icons.SUCCESS} Active: {backend_info['name']}"
        if backend == 'ibm_cloud':
            if self.main_window.backend_manager.check_ibm_token():
                status_text += " (Token configured)"
            else:
                status_text += " (No token - using simulator)"
        
        if hasattr(self, 'backend_status_label'):
            self.backend_status_label.setText(status_text)
        
        # Display backend info in test results
        info_text = f"""

   QUANTUM BACKEND STATUS                                  

   Backend: {backend_info['name']:<45} 
   Type: {backend_info['type']:<48} 
   Available: {'YES' if backend_info['available'] else 'NO':<45} 
                                                           
   {backend_info['description']:<56} 
"""
        if hasattr(self, 'test_results'):
            self.test_results.setPlainText(info_text)
    
    def toggle_token_visibility(self):
        """Toggle API token visibility"""
        if self.token_input.echoMode() == QLineEdit.Password:
            self.token_input.setEchoMode(QLineEdit.Normal)
        else:
            self.token_input.setEchoMode(QLineEdit.Password)
    
    def save_ibm_token(self):
        """Save IBM Quantum token"""
        token = self.token_input.text().strip()
        
        if not token:
            QMessageBox.warning(self, "No Token", "Please enter your IBM Quantum API token!")
            return
        
        if len(token) < 30:
            QMessageBox.warning(
                self, "Invalid Token",
                "This doesn't look like a valid IBM Quantum token.\n"
                "Tokens are typically much longer."
            )
            return
        
        try:
            if not QISKIT_IBM_AVAILABLE:
                QMessageBox.critical(
                    self, "Missing Package",
                    "qiskit-ibm-runtime is not installed!\n\n"
                    "Install it with:\npip install qiskit-ibm-runtime"
                )
                return
            
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            QiskitRuntimeService.save_account(
                channel="ibm_cloud",
                token=token,
                overwrite=True
            )
            
            self.ibm_status_label.setText(f"{Icons.SUCCESS} Token saved successfully!")
            self.ibm_status_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")
            
            QMessageBox.information(
                self, "Success",
                "✅ IBM Quantum API token saved successfully!\n\n"
                "Click 'Test Connection' to verify."
            )
            
            self.token_input.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save token:\n{e}")
            self.ibm_status_label.setText(f"{Icons.ERROR} Error saving token")
            self.ibm_status_label.setStyleSheet(f"color: {Colors.ERROR};")
    
    def test_ibm_connection(self):
        """Test IBM Quantum connection"""
        try:
            if not QISKIT_IBM_AVAILABLE:
                QMessageBox.warning(
                    self, "Package Missing",
                    "qiskit-ibm-runtime is not installed!"
                )
                return
            
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            service = QiskitRuntimeService(channel="ibm_cloud")
            backends = service.backends()
            
            result_text = f"""

   IBM QUANTUM CONNECTION - SUCCESS ✅                     

                                                           
   Available Quantum Backends: {len(backends):<27} 
                                                           
"""
            
            for i, backend in enumerate(backends[:5], 1):
                name = backend.name
                try:
                    qubits = backend.num_qubits
                    result_text += f"   {i}. {name:<50}  \n"
                    result_text += f"      └─ {qubits} qubits{' ' * (45 - len(str(qubits)))} \n"
                except:
                    result_text += f"   {i}. {name:<50}  \n"
            
            if len(backends) > 5:
                result_text += f"   ... and {len(backends) - 5} more backends{' ' * (32 - len(str(len(backends) - 5)))} \n"
            
            result_text += """                                                           
   Status: Ready for quantum computing! 
"""
            
            self.test_results.setPlainText(result_text)
            
            self.ibm_status_label.setText(f"{Icons.SUCCESS} Connected! {len(backends)} backends available")
            self.ibm_status_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")
            
            QMessageBox.information(
                self, "Connection Success",
                f"✅ Connected to IBM Quantum!\n\n"
                f"Available backends: {len(backends)}"
            )
            
        except Exception as e:
            error_msg = f"""

  IBM QUANTUM CONNECTION - FAILED ❌                      
                                                           
   Error: {str(e)[:50]:<50} 
                                                           
   Possible reasons:                                       
   • No token saved                                       
   • Invalid or expired token                             
   • No internet connection                               
   • IBM service unavailable                              
                                                           
   Solution: Get token from https://quantum.ibm.com/      
"""
            self.test_results.setPlainText(error_msg)
            
            self.ibm_status_label.setText(f"{Icons.ERROR} Connection failed")
            self.ibm_status_label.setStyleSheet(f"color: {Colors.ERROR};")
    
    def remove_ibm_token(self):
        """Remove saved IBM token"""
        reply = QMessageBox.question(
            self, "Confirm Removal",
            "Remove saved IBM Quantum token?\n\n"
            "You'll need to re-enter it to use quantum hardware.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        try:
            if not QISKIT_IBM_AVAILABLE:
                return
            
            from qiskit_ibm_runtime import QiskitRuntimeService
            QiskitRuntimeService.delete_account(channel="ibm_cloud")
            
            self.ibm_status_label.setText("Token removed")
            self.ibm_status_label.setProperty("muted", True)
            
            QMessageBox.information(self, "Token Removed", "IBM Quantum token removed.")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to remove token:\n{e}")
    
    def test_quantum_circuit(self):
        """Test quantum circuit creation and execution"""
        try:
            n_qubits = 3
            log_message(f"Testing quantum circuit with {n_qubits} qubits...", "INFO")
            
            # Create test circuit
            use_ibm = (self.main_window.backend_manager.current_backend == 'ibm_cloud' and
                      self.main_window.backend_manager.check_ibm_token())
            
            test_circuit = create_quantum_circuit(n_qubits, use_ibm)
            
            # Test with sample features
            test_features = np.array([0.5, 0.8, 0.3])
            quantum_result = test_circuit.forward(test_features)
            
            # Get circuit info
            circuit_info = test_circuit.get_circuit_info()
            
            mode = "REAL QUANTUM" if QISKIT_AVAILABLE else "SIMULATION"
            
            result_text = f"""
   {mode} CIRCUIT TEST - SUCCESS ✅{' ' * (20 - len(mode))} 

   Circuit Creation: PASSED                                
   Parameter Binding: PASSED                               
   Quantum Execution: PASSED                               
                                                           
   Circuit Details:                                        
      Qubits: {circuit_info['n_qubits']:<46} 
      Parameters: {circuit_info['n_parameters']:<42} 
      Depth: {circuit_info['circuit_depth']:<47} 
      Gates: {circuit_info['gate_count']:<47} 
      Backend: {circuit_info['backend']:<43} 
                                                           
   Test Execution:                                         
      Input: [{test_features[0]:.2f}, {test_features[1]:.2f}, {test_features[2]:.2f}]{' ' * 31} 
      Output: {quantum_result:.6f}{' ' * 39} 
      Range: [-1.0, 1.0]                                  
                                                           
   Verified Features:                                      
     Feature encoding                                     
     Parameterized gates                                  
     Quantum computation                                  
     Measurements                                         
     Gradient computation ready                           
                                                           
   System ready for quantum federated learning!        
"""
            
            self.test_results.setPlainText(result_text)
            
            QMessageBox.information(
                self, "Test Success",
                f"✅ Quantum circuit test completed!\n\n"
                f"Backend: {circuit_info['backend']}\n"
                f"Output: {quantum_result:.6f}"
            )
            
        except Exception as e:
            error_text = f"""

   CIRCUIT TEST FAILED ❌                                  
   Error: {str(e)[:50]:<50} 


Traceback:
{traceback.format_exc()}
"""
            self.test_results.setPlainText(error_text)
            
            QMessageBox.critical(self, "Test Failed", f"Circuit test failed:\n{e}")
    
    def show_circuit_diagram(self):
        """Show quantum circuit diagram"""
        try:
            n_qubits = 3
            test_circuit = create_quantum_circuit(n_qubits, use_ibm_hardware=False)
            diagram = test_circuit.get_circuit_diagram()
            
            self.test_results.setPlainText(diagram)
            
        except Exception as e:
            self.test_results.setPlainText(f"Circuit diagram error:\n{e}")

# ═══════════════════════════════════════════════════════════════
# DATASET WIDGET (ENHANCED FOR MULTI-MODAL)
# ═══════════════════════════════════════════════════════════════

class DatasetWidget(QWidget):
    """
    Dataset upload and analysis widget - MULTI-MODAL.
    Supports numerical CSV, image folders, and text data.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.validator = DatasetValidator()
        self.current_file = None
        self.current_data_type = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Scrollable area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
    
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)

        # Upload zone
        self.upload_zone = self.create_upload_zone()
        scroll_layout.addWidget(self.upload_zone)
        
        # Analysis card
        self.analysis_card = self.create_analysis_card()
        scroll_layout.addWidget(self.analysis_card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(120)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.DATASET} Universal Dataset Management")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Upload numerical CSV, image folders, or text data for quantum federated learning")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        # Capability indicator
        caps = CapabilityChecker.get_available_modalities()
        cap_text = "Supported: "
        cap_icons = []
        if caps['numerical']['available']:
            cap_icons.append(f"{Icons.NUMERICAL} Numerical")
        if caps['image']['available']:
            cap_icons.append(f"{Icons.IMAGE} Images")
        if caps['text']['available']:
            cap_icons.append(f"{Icons.TEXT} Text")
        
        cap_label = QLabel(cap_text + " | ".join(cap_icons))
        cap_label.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.7);")
        layout.addWidget(cap_label)
        
        return header_frame
    
    def create_upload_zone(self):
        """Create drag-and-drop upload zone"""
        zone = DropZoneWidget(self)
        zone.file_dropped.connect(self.handle_file_upload)
        return zone
    
    def create_analysis_card(self):
        """Create dataset analysis display card"""
        card = QGroupBox("Dataset Analysis")
        card_layout = QVBoxLayout(card)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        analyze_btn = QPushButton(f"{Icons.SEARCH} Analyze Dataset")
        analyze_btn.clicked.connect(self.analyze_dataset)
        button_layout.addWidget(analyze_btn)
        
        visualize_btn = QPushButton(f"{Icons.RESULTS} Visualize Data")
        visualize_btn.clicked.connect(self.visualize_dataset)
        button_layout.addWidget(visualize_btn)
        
        next_btn = QPushButton(f"{Icons.PLAY} Configure Training")
        next_btn.setProperty("primary", True)
        next_btn.clicked.connect(self.proceed_to_config)
        button_layout.addWidget(next_btn)
        
        card_layout.addLayout(button_layout)
        
        # Analysis display
        self.analysis_display = QTextEdit()
        self.analysis_display.setReadOnly(True)
        self.analysis_display.setStyleSheet(get_log_viewer_stylesheet())
        self.analysis_display.setMinimumHeight(350) 
        card_layout.addWidget(self.analysis_display)
        
        # Initially show welcome message
        self.show_welcome_message()
        
        return card
    
    def show_welcome_message(self):
        """Show welcome message in analysis display"""
        caps = CapabilityChecker.get_available_modalities()
        
        welcome = f"""

   UNIVERSAL DATASET MANAGEMENT - MULTI-MODAL              

   {Icons.UPLOAD} Upload Your Dataset                                    
   Drag and drop your file/folder or click to browse      
                                                           
"""
        
        # Numerical
        if caps['numerical']['available']:
            welcome += f"""║  {Icons.NUMERICAL} NUMERICAL DATA:                                     
   • CSV (.csv), Excel (.xlsx, .xls), JSON               
   • Last column = target/labels                         
   • Min 10 samples, 2-100 classes                       
                                                           
"""
        
        # Images
        if caps['image']['available']:
            welcome += f"""║  {Icons.IMAGE} IMAGE DATA:                                           
   • Folder with class subfolders                        
   • Structure: folder/class_name/images.jpg             
   • Formats: JPG, PNG, BMP, TIFF                        
   • CNN features auto-extracted                         
                                                           
"""
        else:
            welcome += f"""║  {Icons.WARNING} IMAGE DATA: Not Available                            
   • Install: pip install torch torchvision pillow       
                                                          
"""
        
        # Text
        if caps['text']['available']:
            welcome += f"""║  {Icons.TEXT} TEXT DATA:                                            
   • CSV with 'text' and 'label' columns, OR             
   • Folder with class subfolders of .txt files          
   • BERT embeddings auto-generated                      
                                                           
"""
        else:
            welcome += f"""║  {Icons.WARNING} TEXT DATA: Not Available                             
   • Install: pip install sentence-transformers          
                                                           
"""
        
        welcome += """║  The platform will automatically:                        
   ✓ Detect data type (numerical/image/text)             
   ✓ Extract features (CNN/BERT for image/text)          
   ✓ Validate and preprocess                             
   ✓ Suggest optimal quantum configurations              
                                                           
   Ready to begin? Upload your data!                    

"""
        self.analysis_display.setPlainText(welcome)
    
    def trigger_file_dialog(self):
        """Trigger file selection dialog (called from menu)"""
        self.upload_zone.open_file_dialog()
    
    def handle_file_upload(self, file_path):
        """Handle uploaded file or folder"""
        self.current_file = file_path
        
        # Try to detect data type
        try:
            self.current_data_type = DataTypeDetector.detect_data_type(file_path)
            log_message(f"Detected data type: {self.current_data_type}", "INFO")
        except Exception as e:
            log_message(f"Could not detect data type: {e}", "WARNING")
            self.current_data_type = None
        
        # Auto-analyze
        self.analyze_dataset()
    
    def analyze_dataset(self):
        """Analyze uploaded dataset (multi-modal)"""
        if not self.current_file:
            QMessageBox.warning(self, "No Dataset", "Please upload a dataset first!")
            return
        
        try:
            # Use universal preprocessor
            preprocessor = UniversalPreprocessor()
            
            self.analysis_display.setPlainText("🔍 Detecting data type...\n")
            QApplication.processEvents()
            
            # Preprocess (auto-detects type and extracts features)
            X, y, preprocessing_info = preprocessor.preprocess(self.current_file)
            
            data_type = preprocessing_info['data_type']
            self.current_data_type = data_type
            
            self.analysis_display.append(f"✅ Data type: {data_type.upper()}\n")
            
            if data_type == 'image':
                self.analysis_display.append(f"🖼️  Extracting image features using {preprocessing_info['feature_extractor']}...\n")
            elif data_type == 'text':
                self.analysis_display.append(f"📝 Generating text embeddings...\n")
            
            QApplication.processEvents()
            
            # Analyze
            analyzer = DatasetAnalyzer()
            analysis = analyzer.analyze(X, y)
            
            # Generate suggestions
            suggester = ConfigurationSuggester()
            suggestions = suggester.suggest_configuration(X, y, data_type)
            
            # Store in main window
            self.main_window.set_dataset(X, y, {
                'preprocessing': preprocessing_info,
                'analysis': analysis,
                'suggestions': suggestions,
                'data_type': data_type
            })
            
            # Display analysis
            self.display_analysis(preprocessing_info, analysis, suggestions)
            
            # Show success message
            self.show_success_message(preprocessing_info, analysis)
            
        except UnsupportedDataTypeError as e:
            QMessageBox.critical(
                self, "Unsupported Data Type",
                f"{e}\n\nCheck the welcome message for supported formats."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dataset processing failed:\n{e}")
            traceback.print_exc()
    
    def show_success_message(self, preprocessing_info, analysis):
        """Show success popup based on data type"""
        data_type = preprocessing_info['data_type']
        
        if data_type == 'image':
            QMessageBox.information(
                self, "Images Processed Successfully",
                f"✅ {Icons.IMAGE} Image dataset ready!\n\n"
                f"Total Images: {preprocessing_info['total_images']}\n"
                f"Feature Extractor: {preprocessing_info['feature_extractor'].upper()}\n"
                f"Feature Dimension: {preprocessing_info['feature_dim']}D\n"
                f"Classes: {analysis['basic_stats']['n_classes']}\n"
                f"Class Names: {', '.join(preprocessing_info['class_names'])}\n\n"
                f"Ready to configure quantum training!"
            )
        elif data_type == 'text':
            QMessageBox.information(
                self, "Text Data Processed Successfully",
                f"✅ {Icons.TEXT} Text dataset ready!\n\n"
                f"Total Texts: {preprocessing_info['total_texts']}\n"
                f"Embedding Model: {preprocessing_info['embedding_model'].split('/')[-1]}\n"
                f"Embedding Dimension: {preprocessing_info['embedding_dim']}D\n"
                f"Classes: {analysis['basic_stats']['n_classes']}\n\n"
                f"Ready to configure quantum training!"
            )
        else:  # numerical
            QMessageBox.information(
                self, "Dataset Processed Successfully",
                f"✅ {Icons.NUMERICAL} Numerical dataset ready!\n\n"
                f"Samples: {analysis['basic_stats']['n_samples']}\n"
                f"Features: {analysis['basic_stats']['n_features']}\n"
                f"Classes: {analysis['basic_stats']['n_classes']}\n\n"
                f"Ready to configure quantum training!"
            )
    
    def display_analysis(self, preprocessing_info, analysis, suggestions):
        """Display comprehensive dataset analysis"""
        data_type = preprocessing_info['data_type']
        basic = analysis['basic_stats']
        
        # Header with data type icon
        type_icon = {
            'numerical': Icons.NUMERICAL,
            'image': Icons.IMAGE,
            'text': Icons.TEXT
        }.get(data_type, Icons.DATASET)
        
        output = f"""

   {type_icon} {data_type.upper()} DATASET ANALYSIS - COMPLETE ✅{' ' * (23 - len(data_type))} 


DATA TYPE: {data_type.upper()}
{'-'*60}
"""
        
        # Type-specific info
        if data_type == 'image':
            output += f"""Feature Extractor: {preprocessing_info['feature_extractor'].upper()}
Device: {preprocessing_info.get('device', 'CPU')}
Feature Dimension: {preprocessing_info['feature_dim']}D
Total Images: {preprocessing_info['total_images']}

IMAGES PER CLASS
{'-'*60}
"""
            for class_name, count in preprocessing_info['images_per_class'].items():
                output += f"• {class_name}: {count} images\n"
        
        elif data_type == 'text':
            output += f"""Embedding Model: {preprocessing_info['embedding_model'].split('/')[-1]}
Embedding Dimension: {preprocessing_info['embedding_dim']}D
Total Texts: {preprocessing_info['total_texts']}

"""
            if 'texts_per_class' in preprocessing_info:
                output += f"""TEXTS PER CLASS
{'-'*60}
"""
                for class_name, count in preprocessing_info['texts_per_class'].items():
                    output += f"• {class_name}: {count} texts\n"
        
        else:  # numerical
            output += f"""Total Samples: {basic['n_samples']}
Features: {basic['n_features']}
"""
        
        output += f"""
CLASSIFICATION TASK
{'-'*60}
- Classes: {basic['n_classes']}
- Problem Type: Multi-class Classification

CLASS DISTRIBUTION
{'-'*60}
"""
        
        for class_label, info in analysis['class_distribution'].items():
            class_name = preprocessing_info.get('class_names', [str(class_label)])[class_label] if 'class_names' in preprocessing_info else str(class_label)
            output += f"• Class {class_label} ({class_name}): {info['count']} samples ({info['percentage']:.1f}%)\n"
        
        output += f"""
DATA QUALITY
{'-'*60}
- Balance Ratio: {analysis['data_quality']['balance_ratio']:.3f}
- Status: {'Balanced ✓' if analysis['data_quality']['is_balanced'] else 'Imbalanced ⚠️'}

QUANTUM CONFIGURATION SUGGESTIONS
{'-'*60}
- Recommended Qubits: {suggestions['qubits']}
- Recommended Clients: {suggestions['n_clients']}
- Recommended Rounds: {suggestions['n_rounds']}
- Local Epochs: {suggestions['local_epochs']}
- Learning Rate: {suggestions['learning_rate']}

QUANTUM ADVANTAGE ASSESSMENT
{'-'*60}
- Score: {suggestions['quantum_advantage_score']}/100
- Potential: {'EXCELLENT' if suggestions['quantum_advantage_score'] > 80 else 'GOOD' if suggestions['quantum_advantage_score'] > 60 else 'MODERATE'}
- Data Type: {data_type.upper()} data is {'well-suited' if data_type in ['image', 'text'] else 'suitable'} for quantum enhancement

REASONING
{'-'*60}
"""
        
        for key, reason in suggestions['reasoning'].items():
            output += f"• {key.title()}: {reason}\n"
        
        output += f"""
{'-'*60}
✅ Dataset is ready for quantum federated learning! 
"""
        
        self.analysis_display.setPlainText(output)
    
    def visualize_dataset(self):
        """Visualize dataset (placeholder)"""
        dataset = self.main_window.get_dataset()
        if not self.main_window.get_dataset():
            QMessageBox.warning(self, "No Dataset", "Please analyze a dataset first!")
            return
        try:
            X = dataset['X']
            y = dataset['y']
            data_type = dataset['metadata'].get('data_type', 'numerical')
        
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Dataset Visualization - {data_type.upper()} Data', fontsize=16, fontweight='bold')
        
            # Plot 1: Class distribution
            unique, counts = np.unique(y, return_counts=True)
            axes[0, 0].bar(unique, counts, color=Colors.PRIMARY, alpha=0.7)
            axes[0, 0].set_title('Class Distribution')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
            # Plot 2: Feature correlation (first 2 features)
            if X.shape[1] >= 2:
                axes[0, 1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
                axes[0, 1].set_title('Feature Space (First 2 Features)')
                axes[0, 1].set_xlabel('Feature 1')
                axes[0, 1].set_ylabel('Feature 2')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Not enough features\nfor scatter plot', 
                           ha='center', va='center')
                axes[0, 1].set_title('Feature Space')
        
            # Plot 3: Feature statistics (box plot of first 5 features)
            n_features_to_show = min(5, X.shape[1])
            axes[1, 0].boxplot([X[:, i] for i in range(n_features_to_show)], 
                          labels=[f'F{i+1}' for i in range(n_features_to_show)])
            axes[1, 0].set_title('Feature Distribution (Box Plot)')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
            # Plot 4: Sample count per class (pie chart)
            axes[1, 1].pie(counts, labels=[f'Class {c}' for c in unique], autopct='%1.1f%%',
                      colors=plt.cm.Set3.colors[:len(unique)])
            axes[1, 1].set_title('Class Balance')
        
            plt.tight_layout()
            plt.show()
        
            log_message("Dataset visualization displayed", "SUCCESS")
        
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to visualize dataset:\n{e}")
            traceback.print_exc()
        
    
    def proceed_to_config(self):
        """Proceed to configuration page"""
        if not self.main_window.get_dataset():
            QMessageBox.warning(self, "No Dataset", "Please upload and analyze a dataset first!")
            return
        
        # Switch to configuration page
        self.main_window.show_page(2)

# ═══════════════════════════════════════════════════════════════
# DRAG-AND-DROP UPLOAD ZONE WIDGET 
# ═══════════════════════════════════════════════════════════════

class DropZoneWidget(QFrame):
    """Drag-and-drop file/folder upload widget - MULTI-MODAL"""
    
    file_dropped = pyqtSignal(str)  # Emits file or folder path
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet(get_upload_zone_stylesheet())
        self.setFixedHeight(320)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Icon
        icon_label = QLabel(Icons.UPLOAD)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            font-size: 48px; 
            background: transparent; 
            color: {Colors.PRIMARY_LIGHT};
        """)
        layout.addWidget(icon_label)
        
        # Main text
        main_label = QLabel("Drag & Drop File or Folder Here")
        main_label.setAlignment(Qt.AlignCenter)
        main_label.setStyleSheet(f"""
            font-size: {Fonts.HEADING2_SIZE}px; 
            font-weight: {Fonts.WEIGHT_BOLD}; 
            background: transparent;
            color: {Colors.TEXT_PRIMARY};             
        """)
        layout.addWidget(main_label)

        # Browse button
        browse_btn = QPushButton(f"{Icons.SEARCH} Browse Files/Folders")
        browse_btn.setProperty("primary", True)
        browse_btn.setFixedSize(240, 45)
        browse_btn.clicked.connect(self.open_file_dialog)
        browse_btn.setCursor(QCursor(Qt.PointingHandCursor))
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)
        
        # Format hint
        format_label = QLabel("Supports: CSV • Excel • Image Folders • Text Files")
        format_label.setAlignment(Qt.AlignCenter)
        format_label.setStyleSheet(f"""
            font-size: {Fonts.SMALL_SIZE}px;
            color: {Colors.TEXT_MUTED};
            background: transparent;
            margin-top: 10px;
        """)
        layout.addWidget(format_label)
    
    def dragEnterEvent(self, event):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.accept()
            self.setProperty("active", True)
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.setProperty("active", False)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def dropEvent(self, event):
        """Handle file/folder drop"""
        self.setProperty("active", False)
        self.style().unpolish(self)
        self.style().polish(self)
        
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_dropped.emit(files[0])
    
    def open_file_dialog(self):
        """Open file/folder selection dialog with data type choice"""
        # Ask user what type of data
        dialog = DataTypeSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data_type = dialog.get_selected_type()
            
            if data_type == 'numerical':
                # File dialog for CSV/Excel
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Numerical Dataset",
                    "",
                    "Data Files (*.csv *.xlsx *.xls *.json);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*.*)"
                )
                if file_path:
                    self.file_dropped.emit(file_path)
            
            elif data_type == 'image':
                # Folder dialog for images
                folder_path = QFileDialog.getExistingDirectory(
                    self,
                    "Select Image Folder (with class subfolders)",
                    "",
                    QFileDialog.ShowDirsOnly
                )
                if folder_path:
                    self.file_dropped.emit(folder_path)
            
            elif data_type == 'text':
                # Choice between CSV or folder
                msg = QMessageBox()
                msg.setWindowTitle("Text Data Format")
                msg.setText("How is your text data organized?")
                msg.setIcon(QMessageBox.Question)
                
                csv_btn = msg.addButton("CSV File", QMessageBox.YesRole)
                folder_btn = msg.addButton("Folder of .txt Files", QMessageBox.NoRole)
                msg.addButton(QMessageBox.Cancel)
                
                msg.exec_()
                
                if msg.clickedButton() == csv_btn:
                    file_path, _ = QFileDialog.getOpenFileName(
                        self, "Select Text CSV", "", "CSV Files (*.csv)"
                    )
                    if file_path:
                        self.file_dropped.emit(file_path)
                elif msg.clickedButton() == folder_btn:
                    folder_path = QFileDialog.getExistingDirectory(
                        self, "Select Text Folder (with class subfolders)"
                    )
                    if folder_path:
                        self.file_dropped.emit(folder_path)

# ═══════════════════════════════════════════════════════════════
# DATA TYPE SELECTION DIALOG 
# ═══════════════════════════════════════════════════════════════

class DataTypeSelectionDialog(QDialog):
    """Dialog for selecting data type before upload"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Type")
        self.setFixedSize(450, 420)
        self.selected_type = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("What type of data are you uploading?")
        title.setStyleSheet(f"font-size: {Fonts.HEADING2_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; border: none; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        layout.addSpacing(15)
        
        # Button group
        self.button_group = QButtonGroup()
        
        # Check capabilities
        caps = CapabilityChecker.get_available_modalities()
        
        # Numerical option
        num_radio = QRadioButton(f"{Icons.NUMERICAL}  Numerical Data (CSV/Excel)")
        num_radio.setProperty('data_type', 'numerical')
        num_radio.setChecked(True)
        num_radio.setStyleSheet("border: none; background: transparent;")
        self.button_group.addButton(num_radio)
        layout.addWidget(num_radio)
        
        num_desc = QLabel("     Tabular data with features and labels")
        num_desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SMALL_SIZE}px; border: none; background: transparent;")
        layout.addWidget(num_desc)
        
        layout.addSpacing(10)
        
        # Image option
        img_radio = QRadioButton(f"{Icons.IMAGE}  Images (Folder of images)")
        img_radio.setProperty('data_type', 'image')
        img_radio.setEnabled(caps['image']['available'])
        img_radio.setStyleSheet("border: none; background: transparent;")
        self.button_group.addButton(img_radio)
        layout.addWidget(img_radio)
        
        img_desc = QLabel(f"     {caps['image']['reason']}")
        if not caps['image']['available']:
            img_desc.setStyleSheet(f"color: {Colors.WARNING}; border: none; background: transparent;")
        else:
            img_desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SMALL_SIZE}px; border: none; background: transparent;")
        layout.addWidget(img_desc)
        
        layout.addSpacing(10)
        
        # Text option
        text_radio = QRadioButton(f"{Icons.TEXT}  Text (Documents/Reviews)")
        text_radio.setProperty('data_type', 'text')
        text_radio.setEnabled(caps['text']['available'])
        text_radio.setStyleSheet("border: none; background: transparent;")
        self.button_group.addButton(text_radio)
        layout.addWidget(text_radio)
        
        text_desc = QLabel(f"     {caps['text']['reason']}")
        if not caps['text']['available']:
            text_desc.setStyleSheet(f"color: {Colors.WARNING}; border: none; background: transparent;")
        else:
            text_desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SMALL_SIZE}px; border: none; background: transparent;")
        layout.addWidget(text_desc)
        
        layout.addStretch()
        
        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.setProperty("primary", True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def accept(self):
        """Handle OK button"""
        for button in self.button_group.buttons():
            if button.isChecked():
                self.selected_type = button.property('data_type')
                break
        super().accept()
    
    def get_selected_type(self):
        """Get selected data type"""
        return self.selected_type

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION WIDGET 
# ═══════════════════════════════════════════════════════════════

class ConfigurationWidget(QWidget):
    """
    Training configuration widget with auto-suggestions.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Scroll area for configuration options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)
        
        # Configuration cards
        self.create_config_cards(scroll_layout)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Start training button
        start_btn = QPushButton(f"{Icons.PLAY} Start Training")
        start_btn.setProperty("primary", True)
        start_btn.setFixedHeight(50)
        start_btn.clicked.connect(self.start_training)
        layout.addWidget(start_btn)
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(100)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.CONFIG} Training Configuration")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Configure quantum and federated learning parameters")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        return header_frame
    
    def create_config_cards(self, layout):
        """Create configuration input cards"""
        
        # Model selection card
        model_card = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_card)
        
        self.model_type_group = QButtonGroup()
        
        quantum_radio = QRadioButton(f"{Icons.QUANTUM} Quantum-Classical Hybrid")
        quantum_radio.setProperty('model_type', 'quantum')
        quantum_radio.setChecked(True)
        self.model_type_group.addButton(quantum_radio)
        model_layout.addWidget(quantum_radio)
        
        quantum_desc = QLabel("Uses quantum circuits + classical NN for enhanced performance")
        quantum_desc.setProperty("muted", True)
        quantum_desc.setContentsMargins(30, 0, 0, 10)
        model_layout.addWidget(quantum_desc)
        
        classical_radio = QRadioButton(f"🖥️ Pure Classical Baseline")
        classical_radio.setProperty('model_type', 'classical')
        self.model_type_group.addButton(classical_radio)
        model_layout.addWidget(classical_radio)
        
        classical_desc = QLabel("Pure neural network for comparison")
        classical_desc.setProperty("muted", True)
        classical_desc.setContentsMargins(30, 0, 0, 10)
        model_layout.addWidget(classical_desc)
        
        layout.addWidget(model_card)
        
        # Quantum parameters card
        quantum_card = QGroupBox("Quantum Circuit Parameters")
        quantum_grid = QGridLayout(quantum_card)
        
        # Qubits
        quantum_grid.addWidget(QLabel("Number of Qubits:"), 0, 0)
        self.qubits_spin = QSpinBox()
        self.qubits_spin.setRange(Config.MIN_QUBITS, Config.MAX_QUBITS)
        self.qubits_spin.setValue(Config.DEFAULT_QUBITS)
        quantum_grid.addWidget(self.qubits_spin, 0, 1)
        self.qubits_hint = QLabel()
        self.qubits_hint.setProperty("muted", True)
        quantum_grid.addWidget(self.qubits_hint, 0, 2)
        
        layout.addWidget(quantum_card)
        
        # Federated learning card
        federated_card = QGroupBox("Federated Learning Parameters")
        federated_grid = QGridLayout(federated_card)
        
        # Clients
        federated_grid.addWidget(QLabel("Number of Clients:"), 0, 0)
        self.clients_spin = QSpinBox()
        self.clients_spin.setRange(Config.MIN_CLIENTS, Config.MAX_CLIENTS)
        self.clients_spin.setValue(Config.DEFAULT_CLIENTS)
        federated_grid.addWidget(self.clients_spin, 0, 1)
        self.clients_hint = QLabel()
        self.clients_hint.setProperty("muted", True)
        federated_grid.addWidget(self.clients_hint, 0, 2)
        
        # Rounds
        federated_grid.addWidget(QLabel("Federated Rounds:"), 1, 0)
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(Config.MIN_ROUNDS, Config.MAX_ROUNDS)
        self.rounds_spin.setValue(Config.DEFAULT_ROUNDS)
        federated_grid.addWidget(self.rounds_spin, 1, 1)
        self.rounds_hint = QLabel()
        self.rounds_hint.setProperty("muted", True)
        federated_grid.addWidget(self.rounds_hint, 1, 2)
        
        # Local epochs
        federated_grid.addWidget(QLabel("Local Epochs:"), 2, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(Config.MIN_LOCAL_EPOCHS, Config.MAX_LOCAL_EPOCHS)
        self.epochs_spin.setValue(Config.DEFAULT_LOCAL_EPOCHS)
        federated_grid.addWidget(self.epochs_spin, 2, 1)
        self.epochs_hint = QLabel()
        self.epochs_hint.setProperty("muted", True)
        federated_grid.addWidget(self.epochs_hint, 2, 2)
        
        # Learning rate
        federated_grid.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(Config.DEFAULT_LEARNING_RATE)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.01)
        federated_grid.addWidget(self.lr_spin, 3, 1)
        self.lr_hint = QLabel()
        self.lr_hint.setProperty("muted", True)
        federated_grid.addWidget(self.lr_hint, 3, 2)
        
        layout.addWidget(federated_card)
        
        # Auto-load suggestions if dataset exists
        self.load_suggestions()
    
    def load_suggestions(self):
        """Load auto-suggested configurations"""
        dataset = self.main_window.get_dataset()
        if not dataset:
            return
        
        suggestions = dataset['metadata']['suggestions']
        
        self.qubits_spin.setValue(suggestions['qubits'])
        self.qubits_hint.setText(suggestions['reasoning']['qubits'])
        
        self.clients_spin.setValue(suggestions['n_clients'])
        self.clients_hint.setText(suggestions['reasoning']['clients'])
        
        self.rounds_spin.setValue(suggestions['n_rounds'])
        self.rounds_hint.setText(suggestions['reasoning']['rounds'])
        
        self.epochs_spin.setValue(suggestions['local_epochs'])
        self.epochs_hint.setText(suggestions['reasoning']['epochs'])
        
        self.lr_spin.setValue(suggestions['learning_rate'])
        self.lr_hint.setText(suggestions['reasoning']['learning_rate'])
    
    def start_training(self):
        """Start training with current configuration"""
        dataset = self.main_window.get_dataset()
        if not dataset:
            QMessageBox.warning(
                self, "No Dataset",
                "Please upload and analyze a dataset first!"
            )
            return
        
        # Switch to training page
        self.main_window.show_page(3)
        
        # Start training on training page
        self.main_window.training_page.execute_training(self.get_config())
    
    def get_config(self):
        """Get current configuration"""
        model_type = 'quantum'
        for button in self.model_type_group.buttons():
            if button.isChecked():
                model_type = button.property('model_type')
                break
        
        return {
            'model_type': model_type,
            'n_qubits': self.qubits_spin.value(),
            'n_clients': self.clients_spin.value(),
            'n_rounds': self.rounds_spin.value(),
            'local_epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value()
        }

# ═══════════════════════════════════════════════════════════════
# BLOCK 8 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 8/10: Upload & Configuration Widgets - LOADED ✅")
print("MULTI-MODAL SUPPORT: Numerical + Image + Text Upload")
print("="*60)

"""
═══════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 9/10: Training & Results Widgets
═══════════════════════════════════════════════════════════════
This block contains:
- Training execution widget with live monitoring
- Results visualization and analysis widget
- Progress tracking and metrics display
- Export functionality

ENHANCED VERSION: Shows data type in results
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# TRAINING THREAD (Background Processing) 
# ═══════════════════════════════════════════════════════════════

class TrainingThread(QThread):
    """
    Background thread for quantum federated training.
    Emits signals for progress updates.
    """
    
    # Signals
    progress_update = pyqtSignal(int, dict)  # (progress_percent, metrics)
    round_complete = pyqtSignal(int, dict)   # (round_number, results)
    training_complete = pyqtSignal(dict)     # (final_results)
    error_occurred = pyqtSignal(str)         # (error_message)
    log_message = pyqtSignal(str)            # (log_text)
    
    def __init__(self, config, dataset, backend_manager):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.backend_manager = backend_manager
        self.stop_flag = False
    
    def run(self):
        """Execute training in background thread"""
        try:
            # Get data type for logging
            data_type = self.dataset['metadata'].get('data_type', 'numerical')
            type_icon = {
                'numerical': Icons.NUMERICAL,
                'image': Icons.IMAGE,
                'text': Icons.TEXT
            }.get(data_type, Icons.DATASET)
            
            self.log_message.emit(f"{type_icon} Initializing quantum federated learning for {data_type.upper()} data...")
            
            X = self.dataset['X']
            y = self.dataset['y']
            n_classes = len(np.unique(y))
            
            # Create federated server
            self.log_message.emit(f"Creating {self.config['model_type'].upper()} federated server...")
            
            use_quantum = (self.config['model_type'] == 'quantum')
            use_ibm = (self.backend_manager.current_backend == 'ibm_cloud' and 
                      self.backend_manager.check_ibm_token())
            
            server = FederatedServer(
                model_type=self.config['model_type'],
                input_dim=X.shape[1],
                n_classes=n_classes,
                n_qubits=self.config['n_qubits'],
                use_ibm=use_ibm
            )
            
            # Distribute data to clients
            self.log_message.emit(f"Distributing data to {self.config['n_clients']} clients...")
            
            distributor = DataDistributor()
            client_data = distributor.distribute_iid(X, y, self.config['n_clients'])
            
            # Create clients
            self.log_message.emit("Creating federated clients...")
            
            for i, (X_client, y_client) in enumerate(client_data):
                model = create_model(
                    self.config['model_type'],
                    X.shape[1],
                    n_classes,
                    self.config['n_qubits'],
                    use_ibm
                )
                
                client = FederatedClient(
                    client_id=i,
                    model=model,
                    data=(X_client, y_client),
                    client_name=f"Client_{i+1}"
                )
                server.add_client(client)
            
            self.log_message.emit(f"Starting {self.config['n_rounds']} federated rounds...")
            
            # Training loop
            start_time = time.time()
            
            for round_num in range(1, self.config['n_rounds'] + 1):
                if self.stop_flag:
                    self.log_message.emit("Training stopped by user")
                    break
                
                self.log_message.emit(f"\n{'='*60}")
                self.log_message.emit(f"FEDERATED ROUND {round_num}/{self.config['n_rounds']}")
                self.log_message.emit(f"{'='*60}")
                
                # Execute federated round
                round_results = server.federated_round(
                    round_number=round_num,
                    local_epochs=self.config['local_epochs'],
                    learning_rate=self.config['learning_rate'],
                    X_test=X,
                    y_test=y
                )
                
                # Calculate progress
                progress = int((round_num / self.config['n_rounds']) * 100)
                
                # Emit progress update
                metrics = {
                    'round': round_num,
                    'total_rounds': self.config['n_rounds'],
                    'global_accuracy': round_results['global_accuracy'],
                    'avg_client_accuracy': round_results['avg_client_accuracy'],
                    'avg_client_loss': round_results['avg_client_loss'],
                    'elapsed_time': time.time() - start_time
                }
                
                self.progress_update.emit(progress, metrics)
                self.round_complete.emit(round_num, round_results)
                
                self.log_message.emit(
                    f"Round {round_num} Summary: "
                    f"Global Acc={round_results['global_accuracy']:.4f}, "
                    f"Avg Client Acc={round_results['avg_client_accuracy']:.4f}"
                )
            
            # Training complete
            total_time = time.time() - start_time
            
            final_results = {
                'server': server,
                'training_history': server.training_history,
                'total_time': total_time,
                'config': self.config,
                'data_type': data_type,  # Include data type
                'final_accuracy': server.training_history['global_accuracies'][-1] if server.training_history['global_accuracies'] else 0,
                'best_accuracy': max(server.training_history['global_accuracies']) if server.training_history['global_accuracies'] else 0
            }
            
            self.log_message.emit(f"\n{'='*60}")
            self.log_message.emit(f"{type_icon} TRAINING COMPLETED! ✅")
            self.log_message.emit(f"{'='*60}")
            self.log_message.emit(f"Final Accuracy: {final_results['final_accuracy']:.4f}")
            self.log_message.emit(f"Best Accuracy: {final_results['best_accuracy']:.4f}")
            self.log_message.emit(f"Total Time: {format_time(total_time)}")
            
            self.training_complete.emit(final_results)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def stop(self):
        """Stop training"""
        self.stop_flag = True

# ═══════════════════════════════════════════════════════════════
# TRAINING WIDGET 
# ═══════════════════════════════════════════════════════════════

class TrainingWidget(QWidget):
    """
    Training execution and monitoring widget.
    Displays live progress and metrics.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.training_thread = None
        self.training_start_time = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Scrollable area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
    
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)
        
        # Control buttons
        control_frame = self.create_control_buttons()
        scroll_layout.addWidget(control_frame)
        
        # Progress section
        progress_card = self.create_progress_card()
        scroll_layout.addWidget(progress_card)
        
        # Metrics display
        metrics_card = self.create_metrics_card()
        scroll_layout.addWidget(metrics_card)
        
        # Training log
        log_card = self.create_log_card()
        scroll_layout.addWidget(log_card, stretch=1)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(100)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.TRAIN} Quantum Federated Training")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Execute and monitor quantum federated learning")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        return header_frame
    
    def create_control_buttons(self):
        """Create control buttons"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        self.start_btn = QPushButton(f"{Icons.PLAY} Start Training")
        self.start_btn.setProperty("primary", True)
        self.start_btn.setFixedHeight(50)
        self.start_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton(f"{Icons.STOP} Stop Training")
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setFixedHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_btn)
        
        return frame
    
    def create_progress_card(self):
        """Create progress display card"""
        card = QGroupBox("Training Progress")
        card_layout = QVBoxLayout(card)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready to start training")
        self.progress_bar.setFixedHeight(35)
        card_layout.addWidget(self.progress_bar)
        
        # Status label
        self.progress_label = QLabel("Waiting to begin...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet(f"font-size: {Fonts.HEADING3_SIZE}px; color: {Colors.PRIMARY_LIGHT};")
        card_layout.addWidget(self.progress_label)
        
        return card
    
    def create_metrics_card(self):
        """Create real-time metrics display"""
        card = QGroupBox("Live Training Metrics")
        card_layout = QGridLayout(card)
        card_layout.setSpacing(15)
        
        # Create metric displays
        self.metric_widgets = {}
        
        metrics = [
            ('current_round', 'Current Round', '0/0', Colors.INFO),
            ('global_accuracy', 'Global Accuracy', '0.000', Colors.SUCCESS),
            ('avg_client_accuracy', 'Avg Client Accuracy', '0.000', Colors.PRIMARY_LIGHT),
            ('training_time', 'Training Time', '00:00', Colors.WARNING)
        ]
        
        for i, (key, label, default_value, color) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2
            
            metric_frame = self.create_metric_display(label, default_value, color)
            self.metric_widgets[key] = metric_frame['value_label']
            card_layout.addWidget(metric_frame['frame'], row, col, 1, 2)
        
        return card
    
    def create_metric_display(self, label, value, color):
        """Create individual metric display"""
        frame = QFrame()
        frame.setStyleSheet(get_metric_card_stylesheet('default'))
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)
        
        label_widget = QLabel(label.upper())
        label_widget.setProperty("label", True)
        layout.addWidget(label_widget)
        
        value_label = QLabel(value)
        value_label.setProperty("value", True)
        value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(value_label)
        
        return {'frame': frame, 'value_label': value_label}
    
    def create_log_card(self):
        """Create training log display"""
        card = QGroupBox("Training Log")
        card_layout = QVBoxLayout(card)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setStyleSheet(get_log_viewer_stylesheet())
        card_layout.addWidget(self.training_log)
        
        # Show welcome message
        self.show_ready_message()
        
        return card
    
    def show_ready_message(self):
        """Show ready message in log"""
        message = f"""

   QUANTUM FEDERATED TRAINING - READY                      

                                                          
   System initialized and ready to begin training.         
                                                           
   Prerequisites:                                          
     Quantum backend configured                           
     Dataset uploaded and preprocessed                    
     Training parameters configured                       
                                                           
   Click 'Start Training' to begin!                     

"""
        self.training_log.setPlainText(message)
    
    def start_training(self):
        """Start training with configuration from config page"""
        # Get configuration from config page
        config = self.main_window.config_page.get_config()
        self.execute_training(config)
    
    def execute_training(self, config):
        """Execute training with given configuration"""
        # Validate prerequisites
        dataset = self.main_window.get_dataset()
        if not dataset:
            QMessageBox.warning(
                self, "No Dataset",
                "Please upload and analyze a dataset first!"
            )
            return
        
        # Reset UI
        self.training_log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing...")
        
        # Disable start button, enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create and start training thread
        self.training_start_time = time.time()
        
        self.training_thread = TrainingThread(
            config=config,
            dataset=dataset,
            backend_manager=self.main_window.backend_manager
        )
        
        # Connect signals
        self.training_thread.progress_update.connect(self.on_progress_update)
        self.training_thread.round_complete.connect(self.on_round_complete)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.error_occurred.connect(self.on_training_error)
        self.training_thread.log_message.connect(self.append_log)
        
        # Start training
        self.training_thread.start()
        
        log_message("Training started in background thread", "INFO")
    
    def stop_training(self):
        """Stop ongoing training"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "Stop Training",
                "Are you sure you want to stop training?\n\n"
                "Progress will be lost.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.append_log("Stopping training...")
                self.training_thread.stop()
                self.training_thread.wait()
                self.reset_training_ui()
    
    def on_progress_update(self, progress, metrics):
        """Handle progress update from training thread"""
        # Update progress bar
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"Round {metrics['round']}/{metrics['total_rounds']} - {progress}%")
        
        # Update progress label
        self.progress_label.setText(
            f"Training Round {metrics['round']} of {metrics['total_rounds']}"
        )
        
        # Update metric displays
        self.metric_widgets['current_round'].setText(
            f"{metrics['round']}/{metrics['total_rounds']}"
        )
        self.metric_widgets['global_accuracy'].setText(
            f"{metrics['global_accuracy']:.3f}"
        )
        self.metric_widgets['avg_client_accuracy'].setText(
            f"{metrics['avg_client_accuracy']:.3f}"
        )
        self.metric_widgets['training_time'].setText(
            format_time(metrics['elapsed_time'])
        )
        
        # Update main window status
        self.main_window.update_status(
            f"Training: Round {metrics['round']}/{metrics['total_rounds']} - "
            f"Accuracy: {metrics['global_accuracy']:.3f}"
        )
    
    def on_round_complete(self, round_num, results):
        """Handle round completion"""
        # This can be used for additional per-round processing if needed
        pass
    
    def on_training_complete(self, results):
        """Handle training completion"""
        # Update progress
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Training Complete! ✅")
        
        self.progress_label.setText(
            f"Training completed successfully! Final Accuracy: {results['final_accuracy']:.4f}"
        )
        self.progress_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: {Fonts.HEADING3_SIZE}px; font-weight: bold;")
        
        # Store results in main window (including data type)
        self.main_window.set_federated_server(results['server'])
        
        # Store data type for results page
        if 'data_type' in results:
            dataset = self.main_window.get_dataset()
            if dataset and 'metadata' in dataset:
                dataset['metadata']['data_type'] = results['data_type']
        
        # Update status
        self.main_window.update_status(
            f"Training Complete - Final Accuracy: {results['final_accuracy']:.4f}"
        )
        
        # Reset UI
        self.reset_training_ui()
        
        # Show completion message
        data_type_icon = {
            'numerical': Icons.NUMERICAL,
            'image': Icons.IMAGE,
            'text': Icons.TEXT
        }.get(results.get('data_type', 'numerical'), Icons.DATASET)
        
        QMessageBox.information(
            self, "Training Complete",
            f"✅ {data_type_icon} Quantum federated training completed!\n\n"
            f"Data Type: {results.get('data_type', 'numerical').upper()}\n"
            f"Final Accuracy: {results['final_accuracy']:.4f}\n"
            f"Best Accuracy: {results['best_accuracy']:.4f}\n"
            f"Total Time: {format_time(results['total_time'])}\n\n"
            f"View results in the Results tab!"
        )
        
        # Automatically switch to results page
        self.main_window.show_page(4)
    
    def on_training_error(self, error_message):
        """Handle training error"""
        self.append_log(f"\n❌ ERROR:\n{error_message}")
        
        self.progress_bar.setFormat("Training Failed ❌")
        self.progress_label.setText("Training failed - see log for details")
        self.progress_label.setStyleSheet(f"color: {Colors.ERROR};")
        
        self.reset_training_ui()
        
        QMessageBox.critical(
            self, "Training Error",
            f"Training failed with error:\n\n{error_message[:200]}...\n\n"
            f"See training log for full details."
        )
    
    def append_log(self, message):
        """Append message to training log"""
        self.training_log.append(message)
        self.training_log.verticalScrollBar().setValue(
            self.training_log.verticalScrollBar().maximum()
        )
    
    def reset_training_ui(self):
        """Reset UI to ready state"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

# ═══════════════════════════════════════════════════════════════
# RESULTS WIDGET 
# ═══════════════════════════════════════════════════════════════

class ResultsWidget(QWidget):
    """
    Results visualization and analysis widget.
    Displays training results with charts and export options.
    ENHANCED: Shows data type in results.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Action buttons
        action_frame = self.create_action_buttons()
        layout.addWidget(action_frame)
        
        # Results display
        results_card = self.create_results_card()
        layout.addWidget(results_card, stretch=1)
    
    def create_header(self):
        """Create page header"""
        header_frame = QFrame()
        header_frame.setStyleSheet(get_header_stylesheet())
        header_frame.setFixedHeight(100)
        
        layout = QVBoxLayout(header_frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel(f"{Icons.RESULTS} Training Results & Analysis")
        title.setStyleSheet(f"font-size: {Fonts.HEADING1_SIZE}px; font-weight: {Fonts.WEIGHT_BOLD}; color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("View and export quantum federated learning results")
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.8);")
        layout.addWidget(subtitle)
        
        return header_frame
    
    def create_action_buttons(self):
        """Create action buttons"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        generate_btn = QPushButton(f"{Icons.SEARCH} Generate Report")
        generate_btn.clicked.connect(self.generate_report)
        layout.addWidget(generate_btn)
        
        plot_btn = QPushButton(f"{Icons.RESULTS} Plot Results")
        plot_btn.clicked.connect(self.plot_results)
        layout.addWidget(plot_btn)
        
        export_btn = QPushButton(f"{Icons.DOWNLOAD} Export Data")
        export_btn.setProperty("primary", True)
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        return frame
    
    def create_results_card(self):
        """Create results display card"""
        card = QGroupBox("Training Results Summary")
        card_layout = QVBoxLayout(card)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setStyleSheet(get_log_viewer_stylesheet())
        card_layout.addWidget(self.results_display)
        
        # Show initial message
        self.show_no_results_message()
        
        return card
    
    def show_no_results_message(self):
        """Show message when no results available"""
        message = f"""

   TRAINING RESULTS - NO DATA                              

                                                           
   No training results available yet.                      
                                                           
   Complete a training session to view results here.       
                                                           
   Results will include:                                   
   • Accuracy progression over rounds                     
   • Client performance comparison                        
   • Training time analysis                               
   • Quantum circuit statistics                           
   • Data type information                                
   • Exportable data and visualizations                   

"""
        self.results_display.setPlainText(message)
    
    def generate_report(self):
        """Generate comprehensive training report"""
        server = self.main_window.get_federated_server()
        if not server:
            QMessageBox.warning(
                self, "No Results",
                "No training results available.\nPlease complete a training session first."
            )
            return
        
        try:
            summary = server.get_training_summary()
            stats = server.get_server_stats()
            history = server.training_history
            
            # Get data type
            dataset = self.main_window.get_dataset()
            data_type = 'numerical'
            if dataset and 'metadata' in dataset:
                data_type = dataset['metadata'].get('data_type', 'numerical')
            
            type_icon = {
                'numerical': Icons.NUMERICAL,
                'image': Icons.IMAGE,
                'text': Icons.TEXT
            }.get(data_type, Icons.DATASET)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode = "QUANTUM" if stats['model_type'] == 'quantum' else "CLASSICAL"
            backend = self.main_window.backend_manager.current_backend
            
            report = f"""

   {type_icon} QUANTUM FEDERATED LEARNING - TRAINING REPORT       

   Generated: {timestamp:<43} 
   Data Type: {data_type.upper():<46} 
   Model Mode: {mode:<45} 
   Backend: {backend.upper():<48} 


DATASET INFORMATION
{'-'*60}
- Data Type: {data_type.upper()}
"""
            
            # Add data-type-specific info
            if dataset and 'metadata' in dataset and 'preprocessing' in dataset['metadata']:
                preprocessing = dataset['metadata']['preprocessing']
                
                if data_type == 'image':
                    report += f"- Feature Extractor: {preprocessing.get('feature_extractor', 'N/A').upper()}\n"
                    report += f"- Feature Dimension: {preprocessing.get('feature_dim', 'N/A')}D\n"
                    report += f"- Total Images: {preprocessing.get('total_images', 'N/A')}\n"
                elif data_type == 'text':
                    model_name = preprocessing.get('embedding_model', '').split('/')[-1]
                    report += f"- Embedding Model: {model_name}\n"
                    report += f"- Embedding Dimension: {preprocessing.get('embedding_dim', 'N/A')}D\n"
                    report += f"- Total Texts: {preprocessing.get('total_texts', 'N/A')}\n"
                else:
                    report += f"- Total Features: {stats.get('input_dim', 'N/A')}\n"
            
            report += f"""- Number of Classes: {stats['n_classes']}

MODEL CONFIGURATION
{'-'*60}
- Model Type: {stats['model_type'].upper()}
- Quantum Qubits: {stats['quantum_qubits']}
- Federated Clients: {stats['n_clients']}
- Training Rounds: {summary['total_rounds']}
- Total Parameters: {stats['total_parameters']}

PERFORMANCE RESULTS
{'-'*60}
- Final Accuracy: {summary['final_global_accuracy']:.4f} ({summary['final_global_accuracy']*100:.1f}%)
- Best Accuracy: {summary['best_global_accuracy']:.4f} ({summary['best_global_accuracy']*100:.1f}%)
- Initial Accuracy: {summary['initial_accuracy']:.4f}
- Improvement: {summary['improvement']:.4f} ({summary['improvement']*100:.1f}%)

TRAINING EFFICIENCY
{'-'*60}
- Total Training Time: {format_time(summary['total_time'])}
- Average Round Time: {format_time(summary['avg_round_time'])}
- Convergence Round: {summary['convergence_round'] if summary['convergence_round'] > 0 else 'Not reached'}

ROUND-BY-ROUND ACCURACY
{'-'*60}
"""
            
            for i, (round_num, acc) in enumerate(zip(history['rounds'], history['global_accuracies'])):
                report += f"• Round {round_num:2d}: {acc:.4f} ({acc*100:.1f}%)\n"
            
            report += f"""
CLIENT SUMMARY
{'-'*60}
"""
            
            for client_stat in summary['client_summary']:
                report += f"• {client_stat['client_name']}:\n"
                report += f"  └─ Samples: {client_stat['total_samples']}, Best Acc: {client_stat['best_accuracy']:.4f}\n"
            
            report += f"""
RESEARCH NOTES
{'-'*60}
✓ Training completed successfully on {data_type.upper()} data
✓ Results ready for analysis and publication
✓ Data exportable for further processing

Experiment completed! 
"""
            
            self.results_display.setPlainText(report)
            
            log_message("Training report generated", "SUCCESS")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report:\n{e}")
    
    def plot_results(self):
        """Plot training results (opens matplotlib window)"""
        server = self.main_window.get_federated_server()
        if not server:
            QMessageBox.warning(
                self, "No Results",
                "No training results available."
            )
            return
        
        try:
            history = server.training_history
            
            # Get data type for title
            dataset = self.main_window.get_dataset()
            data_type = 'numerical'
            if dataset and 'metadata' in dataset:
                data_type = dataset['metadata'].get('data_type', 'numerical')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Quantum Federated Learning Results ({data_type.upper()} Data)', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Global accuracy over rounds
            axes[0, 0].plot(history['rounds'], history['global_accuracies'], 
                           'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_title('Global Model Accuracy')
            axes[0, 0].set_xlabel('Federated Round')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1.05)
            
            # Plot 2: Average client accuracy
            axes[0, 1].plot(history['rounds'], history['avg_client_accuracies'], 
                           'g-s', linewidth=2, markersize=6)
            axes[0, 1].set_title('Average Client Accuracy')
            axes[0, 1].set_xlabel('Federated Round')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1.05)
            
            # Plot 3: Average client loss
            axes[1, 0].plot(history['rounds'], history['avg_client_losses'], 
                           'r-^', linewidth=2, markersize=6)
            axes[1, 0].set_title('Average Client Loss')
            axes[1, 0].set_xlabel('Federated Round')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Round times
            axes[1, 1].bar(history['rounds'], history['round_times'], 
                          color='purple', alpha=0.7)
            axes[1, 1].set_title('Training Time per Round')
            axes[1, 1].set_xlabel('Federated Round')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
            
            log_message("Results plotted successfully", "SUCCESS")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot results:\n{e}")
    
    def export_results(self):
        """Export results to CSV file"""
        server = self.main_window.get_federated_server()
        if not server:
            QMessageBox.warning(
                self, "No Results",
                "No training results available to export."
            )
            return
        
        try:
            # Get data type
            dataset = self.main_window.get_dataset()
            data_type = 'numerical'
            if dataset and 'metadata' in dataset:
                data_type = dataset['metadata'].get('data_type', 'numerical')
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Training Results",
                f"quantum_fl_{data_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
            
            # Prepare data
            history = server.training_history
            
            df = pd.DataFrame({
                'Round': history['rounds'],
                'Global_Accuracy': history['global_accuracies'],
                'Avg_Client_Accuracy': history['avg_client_accuracies'],
                'Avg_Client_Loss': history['avg_client_losses'],
                'Round_Time_Seconds': history['round_times']
            })
            
            # Add metadata row
            metadata_row = pd.DataFrame([{
                'Round': f'DATA_TYPE: {data_type}',
                'Global_Accuracy': '',
                'Avg_Client_Accuracy': '',
                'Avg_Client_Loss': '',
                'Round_Time_Seconds': ''
            }])
            
            # Combine
            df_with_metadata = pd.concat([metadata_row, df], ignore_index=True)
            
            # Export
            df_with_metadata.to_csv(file_path, index=False)
            
            QMessageBox.information(
                self, "Export Complete",
                f"✅ Results exported successfully!\n\n"
                f"File: {file_path}\n"
                f"Data Type: {data_type.upper()}\n"
                f"Rows: {len(df)}"
            )
            
            log_message(f"Results exported to {file_path}", "SUCCESS")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{e}")

# ═══════════════════════════════════════════════════════════════
# BLOCK 9 COMPLETE
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 9/10: Training & Results Widgets - LOADED ✅")
print("ENHANCED: Shows data type in training and results")
print("="*60)
"""
════════════════════════════════════════════════════════════════
QUANTUM FEDERATED LEARNING PLATFORM 
Block 10/10: Main Entry Point & Launcher
════════════════════════════════════════════════════════════════
This block contains:
- Main application launcher
- Exception handling and crash recovery
- Startup checks and initialization
- Application entry point
════════════════════════════════════════════════════════════════
"""

# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT & APPLICATION LAUNCHER
# ════════════════════════════════════════════════════════════════

def check_dependencies():
    """
    Check if all required dependencies are installed.
    Returns: (success: bool, missing: list, warnings: list)
    """
    missing = []
    warnings = []
    
    # Required packages
    required = {
        'PyQt5': 'PyQt5',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    for module, package_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package_name)
    
    # Optional packages (warnings only)
    optional = {
        'qiskit': 'qiskit',
        'qiskit_aer': 'qiskit-aer',
        'qiskit_ibm_runtime': 'qiskit-ibm-runtime',
        'openpyxl': 'openpyxl'
    }
    
    for module, package_name in optional.items():
        try:
            __import__(module)
        except ImportError:
            warnings.append(package_name)
    
    success = len(missing) == 0
    return success, missing, warnings


def show_dependency_error(missing_packages):
    """Show error dialog for missing dependencies"""
    from PyQt5.QtWidgets import QApplication, QMessageBox
    
    app = QApplication(sys.argv)
    
    package_list = '\n'.join(f"  • {pkg}" for pkg in missing_packages)
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Missing Dependencies")
    msg.setText("Cannot start application - required packages are missing!")
    msg.setInformativeText(
        f"Please install the following packages:\n\n{package_list}\n\n"
        f"Install command:\n"
        f"pip install {' '.join(missing_packages)}"
    )
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def show_startup_warnings(warnings):
    """Show warnings about optional dependencies"""
    if not warnings:
        return
    
    warning_list = '\n'.join(f"  • {pkg}" for pkg in warnings)
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Optional Dependencies Missing")
    msg.setText("Some optional features may not be available.")
    msg.setInformativeText(
        f"Missing packages:\n\n{warning_list}\n\n"
        f"These packages are optional but recommended for full functionality.\n\n"
        f"Install command:\n"
        f"pip install {' '.join(warnings)}"
    )
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def setup_exception_handler():
    """Setup global exception handler for crash recovery"""
    def exception_hook(exctype, value, tb):
        """Handle uncaught exceptions"""
        error_msg = ''.join(traceback.format_exception(exctype, value, tb))
        
        print("\n" + "="*60)
        print("UNHANDLED EXCEPTION")
        print("="*60)
        print(error_msg)
        print("="*60)
        
        # Show error dialog
        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Application Error")
            msg.setText("An unexpected error occurred!")
            msg.setDetailedText(error_msg)
            msg.setInformativeText(
                "The application encountered an unexpected error.\n\n"
                "Please save your work and restart the application.\n"
                "If this persists, please report this error."
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        except:
            pass
        
        # Exit gracefully
        sys.exit(1)
    
    sys.excepthook = exception_hook


def create_application():
    """Create and configure the QApplication instance"""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Quantum Federated Learning Platform")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("University Quantum Research Lab")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application-wide stylesheet
    app.setStyleSheet(get_main_stylesheet())
    
    return app


def show_splash_screen(app):
    """Show splash screen during startup"""
    from PyQt5.QtWidgets import QSplashScreen
    from PyQt5.QtGui import QPixmap, QPainter, QFont
    from PyQt5.QtCore import Qt
    
    # Create splash screen pixmap
    pixmap = QPixmap(600, 400)
    pixmap.fill(QColor(Colors.BG_DARKEST))  # FIXED
    
    painter = QPainter(pixmap)
    
    # Draw gradient background
    gradient = QLinearGradient(0, 0, 0, 400)
    gradient.setColorAt(0, QColor(Colors.PRIMARY))
    gradient.setColorAt(1, QColor(Colors.PRIMARY_DARK))
    painter.fillRect(pixmap.rect(), gradient)
    
    # Draw title
    painter.setPen(QColor("white"))
    font = QFont(Fonts.PRIMARY_FAMILY, 28, Fonts.WEIGHT_BOLD)  # FIXED
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, 
                    f"{Icons.QUANTUM}\nQuantum Federated Learning\nPlatform")
    
    # Draw version
    font = QFont(Fonts.PRIMARY_FAMILY, 14)  # FIXED
    painter.setFont(font)
    painter.drawText(20, 360, "Version 1.0.0 - Research Edition")
    
    painter.end()
    
    # Create and show splash screen
    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    
    return splash


def initialize_directories():
    """Create necessary directories for the application"""
    directories = [
        Paths.RESOURCES_DIR,
        Paths.IMAGES_DIR,
        Paths.EXPORTS_DIR,
        Paths.CACHE_DIR
    ]  # FIXED - only use paths defined in Block 1
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def startup_checks():
    """Perform startup checks and initialization"""
    print("\n" + "="*60)
    print("QUANTUM FEDERATED LEARNING PLATFORM")
    print("="*60)
    print(f"Starting up... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Check dependencies
    print("Checking dependencies...")
    success, missing, warnings = check_dependencies()
    
    if not success:
        print(f"❌ FAILED: Missing required packages: {missing}")
        return False, missing, warnings
    
    print("✅ All required dependencies found")
    
    if warnings:
        print(f"⚠️  Optional packages missing: {warnings}")
    else:
        print("✅ All optional dependencies found")
    
    # Initialize directories
    print("\nInitializing directories...")
    try:
        initialize_directories()
        print("✅ Directories initialized")
    except Exception as e:
        print(f"⚠️  Directory initialization warning: {e}")
    
    # Check quantum backends
    print("\nChecking quantum backends...")
    if QISKIT_AVAILABLE:
        print("✅ Qiskit available - quantum features enabled")
    else:
        print("⚠️  Qiskit not available - using classical simulation")
    
    if QISKIT_IBM_AVAILABLE:
        print("✅ IBM Quantum Runtime available")
    else:
        print("⚠️  IBM Quantum Runtime not available")
    
    print("\n" + "="*60)
    print("Startup checks complete!")
    print("="*60 + "\n")
    
    return True, missing, warnings


def main():
    """
    Main application entry point.
    
    This function:
    1. Performs startup checks
    2. Initializes the Qt application
    3. Creates and shows the main window
    4. Starts the event loop
    """
    
    # Setup exception handler
    setup_exception_handler()
    
    # Perform startup checks
    success, missing, warnings_list = startup_checks()
    
    if not success:
        # Show error and exit
        show_dependency_error(missing)
        sys.exit(1)
    
    # Create Qt application
    app = create_application()
    
    # Show splash screen
    splash = show_splash_screen(app)
    
    # Small delay for splash screen
    import time
    time.sleep(1.5)
    
    # Show warnings if any
    if warnings_list:
        splash.close()
        show_startup_warnings(warnings_list)
    
    # Create main window
    print("Creating main window...")
    main_window = QuantumFederatedLearningApp()
    
    # Close splash and show main window
    splash.finish(main_window)
    main_window.show()
    
    print("Application ready! \n")
    
    # Log startup
    log_message("Application started successfully", "SUCCESS")
    
    # Start event loop
    exit_code = app.exec_()
    
    # Cleanup
    print("\n" + "="*60)
    print("Application shutdown")
    print("="*60)
    log_message("Application shutdown", "INFO")
    
    sys.exit(exit_code)


# ════════════════════════════════════════════════════════════════
# BLOCK 10 COMPLETE
# ════════════════════════════════════════════════════════════════

print("="*60)
print("BLOCK 10/10: Main Entry Point & Launcher - LOADED ✅")
print("="*60)


# ════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Application entry point.
    
    This is the starting point when the script is executed directly.
    It calls the main() function which handles all initialization
    and starts the Qt event loop.
    """
    
        
    main()