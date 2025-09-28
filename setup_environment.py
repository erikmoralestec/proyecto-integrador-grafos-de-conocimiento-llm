"""
Setup script for User Complaints Knowledge Graph Research Project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/graphs",
        "notebooks/eda",
        "notebooks/experiments",
        "src/data_processing",
        "src/graph_construction",
        "src/llm_integration",
        "src/evaluation",
        "models/checkpoints",
        "models/fine_tuned",
        "configs",
        "docs/research",
        "docs/methodology",
        "results/figures",
        "results/tables",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_config_files():
    """Create configuration files"""
    
    # Main config
    config_content = """# User Complaints Knowledge Graph Research Configuration

# Data paths
DATA_PATHS = {
    'raw_complaints': 'data/raw/CMPLT_2025.csv',
    'field_descriptions': 'data/raw/CMPL.txt',
    'processed_data': 'data/processed/',
    'graphs': 'data/graphs/'
}

# Model configurations
MODEL_CONFIGS = {
    'llm_models': ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet'],
    'embedding_models': ['text-embedding-ada-002', 'sentence-transformers/all-MiniLM-L6-v2'],
    'graph_models': ['networkx', 'neo4j', 'rdflib']
}

# Research parameters
RESEARCH_PARAMS = {
    'sample_size': 1000,  # For initial experiments
    'confidence_threshold': 0.8,
    'max_tokens': 4000,
    'temperature': 0.1
}

# Evaluation metrics
EVALUATION_METRICS = [
    'precision',
    'recall', 
    'f1_score',
    'graph_density',
    'clustering_coefficient',
    'node_centrality'
]
"""
    
    with open('configs/main_config.py', 'w') as f:
        f.write(config_content)
    
    # Environment file
    env_content = """# Environment variables for the project
# Copy this to .env and fill in your values

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# OpenAI API (if using)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face (if using)
HUGGINGFACE_API_KEY=your_hf_api_key_here

# Project paths
PROJECT_ROOT=.
DATA_DIR=data
RESULTS_DIR=results
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("📝 Created configuration files")

def main():
    """Main setup function"""
    print("🚀 Setting up User Complaints Knowledge Graph Research Environment")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create config files
    create_config_files()
    
    print("\n" + "="*70)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure your API keys")
    print("2. Run: python src/eda_analysis.py")
    print("3. Start with the Jupyter notebooks in notebooks/")
    print("="*70)

if __name__ == "__main__":
    main()
