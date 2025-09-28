# User Complaints Knowledge Graph Research Configuration

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
