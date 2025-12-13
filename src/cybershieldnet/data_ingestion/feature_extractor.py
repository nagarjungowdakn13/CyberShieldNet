import torch
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import networkx as nx

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extraction for multi-modal threat intelligence data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.vectorizers = {}
        self.dimension_reducers = {}
        
    def extract_temporal_features(self, 
                                 sequences: torch.Tensor,
                                 feature_type: str = 'statistical') -> torch.Tensor:
        """Extract temporal features from sequences"""
        logger.info(f"Extracting {feature_type} temporal features...")
        
        if feature_type == 'statistical':
            return self._extract_statistical_features(sequences)
        elif feature_type == 'spectral':
            return self._extract_spectral_features(sequences)
        else:
            return sequences
    
    def extract_graph_features(self, 
                             graph_data: Dict,
                             feature_type: str = 'structural') -> torch.Tensor:
        """Extract graph features from network data"""
        logger.info(f"Extracting {feature_type} graph features...")
        
        if feature_type == 'structural':
            return self._extract_structural_features(graph_data)
        elif feature_type == 'spectral':
            return self._extract_spectral_graph_features(graph_data)
        else:
            return graph_data['x']
    
    def extract_behavioral_features(self, 
                                  raw_behavior: torch.Tensor,
                                  feature_type: str = 'anomaly') -> torch.Tensor:
        """Extract behavioral features from raw data"""
        logger.info(f"Extracting {feature_type} behavioral features...")
        
        if feature_type == 'anomaly':
            return self._extract_anomaly_features(raw_behavior)
        elif feature_type == 'temporal_behavior':
            return self._extract_temporal_behavior_features(raw_behavior)
        else:
            return raw_behavior
    
    def extract_text_features(self, 
                            text_data: List[str],
                            feature_type: str = 'tfidf') -> torch.Tensor:
        """Extract features from text data (threat reports, logs)"""
        logger.info(f"Extracting {feature_type} text features...")
        
        if feature_type == 'tfidf':
            return self._extract_tfidf_features(text_data)
        elif feature_type == 'embeddings':
            return self._extract_embedding_features(text_data)
        else:
            return torch.tensor([])
    
    def _extract_statistical_features(self, sequences: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from sequences"""
        features = []
        
        for seq in sequences:
            seq_np = seq.numpy()
            
            # Basic statistical features
            mean = np.mean(seq_np, axis=0)
            std = np.std(seq_np, axis=0)
            max_val = np.max(seq_np, axis=0)
            min_val = np.min(seq_np, axis=0)
            median = np.median(seq_np, axis=0)
            
            # Trend features
            if len(seq_np) > 1:
                slopes = np.polyfit(range(len(seq_np)), seq_np, 1)[0]
            else:
                slopes = np.zeros_like(mean)
            
            # Combine all features
            seq_features = np.concatenate([mean, std, max_val, min_val, median, slopes])
            features.append(seq_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_spectral_features(self, sequences: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from sequences"""
        features = []
        
        for seq in sequences:
            seq_np = seq.numpy()
            
            # FFT features
            fft = np.fft.fft(seq_np, axis=0)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Spectral statistics
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
            spectral_rolloff = np.percentile(magnitude, 85)
            
            # Combine features
            seq_features = np.concatenate([
                magnitude.mean(axis=0),
                phase.mean(axis=0),
                [spectral_centroid, spectral_rolloff]
            ])
            features.append(seq_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_structural_features(self, graph_data: Dict) -> torch.Tensor:
        """Extract structural features from graph"""
        features = []
        
        # Convert to networkx graph for feature extraction
        edge_index = graph_data['edge_index'].numpy()
        num_nodes = graph_data['x'].size(0)
        
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.T)
        
        # Extract node-level structural features
        for node in range(num_nodes):
            node_features = []
            
            # Degree centrality
            degree = G.degree(node)
            node_features.append(degree)
            
            # Betweenness centrality (approximate for large graphs)
            if num_nodes < 1000:
                betweenness = nx.betweenness_centrality(G).get(node, 0)
                node_features.append(betweenness)
            else:
                node_features.append(0)
            
            # Clustering coefficient
            clustering = nx.clustering(G, node)
            node_features.append(clustering)
            
            # Eigenvector centrality (approximate)
            if num_nodes < 500:
                try:
                    eigenvector = nx.eigenvector_centrality(G).get(node, 0)
                    node_features.append(eigenvector)
                except:
                    node_features.append(0)
            else:
                node_features.append(0)
            
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_anomaly_features(self, behavior_data: torch.Tensor) -> torch.Tensor:
        """Extract anomaly detection features from behavioral data"""
        features = []
        
        for behavior in behavior_data:
            behavior_np = behavior.numpy()
            
            # Statistical anomaly indicators
            z_scores = np.abs((behavior_np - behavior_np.mean()) / (behavior_np.std() + 1e-8))
            outlier_count = np.sum(z_scores > 3)  # Beyond 3 standard deviations
            
            # Entropy-based features
            from scipy.stats import entropy
            hist, _ = np.histogram(behavior_np, bins=10)
            prob = hist / np.sum(hist)
            behavior_entropy = entropy(prob)
            
            # Volatility features
            if len(behavior_np) > 1:
                changes = np.diff(behavior_np)
                volatility = np.std(changes)
            else:
                volatility = 0
            
            # Combine features
            behavior_features = np.array([outlier_count, behavior_entropy, volatility])
            features.append(behavior_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_tfidf_features(self, text_data: List[str]) -> torch.Tensor:
        """Extract TF-IDF features from text data"""
        if 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        vectorizer = self.vectorizers['tfidf']
        features = vectorizer.fit_transform(text_data).toarray()
        
        return torch.tensor(features, dtype=torch.float32)
    
    def reduce_dimensionality(self, 
                            features: torch.Tensor, 
                            method: str = 'pca',
                            n_components: int = 50) -> torch.Tensor:
        """Reduce feature dimensionality"""
        if method not in self.dimension_reducers:
            if method == 'pca':
                self.dimension_reducers[method] = PCA(n_components=n_components)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        reducer = self.dimension_reducers[method]
        features_np = features.numpy()
        reduced_features = reducer.fit_transform(features_np)
        
        return torch.tensor(reduced_features, dtype=torch.float32)