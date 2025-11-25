"""
Hybrid Recommendation Model for AITH 2025
Team: OneManArmy

This module implements a hybrid recommendation system combining:
1. LightFM for collaborative filtering with content features
2. Content-based similarity for cold-start handling
3. Popularity-based fallback for completely unknown items

The model is optimized for Recall@K metrics.
"""

import os
import numpy as np
import pickle
from scipy import sparse
from scipy.spatial.distance import cosine
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False
    print("[WARNING] LightFM not installed. Install with: pip install lightfm")


class HybridRecommender:
    """
    Hybrid Recommendation System combining:
    - LightFM (collaborative filtering + content features)
    - Content-based similarity (for cold-start)
    - Popularity fallback
    """
    
    def __init__(self, n_components=128, loss='warp', learning_rate=0.05,
                 item_alpha=1e-6, user_alpha=1e-6, random_state=42):
        """
        Initialize the hybrid recommender.
        
        Args:
            n_components: Number of latent factors
            loss: Loss function ('warp' optimizes ranking, 'bpr' for implicit)
            learning_rate: Learning rate for SGD
            item_alpha: L2 regularization for item features
            user_alpha: L2 regularization for user features
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.loss = loss
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.random_state = random_state
        
        self.model = None
        self.user_features = None
        self.item_features = None
        self.interactions = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.imdb_to_ml = {}
        self.ml_to_imdb = {}
        
        # Content-based components
        self.item_genre_matrix = None
        self.movie_popularity = {}
        self.genres = []
        
    def build_model(self):
        """Initialize the LightFM model"""
        if not LIGHTFM_AVAILABLE:
            raise ImportError("LightFM is required. Install with: pip install lightfm")
        
        self.model = LightFM(
            no_components=self.n_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
            random_state=self.random_state
        )
        
        return self.model
    
    def fit(self, interactions, user_features=None, item_features=None, 
            epochs=30, num_threads=4, verbose=True):
        """
        Train the model.
        
        Args:
            interactions: User-item interaction matrix (sparse)
            user_features: User feature matrix (sparse)
            item_features: Item feature matrix (sparse)
            epochs: Number of training epochs
            num_threads: Number of threads for training
            verbose: Print progress
        """
        if self.model is None:
            self.build_model()
        
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features
        
        print(f"[INFO] Training LightFM model...")
        print(f"  - Interactions: {interactions.shape}")
        print(f"  - User features: {user_features.shape if user_features is not None else 'None'}")
        print(f"  - Item features: {item_features.shape if item_features is not None else 'None'}")
        print(f"  - Epochs: {epochs}")
        
        for epoch in range(epochs):
            self.model.fit_partial(
                interactions,
                user_features=user_features,
                item_features=item_features,
                epochs=1,
                num_threads=num_threads,
                verbose=False
            )
            
            if verbose and (epoch + 1) % 5 == 0:
                # Calculate training metrics
                train_recall = recall_at_k(
                    self.model, interactions,
                    user_features=user_features,
                    item_features=item_features,
                    k=5, num_threads=num_threads
                ).mean()
                print(f"  Epoch {epoch + 1}/{epochs} - Recall@5: {train_recall:.4f}")
        
        print("[INFO] Training complete!")
        
    def predict_score(self, user_idx, item_idx):
        """Get prediction score for a single user-item pair"""
        if self.model is None:
            return 3.0  # Default score
        
        try:
            scores = self.model.predict(
                user_ids=np.array([user_idx]),
                item_ids=np.array([item_idx]),
                user_features=self.user_features,
                item_features=self.item_features
            )
            return float(scores[0])
        except:
            return 3.0
    
    def predict_for_user(self, user_idx, item_indices=None):
        """Get prediction scores for a user across all items"""
        if self.model is None:
            return np.full(self.interactions.shape[1], 3.0)
        
        if item_indices is None:
            item_indices = np.arange(self.interactions.shape[1])
        
        try:
            scores = self.model.predict(
                user_ids=np.full(len(item_indices), user_idx),
                item_ids=item_indices,
                user_features=self.user_features,
                item_features=self.item_features
            )
            return scores
        except:
            return np.full(len(item_indices), 3.0)
    
    def get_top_k_recommendations(self, user_idx, k=5, exclude_known=True):
        """Get top-K recommendations for a user"""
        scores = self.predict_for_user(user_idx)
        
        if exclude_known and self.interactions is not None:
            # Zero out items the user has already interacted with
            known_items = self.interactions[user_idx].nonzero()[1]
            scores[known_items] = -np.inf
        
        # Get top-K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    def content_based_score(self, user_genre_prefs, item_idx):
        """
        Calculate content-based score using genre similarity.
        Used for cold-start scenarios.
        """
        if self.item_genre_matrix is None or item_idx >= len(self.item_genre_matrix):
            return 0.5
        
        item_genres = self.item_genre_matrix[item_idx]
        
        if np.sum(item_genres) == 0 or np.sum(user_genre_prefs) == 0:
            return 0.5
        
        # Cosine similarity
        similarity = 1 - cosine(user_genre_prefs, item_genres)
        return similarity if not np.isnan(similarity) else 0.5
    
    def popularity_score(self, item_idx):
        """Get popularity score for an item"""
        if item_idx in self.idx_to_item:
            movie_id = self.idx_to_item[item_idx]
            if movie_id in self.movie_popularity:
                return self.movie_popularity[movie_id].get('popularity_score', 0.5)
        return 0.5
    
    def hybrid_predict(self, user_idx, item_idx, user_genre_prefs=None, 
                       is_known_user=True, is_known_item=True):
        """
        Hybrid prediction combining multiple signals.
        
        Args:
            user_idx: User index (or None if unknown)
            item_idx: Item index (or None if unknown)
            user_genre_prefs: User's genre preferences (for cold-start)
            is_known_user: Whether user is in training data
            is_known_item: Whether item is in training data
        """
        # Weights for different scenarios
        if is_known_user and is_known_item:
            # Best case: Use full collaborative filtering
            cf_weight = 0.8
            content_weight = 0.2
            popularity_weight = 0.0
        elif is_known_user and not is_known_item:
            # Known user, unknown movie: Content-based
            cf_weight = 0.0
            content_weight = 0.7
            popularity_weight = 0.3
        elif not is_known_user and is_known_item:
            # Unknown user, known movie: Popularity + content
            cf_weight = 0.0
            content_weight = 0.4
            popularity_weight = 0.6
        else:
            # Unknown both: Pure popularity
            cf_weight = 0.0
            content_weight = 0.0
            popularity_weight = 1.0
        
        score = 0.0
        
        # Collaborative filtering score
        if cf_weight > 0 and is_known_user and is_known_item:
            cf_score = self.predict_score(user_idx, item_idx)
            # Normalize to 0-1
            cf_score = (cf_score + 5) / 10  # Assuming scores roughly in [-5, 5]
            score += cf_weight * cf_score
        
        # Content-based score
        if content_weight > 0 and user_genre_prefs is not None:
            content_score = self.content_based_score(user_genre_prefs, item_idx)
            score += content_weight * content_score
        
        # Popularity score
        if popularity_weight > 0:
            pop_score = self.popularity_score(item_idx)
            score += popularity_weight * pop_score
        
        return score
    
    def evaluate(self, test_interactions, k_values=[1, 3, 5]):
        """
        Evaluate model on test data.
        
        Returns recall@k for different k values.
        """
        results = {}
        
        for k in k_values:
            recall = recall_at_k(
                self.model, test_interactions,
                user_features=self.user_features,
                item_features=self.item_features,
                k=k, num_threads=4
            ).mean()
            results[f'recall@{k}'] = recall
            print(f"  Recall@{k}: {recall:.4f}")
        
        return results
    
    def save_model(self, path='Resources/hybrid_model.pkl'):
        """Save the trained model and all components"""
        print(f"[INFO] Saving model to {path}...")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'lightfm_model': self.model,
            'n_components': self.n_components,
            'loss': self.loss,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'imdb_to_ml': self.imdb_to_ml,
            'ml_to_imdb': self.ml_to_imdb,
            'genres': self.genres,
            'movie_popularity': self.movie_popularity,
            'model_type': 'hybrid_lightfm'
        }
        
        # Save sparse matrices separately for efficiency
        if self.user_features is not None:
            model_data['user_features'] = self.user_features
        if self.item_features is not None:
            model_data['item_features'] = self.item_features
        if self.item_genre_matrix is not None:
            model_data['item_genre_matrix'] = self.item_genre_matrix
        if self.interactions is not None:
            model_data['interactions'] = self.interactions
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  - Model saved successfully!")
        
    def load_model(self, path='Resources/hybrid_model.pkl'):
        """Load a trained model"""
        print(f"[INFO] Loading model from {path}...")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data.get('lightfm_model')
        self.n_components = model_data.get('n_components', 128)
        self.loss = model_data.get('loss', 'warp')
        self.user_to_idx = model_data.get('user_to_idx', {})
        self.idx_to_user = model_data.get('idx_to_user', {})
        self.item_to_idx = model_data.get('item_to_idx', {})
        self.idx_to_item = model_data.get('idx_to_item', {})
        self.imdb_to_ml = model_data.get('imdb_to_ml', {})
        self.ml_to_imdb = model_data.get('ml_to_imdb', {})
        self.genres = model_data.get('genres', [])
        self.movie_popularity = model_data.get('movie_popularity', {})
        
        self.user_features = model_data.get('user_features')
        self.item_features = model_data.get('item_features')
        self.item_genre_matrix = model_data.get('item_genre_matrix')
        self.interactions = model_data.get('interactions')
        
        print(f"  - Model loaded successfully!")
        print(f"  - Model type: {model_data.get('model_type', 'unknown')}")


class ContentBasedRecommender:
    """
    Pure content-based recommender for cold-start scenarios.
    Uses genre similarity and movie metadata.
    """
    
    def __init__(self):
        self.movie_genres = {}
        self.genres = []
        self.genre_to_idx = {}
        
    def build_from_features(self, item_features, genres, item_to_idx, idx_to_item):
        """Build content model from item features"""
        self.item_features = item_features
        self.genres = genres
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        
    def get_similar_items(self, item_idx, k=10):
        """Find k most similar items based on genre"""
        if self.item_features is None:
            return []
        
        item_vector = self.item_features[item_idx].toarray().flatten()
        
        similarities = []
        for i in range(self.item_features.shape[0]):
            if i != item_idx:
                other_vector = self.item_features[i].toarray().flatten()
                sim = 1 - cosine(item_vector, other_vector)
                if not np.isnan(sim):
                    similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def recommend_for_genre_prefs(self, genre_prefs, k=5, exclude_items=None):
        """Recommend items based on genre preferences"""
        if self.item_features is None:
            return []
        
        scores = []
        for i in range(self.item_features.shape[0]):
            if exclude_items and i in exclude_items:
                continue
            
            item_vector = self.item_features[i].toarray().flatten()
            sim = 1 - cosine(genre_prefs, item_vector)
            if not np.isnan(sim):
                scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def train_hybrid_model(data_dir='data/aith-dataset', output_path='Resources/hybrid_model.pkl',
                       n_components=128, epochs=30):
    """
    Main training function for the hybrid model.
    """
    print("=" * 60)
    print("AITH 2025 - Hybrid Model Training")
    print("Team: OneManArmy")
    print("=" * 60)
    
    # Import feature engineering
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.feature_engineering import FeatureEngineer
    
    # Create features
    print("\n[STEP 1] Feature Engineering...")
    fe = FeatureEngineer(data_dir)
    fe.load_data()
    fe.extract_all_genres()
    fe.create_item_features()
    fe.create_user_features()
    fe.create_interaction_matrix()
    fe.create_binary_interactions()
    fe.create_movie_popularity_features()
    fe.get_imdb_to_ml_mapping()
    
    # Initialize hybrid model
    print("\n[STEP 2] Building Hybrid Model...")
    model = HybridRecommender(
        n_components=n_components,
        loss='warp',
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6
    )
    
    # Set mappings
    model.user_to_idx = fe.user_to_idx
    model.idx_to_user = fe.idx_to_user
    model.item_to_idx = fe.item_to_idx
    model.idx_to_item = fe.idx_to_item
    model.imdb_to_ml = fe.imdb_to_ml
    model.ml_to_imdb = fe.ml_to_imdb
    model.genres = fe.genres
    model.movie_popularity = fe.movie_popularity
    model.item_genre_matrix = fe.item_features.toarray()
    
    # Train model
    print("\n[STEP 3] Training Model...")
    model.fit(
        fe.binary_interactions,
        user_features=fe.user_features,
        item_features=fe.item_features,
        epochs=epochs,
        num_threads=4
    )
    
    # Evaluate
    print("\n[STEP 4] Evaluation...")
    model.evaluate(fe.binary_interactions, k_values=[1, 3, 5])
    
    # Save model
    print("\n[STEP 5] Saving Model...")
    model.save_model(output_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_hybrid_model()

