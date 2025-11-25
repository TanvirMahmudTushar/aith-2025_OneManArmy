"""
SVD-based Recommendation Model (Fallback for Windows)
Team: OneManArmy

This module provides an SVD-based recommender as a fallback 
when LightFM cannot be installed (Windows compatibility issues).
Uses scikit-learn's TruncatedSVD which is always available.
"""

import os
import numpy as np
import pickle
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class SVDRecommender:
    """
    SVD-based Recommendation System using sklearn's TruncatedSVD.
    
    This is a simpler alternative to LightFM that works on all platforms.
    """
    
    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, interactions, n_iter=10):
        """
        Train SVD model on interaction matrix.
        
        Args:
            interactions: User-item interaction matrix (sparse)
            n_iter: Number of iterations for SVD
        """
        print(f"[INFO] Training SVD model with {self.n_components} components...")
        
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            n_iter=n_iter,
            random_state=self.random_state
        )
        
        # Fit SVD
        self.user_factors = self.svd.fit_transform(interactions)
        self.item_factors = self.svd.components_.T
        
        # Normalize factors
        self.user_factors = normalize(self.user_factors)
        self.item_factors = normalize(self.item_factors)
        
        print(f"[INFO] User factors shape: {self.user_factors.shape}")
        print(f"[INFO] Item factors shape: {self.item_factors.shape}")
        print(f"[INFO] Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        
    def predict(self, user_idx, item_idx):
        """Predict score for user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            return 0.5
        
        try:
            score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            return float(score)
        except:
            return 0.5
    
    def predict_all_items(self, user_idx):
        """Predict scores for all items for a user"""
        if self.user_factors is None:
            return np.zeros(self.item_factors.shape[0])
        
        return np.dot(self.user_factors[user_idx], self.item_factors.T)
    
    def get_top_k(self, user_idx, k=5, exclude_items=None):
        """Get top-K recommendations for a user"""
        scores = self.predict_all_items(user_idx)
        
        if exclude_items is not None:
            scores[list(exclude_items)] = -np.inf
        
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return top_k_idx, scores[top_k_idx]


class HybridSVDRecommender:
    """
    Hybrid SVD + Content-based recommender.
    Works on all platforms without LightFM dependency.
    """
    
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.svd_model = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.imdb_to_ml = {}
        self.ml_to_imdb = {}
        
        # Features
        self.item_genre_matrix = None
        self.genres = []
        self.genre_to_idx = {}
        
        # Popularity
        self.movie_popularity = {}
        
        # Interactions
        self.interactions = None
        
    def fit(self, interactions, item_features=None):
        """Train the model"""
        self.interactions = interactions
        
        # Train SVD
        self.svd_model = SVDRecommender(n_components=self.n_components)
        self.svd_model.fit(interactions)
        
        if item_features is not None:
            self.item_genre_matrix = item_features.toarray() if sparse.issparse(item_features) else item_features
    
    def predict_rating(self, user_idx, item_idx, user_genre_prefs=None, 
                       is_known_user=True, is_known_item=True):
        """Generate prediction for user-item pair"""
        
        if is_known_user and is_known_item:
            # Use SVD
            cf_score = self.svd_model.predict(user_idx, item_idx)
            content_score = self._content_score(user_genre_prefs, item_idx) if user_genre_prefs is not None else 0.5
            return 0.7 * cf_score + 0.3 * content_score
        
        elif is_known_user and not is_known_item:
            # Content-based only
            if user_genre_prefs is not None:
                return self._content_score(user_genre_prefs, item_idx)
            return 0.5
        
        elif not is_known_user and is_known_item:
            # Popularity + content
            pop_score = self._popularity_score(item_idx)
            content_score = self._content_score(user_genre_prefs, item_idx) if user_genre_prefs is not None else 0.5
            return 0.6 * pop_score + 0.4 * content_score
        
        else:
            return 0.5
    
    def _content_score(self, user_prefs, item_idx):
        """Calculate content-based score"""
        if self.item_genre_matrix is None or item_idx >= len(self.item_genre_matrix):
            return 0.5
        
        item_genres = self.item_genre_matrix[item_idx]
        
        if np.sum(item_genres) == 0 or np.sum(user_prefs) == 0:
            return 0.5
        
        try:
            similarity = 1 - cosine(user_prefs, item_genres)
            return similarity if not np.isnan(similarity) else 0.5
        except:
            return 0.5
    
    def _popularity_score(self, item_idx):
        """Get popularity score"""
        if item_idx in self.idx_to_item:
            movie_id = self.idx_to_item[item_idx]
            if movie_id in self.movie_popularity:
                return self.movie_popularity[movie_id].get('popularity_score', 0.5)
        return 0.5
    
    def save(self, path):
        """Save model"""
        model_data = {
            'svd_model': self.svd_model,
            'model_type': 'hybrid_svd',
            'n_components': self.n_components,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'imdb_to_ml': self.imdb_to_ml,
            'ml_to_imdb': self.ml_to_imdb,
            'genres': self.genres,
            'genre_to_idx': self.genre_to_idx,
            'item_genre_matrix': self.item_genre_matrix,
            'movie_popularity': self.movie_popularity,
            'interactions': self.interactions
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[INFO] Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svd_model = model_data.get('svd_model')
        self.n_components = model_data.get('n_components', 100)
        self.user_to_idx = model_data.get('user_to_idx', {})
        self.idx_to_user = model_data.get('idx_to_user', {})
        self.item_to_idx = model_data.get('item_to_idx', {})
        self.idx_to_item = model_data.get('idx_to_item', {})
        self.imdb_to_ml = model_data.get('imdb_to_ml', {})
        self.ml_to_imdb = model_data.get('ml_to_imdb', {})
        self.genres = model_data.get('genres', [])
        self.genre_to_idx = model_data.get('genre_to_idx', {})
        self.item_genre_matrix = model_data.get('item_genre_matrix')
        self.movie_popularity = model_data.get('movie_popularity', {})
        self.interactions = model_data.get('interactions')
        
        print(f"[INFO] Model loaded from {path}")


def train_svd_model(data_dir='data/aith-dataset', output_path='Resources/hybrid_model.pkl'):
    """Train SVD-based hybrid model (Windows compatible)"""
    import pandas as pd
    from scipy import sparse
    from tqdm import tqdm
    
    print("="*60)
    print("AITH 2025 - SVD Hybrid Model Training (Windows Compatible)")
    print("Team: OneManArmy")
    print("="*60)
    
    ml_path = os.path.join(data_dir, 'ml-latest-small')
    
    # Load data
    print("\n[INFO] Loading data...")
    ratings_df = pd.read_csv(os.path.join(ml_path, 'ratings.csv'))
    movies_df = pd.read_csv(os.path.join(ml_path, 'movies.csv'))
    links_df = pd.read_csv(os.path.join(ml_path, 'links.csv'))
    
    print(f"  Ratings: {len(ratings_df):,}")
    print(f"  Users: {ratings_df['userId'].nunique()}")
    print(f"  Movies: {ratings_df['movieId'].nunique()}")
    
    # Create mappings
    users = sorted(ratings_df['userId'].unique())
    items = sorted(ratings_df['movieId'].unique())
    
    n_users = len(users)
    n_items = len(items)
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {m: i for i, m in enumerate(items)}
    idx_to_item = {i: m for m, i in item_to_idx.items()}
    
    # IMDB mapping
    imdb_to_ml = {}
    ml_to_imdb = {}
    for _, row in links_df.iterrows():
        ml_id = int(row['movieId'])
        imdb_id = str(int(row['imdbId'])).zfill(7)
        imdb_link = f"https://www.imdb.com/title/tt{imdb_id}/"
        imdb_to_ml[imdb_link] = ml_id
        imdb_to_ml[imdb_link.rstrip('/')] = ml_id
        ml_to_imdb[ml_id] = imdb_link
    
    # Extract genres
    all_genres = set()
    for genres in movies_df['genres']:
        if genres != '(no genres listed)':
            all_genres.update(genres.split('|'))
    genres = sorted(list(all_genres))
    n_genres = len(genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    
    # Create item features
    print("\n[INFO] Creating features...")
    rows, cols, data = [], [], []
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        if movie_id not in item_to_idx:
            continue
        item_idx = item_to_idx[movie_id]
        movie_genres = row['genres']
        if movie_genres != '(no genres listed)':
            for genre in movie_genres.split('|'):
                if genre in genre_to_idx:
                    rows.append(item_idx)
                    cols.append(genre_to_idx[genre])
                    data.append(1.0)
    
    item_features = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_genres))
    
    # Create interactions
    print("[INFO] Creating interaction matrix...")
    RATING_THRESHOLD = 3.5
    rows, cols, data = [], [], []
    for _, row in ratings_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        if user_id in user_to_idx and movie_id in item_to_idx:
            if rating >= RATING_THRESHOLD:
                rows.append(user_to_idx[user_id])
                cols.append(item_to_idx[movie_id])
                data.append(1.0)
    
    interactions = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    print(f"  Positive interactions: {interactions.nnz:,}")
    
    # Movie popularity
    rating_counts = ratings_df.groupby('movieId').size()
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
    max_count = rating_counts.max()
    
    movie_popularity = {}
    for movie_id in item_to_idx.keys():
        count = rating_counts.get(movie_id, 0)
        avg = avg_ratings.get(movie_id, 3.0)
        movie_popularity[movie_id] = {
            'count': count,
            'normalized_count': count / max_count if max_count > 0 else 0,
            'avg_rating': avg,
            'popularity_score': (count / max_count) * (avg / 5.0) if max_count > 0 else 0
        }
    
    # Train model
    print("\n[INFO] Training SVD model...")
    model = HybridSVDRecommender(n_components=100)
    model.user_to_idx = user_to_idx
    model.idx_to_user = idx_to_user
    model.item_to_idx = item_to_idx
    model.idx_to_item = idx_to_item
    model.imdb_to_ml = imdb_to_ml
    model.ml_to_imdb = ml_to_imdb
    model.genres = genres
    model.genre_to_idx = genre_to_idx
    model.movie_popularity = movie_popularity
    
    model.fit(interactions, item_features)
    
    # Save model
    print("\n[INFO] Saving model...")
    model.save(output_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    train_svd_model()

