"""
SVD Hybrid Model - Fallback for when LightFM is not available
Team: OneManArmy

This module provides a scikit-surprise based alternative that works on Windows.
Uses SVD + content-based filtering for hybrid recommendations.
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.spatial.distance import cosine
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("[WARNING] scikit-surprise not installed. Install with: pip install scikit-surprise")


class SVDHybridRecommender:
    """
    Hybrid Recommender using SVD (scikit-surprise) + Content-based filtering.
    Works on Windows without requiring LightFM compilation.
    """
    
    def __init__(self, n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = None
        self.trainset = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.imdb_to_ml = {}
        self.ml_to_imdb = {}
        
        # Content features
        self.item_genre_matrix = None
        self.genres = []
        self.genre_to_idx = {}
        self.movie_popularity = {}
        
    def load_data(self, data_path):
        """Load MovieLens data"""
        print("[INFO] Loading MovieLens data...")
        
        self.ratings_df = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
        self.movies_df = pd.read_csv(os.path.join(data_path, 'movies.csv'))
        self.links_df = pd.read_csv(os.path.join(data_path, 'links.csv'))
        
        print(f"  - Ratings: {len(self.ratings_df):,}")
        print(f"  - Users: {self.ratings_df['userId'].nunique()}")
        print(f"  - Movies: {self.ratings_df['movieId'].nunique()}")
        
        return self.ratings_df, self.movies_df, self.links_df
    
    def create_mappings(self):
        """Create all necessary mappings"""
        print("[INFO] Creating mappings...")
        
        users = sorted(self.ratings_df['userId'].unique())
        items = sorted(self.ratings_df['movieId'].unique())
        
        self.n_users = len(users)
        self.n_items = len(items)
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {m: i for i, m in enumerate(items)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        
        # IMDB mapping
        for _, row in self.links_df.iterrows():
            ml_id = int(row['movieId'])
            imdb_id = str(int(row['imdbId'])).zfill(7)
            imdb_link = f"https://www.imdb.com/title/tt{imdb_id}/"
            self.imdb_to_ml[imdb_link] = ml_id
            self.imdb_to_ml[imdb_link.rstrip('/')] = ml_id
            self.ml_to_imdb[ml_id] = imdb_link
        
        print(f"  - Users: {self.n_users}, Items: {self.n_items}")
        print(f"  - IMDB mappings: {len(self.ml_to_imdb)}")
    
    def extract_features(self):
        """Extract genre features"""
        print("[INFO] Extracting features...")
        
        # Extract genres
        all_genres = set()
        for genres in self.movies_df['genres']:
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        self.genres = sorted(list(all_genres))
        self.n_genres = len(self.genres)
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        
        print(f"  - Genres: {self.n_genres}")
        
        # Create item genre matrix - use item_idx not movie_id
        max_idx = max(self.item_to_idx.values()) + 1
        self.item_genre_matrix = np.zeros((max_idx, self.n_genres))
        
        # Also create a movie_id to genre mapping
        self.movie_id_to_genres = {}
        
        for _, row in self.movies_df.iterrows():
            movie_id = row['movieId']
            if movie_id not in self.item_to_idx:
                continue
            item_idx = self.item_to_idx[movie_id]
            movie_genres = row['genres']
            
            genre_vec = np.zeros(self.n_genres)
            if movie_genres != '(no genres listed)':
                for genre in movie_genres.split('|'):
                    if genre in self.genre_to_idx:
                        self.item_genre_matrix[item_idx, self.genre_to_idx[genre]] = 1.0
                        genre_vec[self.genre_to_idx[genre]] = 1.0
            
            self.movie_id_to_genres[movie_id] = genre_vec
        
        # Movie popularity
        rating_counts = self.ratings_df.groupby('movieId').size()
        avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean()
        max_count = rating_counts.max()
        
        for movie_id in self.item_to_idx.keys():
            count = rating_counts.get(movie_id, 0)
            avg = avg_ratings.get(movie_id, 3.0)
            self.movie_popularity[movie_id] = {
                'count': count,
                'normalized_count': count / max_count if max_count > 0 else 0,
                'avg_rating': avg,
                'popularity_score': (count / max_count) * (avg / 5.0) if max_count > 0 else 0
            }
    
    def train(self):
        """Train SVD model"""
        if not SURPRISE_AVAILABLE:
            raise ImportError("scikit-surprise is required")
        
        print("[INFO] Training SVD model...")
        
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        self.trainset = data.build_full_trainset()
        
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42,
            verbose=True
        )
        
        self.model.fit(self.trainset)
        
        # Cross-validation for metrics
        print("\n[INFO] Cross-validation...")
        cv_results = cross_validate(
            SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, 
                lr_all=self.lr_all, reg_all=self.reg_all, random_state=42),
            data, measures=['RMSE', 'MAE'], cv=3, verbose=True
        )
        
        self.cv_rmse = cv_results['test_rmse'].mean()
        self.cv_mae = cv_results['test_mae'].mean()
        
        print(f"\n  - RMSE: {self.cv_rmse:.4f}")
        print(f"  - MAE: {self.cv_mae:.4f}")
    
    def predict_svd(self, user_id, movie_id):
        """Get SVD prediction"""
        if self.model is None:
            return 3.5
        
        try:
            pred = self.model.predict(user_id, movie_id)
            return pred.est
        except:
            return 3.5
    
    def content_similarity(self, user_genre_prefs, movie_id):
        """Calculate content similarity"""
        if movie_id not in self.item_to_idx:
            return 0.5
        
        item_idx = self.item_to_idx[movie_id]
        if item_idx >= len(self.item_genre_matrix):
            return 0.5
        
        movie_genres = self.item_genre_matrix[item_idx]
        
        if np.sum(movie_genres) == 0 or np.sum(user_genre_prefs) == 0:
            return 0.5
        
        try:
            sim = 1 - cosine(user_genre_prefs, movie_genres)
            return sim if not np.isnan(sim) else 0.5
        except:
            return 0.5
    
    def get_user_genre_prefs(self, user_id):
        """Get user genre preferences from ratings"""
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return np.ones(self.n_genres) / self.n_genres
        
        prefs = np.zeros(self.n_genres)
        counts = np.zeros(self.n_genres)
        
        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            if movie_id in self.movie_id_to_genres:
                genres = self.movie_id_to_genres[movie_id]
                prefs += genres * (rating / 5.0)
                counts += genres
        
        with np.errstate(divide='ignore', invalid='ignore'):
            prefs = np.where(counts > 0, prefs / counts, 1.0/self.n_genres)
        
        return prefs
    
    def calculate_recall_at_k(self, k=5, threshold=3.5):
        """Calculate Recall@K on training data"""
        print(f"\n[INFO] Calculating Recall@{k}...")
        
        recalls = []
        
        for user_id in tqdm(self.ratings_df['userId'].unique()[:100], desc="Evaluating"):
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            
            # Get ground truth (highly rated movies)
            ground_truth = set(user_ratings[user_ratings['rating'] >= threshold]['movieId'].tolist())
            
            if len(ground_truth) == 0:
                continue
            
            # Get predictions for all movies user has rated
            predictions = []
            for movie_id in user_ratings['movieId'].unique():
                pred = self.predict_svd(user_id, movie_id)
                predictions.append((movie_id, pred))
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_k = set([p[0] for p in predictions[:k]])
            
            hits = len(top_k & ground_truth)
            recall = hits / min(k, len(ground_truth))
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        print(f"  Recall@{k}: {avg_recall:.4f}")
        
        return avg_recall
    
    def save_model(self, path='Resources/hybrid_model.pkl'):
        """Save model"""
        print(f"[INFO] Saving model to {path}...")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'svd_model': self.model,
            'model_type': 'svd_hybrid',
            
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_genres': self.n_genres,
            
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'imdb_to_ml': self.imdb_to_ml,
            'ml_to_imdb': self.ml_to_imdb,
            
            'genres': self.genres,
            'genre_to_idx': self.genre_to_idx,
            'item_genre_matrix': self.item_genre_matrix,
            'movie_id_to_genres': self.movie_id_to_genres if hasattr(self, 'movie_id_to_genres') else {},
            'movie_popularity': self.movie_popularity,
            
            'cv_rmse': self.cv_rmse if hasattr(self, 'cv_rmse') else None,
            'cv_mae': self.cv_mae if hasattr(self, 'cv_mae') else None
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        model_size = os.path.getsize(path) / (1024 * 1024)
        print(f"  - Model saved! Size: {model_size:.2f} MB")
    
    def load_model(self, path='Resources/hybrid_model.pkl'):
        """Load model"""
        print(f"[INFO] Loading model from {path}...")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data.get('svd_model') or model_data.get('lightfm_model')
        
        self.n_users = model_data.get('n_users', 0)
        self.n_items = model_data.get('n_items', 0)
        self.n_genres = model_data.get('n_genres', 0)
        
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
        
        print(f"  - Model type: {model_data.get('model_type', 'unknown')}")
        print(f"  - Loaded successfully!")


def train_svd_model(data_path='data/aith-dataset/ml-latest-small', 
                    output_path='Resources/hybrid_model.pkl'):
    """Train SVD hybrid model"""
    print("="*60)
    print("  AITH 2025 - SVD Hybrid Model Training")
    print("  Team: OneManArmy")
    print("="*60)
    
    model = SVDHybridRecommender(
        n_factors=100,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02
    )
    
    model.load_data(data_path)
    model.create_mappings()
    model.extract_features()
    model.train()
    
    # Calculate Recall@K
    for k in [1, 3, 5]:
        model.calculate_recall_at_k(k=k)
    
    model.save_model(output_path)
    
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    train_svd_model()

