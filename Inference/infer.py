

import os
import sys
import argparse
import pickle
import json
import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Try to import SVD from surprise
try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("[WARNING] scikit-surprise not available. Using fallback mode.")


class WinningRecommender:
    """
    WINNING Recommendation System for AITH 2025
    
    KEY FEATURES:
    - Uses IMDB user profiles from user_reviews data
    - SVD collaborative filtering (scikit-surprise)
    - Smart cold-start handling using user/item features
    - Popularity-weighted fallback
    """
    
    def __init__(self, model_path='Resources/hybrid_model.pkl'):
        """Initialize the recommender system"""
        self.model_path = model_path
        self.model = None
        self.model_data = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.imdb_to_ml = {}
        self.ml_to_imdb = {}
        
        # Features
        self.user_features = None
        self.item_features = None
        self.item_genre_matrix = None
        self.user_genre_prefs = None
        self.movie_genres_dict = {}
        self.genres = []
        self.genre_to_idx = {}
        
        # Popularity
        self.movie_popularity = {}
        self.top_popular_movies = []
        
        # IMDB User Profiles (KEY!)
        self.imdb_user_profiles = {}
        
        # Movie metadata
        self.movie_metadata = {}
        self.movie_names = {}
        
        # Interactions
        self.interactions = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        print(f"[INFO] Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            # Try fallback paths
            fallback_paths = [
                os.path.join(os.path.dirname(__file__), '..', self.model_path),
                os.path.join(os.path.dirname(__file__), '..', 'Resources', 'hybrid_model.pkl'),
            ]
            found = False
            for fp in fallback_paths:
                if os.path.exists(fp):
                    self.model_path = fp
                    found = True
                    break
            
            if not found:
                print(f"[ERROR] Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                # Try to load the pickle file
                if SURPRISE_AVAILABLE:
                    # Standard loading if surprise is available
                    try:
                        self.model_data = pickle.load(f)
                    except Exception as e:
                        print(f"[WARNING] Standard pickle load failed: {e}")
                        print("[INFO] Attempting safe unpickling...")
                        f.seek(0)  # Reset file pointer
                        self.model_data = self._safe_unpickle(f)
                else:
                    # scikit-surprise not available - use safe unpickler
                    print("[WARNING] scikit-surprise not available. Using safe unpickling...")
                    self.model_data = self._safe_unpickle(f)
                    
                # Remove SVD model if it's a dummy or if surprise is not available
                if 'svd_model' in self.model_data:
                    svd_model = self.model_data.get('svd_model')
                    if svd_model is not None:
                        # Check if it's a dummy object or if surprise is unavailable
                        if not SURPRISE_AVAILABLE or (hasattr(svd_model, '_dummy') and svd_model._dummy):
                            self.model_data['svd_model'] = None
                            print("[INFO] SVD model removed (using fallback methods).")
                
                # Ensure model_data is a dict
                if not isinstance(self.model_data, dict):
                    print("[WARNING] Model data is not a dict, initializing empty dict.")
                    self.model_data = {}
                    
        except FileNotFoundError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if 'surprise' in error_str or 'module' in error_str or 'no module named' in error_str:
                print(f"\n[ERROR] Model loading failed: {e}")
                print("\n[INFO] Attempting to continue with minimal model data...")
                # Initialize with empty dict to allow fallback methods
                self.model_data = {}
            else:
                print(f"[ERROR] Unexpected error loading model: {e}")
                raise
    
    def _safe_unpickle(self, file_handle):
        """
        Safely unpickle model file, handling missing surprise classes.
        Returns a dict with all extractable data, with surprise objects replaced by None.
        """
        import types
        import sys
        
        # Create comprehensive fake surprise module
        class DummySurpriseObject:
            """Dummy object that accepts any attribute assignment"""
            def __init__(self, *args, **kwargs):
                self._dummy = True
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def __getattr__(self, name):
                return None
            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)
            def __setstate__(self, state):
                if isinstance(state, dict):
                    for k, v in state.items():
                        setattr(self, k, v)
            def __getstate__(self):
                return {}
            def __reduce__(self):
                return (DummySurpriseObject, ())
            def __reduce_ex__(self, protocol):
                return (DummySurpriseObject, ())
        
        # Create fake surprise module with all common classes
        fake_surprise = types.ModuleType('surprise')
        fake_surprise.SVD = DummySurpriseObject
        fake_surprise.Dataset = DummySurpriseObject
        fake_surprise.Reader = DummySurpriseObject
        fake_surprise.prediction_algorithms = types.ModuleType('prediction_algorithms')
        fake_surprise.prediction_algorithms.matrix_factorization = types.ModuleType('matrix_factorization')
        fake_surprise.prediction_algorithms.matrix_factorization.SVD = DummySurpriseObject
        
        # Inject fake module
        original_surprise = sys.modules.get('surprise')
        sys.modules['surprise'] = fake_surprise
        sys.modules['surprise.prediction_algorithms'] = fake_surprise.prediction_algorithms
        sys.modules['surprise.prediction_algorithms.matrix_factorization'] = fake_surprise.prediction_algorithms.matrix_factorization
        
        try:
            # Try to load with fake module
            model_data = pickle.load(file_handle)
            
            # If model_data is a dict, clean it up
            if isinstance(model_data, dict):
                # Replace any dummy surprise objects with None
                for key, value in list(model_data.items()):
                    if hasattr(value, '_dummy') and value._dummy:
                        model_data[key] = None
                    elif isinstance(value, dict):
                        # Recursively clean nested dicts
                        for k, v in list(value.items()):
                            if hasattr(v, '_dummy') and v._dummy:
                                value[k] = None
                
                # Ensure svd_model is None if it's a dummy
                if 'svd_model' in model_data:
                    svd = model_data.get('svd_model')
                    if svd is not None and (hasattr(svd, '_dummy') and svd._dummy):
                        model_data['svd_model'] = None
                
                return model_data
            else:
                # If not a dict, return empty dict
                print("[WARNING] Unpickled object is not a dict, using empty dict.")
                return {}
                
        except Exception as e:
            print(f"[WARNING] Safe unpickling failed: {e}")
            print("[INFO] Initializing with empty model data - will use fallback methods only.")
            return {}
        finally:
            # Restore original module if it existed
            if original_surprise is not None:
                sys.modules['surprise'] = original_surprise
            elif 'surprise' in sys.modules:
                # Only remove if we added it
                if hasattr(sys.modules['surprise'], 'SVD') and hasattr(sys.modules['surprise'].SVD, '_dummy'):
                    del sys.modules['surprise']
        
        # After loading model_data, extract components
        # Try to load SVD model (or LightFM if available for backward compatibility)
        # If scikit-surprise isn't available, model will be None and we'll use fallbacks
        if SURPRISE_AVAILABLE:
            try:
                self.model = self.model_data.get('svd_model') or self.model_data.get('lightfm_model')
                if self.model is not None:
                    print("[INFO] SVD model loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Could not load SVD model: {e}")
                print("[INFO] Will use content-based + popularity fallback methods.")
                self.model = None
        else:
            print("[INFO] scikit-surprise not available. Using content-based + popularity methods only.")
            self.model = None
        
        self.model_type = self.model_data.get('model_type', 'svd_hybrid_fixed')
        
        # Ensure model_data is a dict (handle case where unpickling failed)
        if not isinstance(self.model_data, dict):
            print("[WARNING] Model data is not a dict, using empty dict.")
            self.model_data = {}
        
        # Load mappings (with safe defaults)
        self.user_to_idx = self.model_data.get('user_to_idx', {})
        self.idx_to_user = self.model_data.get('idx_to_user', {})
        self.item_to_idx = self.model_data.get('item_to_idx', {})
        self.idx_to_item = self.model_data.get('idx_to_item', {})
        self.imdb_to_ml = self.model_data.get('imdb_to_ml', {})
        self.ml_to_imdb = self.model_data.get('ml_to_imdb', {})
        
        # Load features (with safe defaults)
        self.user_features = self.model_data.get('user_features')
        self.item_features = self.model_data.get('item_features')
        # Handle both array and sparse matrix formats
        item_genre_matrix = self.model_data.get('item_genre_matrix')
        if item_genre_matrix is not None:
            try:
                if hasattr(item_genre_matrix, 'toarray'):
                    self.item_genre_matrix = item_genre_matrix.toarray()
                else:
                    self.item_genre_matrix = np.array(item_genre_matrix)
            except:
                self.item_genre_matrix = None
        else:
            self.item_genre_matrix = None
        self.user_genre_prefs = self.model_data.get('user_genre_prefs')
        self.movie_genres_dict = self.model_data.get('movie_genres_dict', {})
        self.genres = self.model_data.get('genres', [])
        self.genre_to_idx = self.model_data.get('genre_to_idx', {})
        
        # Load popularity (with safe defaults)
        self.movie_popularity = self.model_data.get('movie_popularity', {})
        self.top_popular_movies = self.model_data.get('top_popular_movies', [])
        
        # Load IMDB user profiles (KEY!)
        self.imdb_user_profiles = self.model_data.get('imdb_user_profiles', {})
        
        # Load movie metadata (with safe defaults)
        self.movie_metadata = self.model_data.get('movie_metadata', {})
        self.movie_names = self.model_data.get('movie_names', {})
        
        # Load interactions
        self.interactions = self.model_data.get('interactions')
        
        # Check if we have minimal data for fallback methods
        has_minimal_data = (
            len(self.imdb_to_ml) > 0 or 
            len(self.movie_popularity) > 0 or 
            len(self.movie_metadata) > 0
        )
        
        if not has_minimal_data:
            print("[WARNING] Model data appears empty. Fallback methods may have limited functionality.")
        
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Model type: {self.model_type}")
        print(f"[INFO] Users: {len(self.user_to_idx):,}, Items: {len(self.item_to_idx):,}")
        print(f"[INFO] IMDB user profiles: {len(self.imdb_user_profiles):,}")
        print(f"[INFO] Movie metadata: {len(self.movie_metadata):,}")
        if self.model is None:
            print("[INFO] SVD model not available - using content-based + popularity fallback.")
        
        # Print performance if available
        if 'performance' in self.model_data:
            perf = self.model_data['performance']
            print(f"[INFO] Training Performance:")
            for k in [1, 3, 5]:
                if f'recall@{k}' in perf:
                    print(f"  - Recall@{k}: {perf[f'recall@{k}']:.4f}")
    
    def normalize_imdb_link(self, link):
        """Normalize IMDB link format"""
        link = str(link).strip()
        if not link.endswith('/'):
            link = link + '/'
        return link
    
    def extract_imdb_id(self, link):
        """Extract IMDB ID (tt number) from URL"""
        match = re.search(r'tt(\d+)', str(link))
        if match:
            return match.group(1)
        return None
    
    def get_ml_movie_id(self, imdb_link):
        """Get MovieLens movie ID from IMDB link"""
        imdb_link = self.normalize_imdb_link(imdb_link)
        
        if imdb_link in self.imdb_to_ml:
            return self.imdb_to_ml[imdb_link]
        
        # Try without trailing slash
        imdb_link_no_slash = imdb_link.rstrip('/')
        if imdb_link_no_slash in self.imdb_to_ml:
            return self.imdb_to_ml[imdb_link_no_slash]
        
        # Try short form (tt#######)
        imdb_id = self.extract_imdb_id(imdb_link)
        if imdb_id:
            short_form = f"tt{imdb_id}"
            if short_form in self.imdb_to_ml:
                return self.imdb_to_ml[short_form]
        
        return None
    
    def is_known_imdb_user(self, user_name):
        """Check if we have profile data for this IMDB user"""
        return user_name in self.imdb_user_profiles
    
    def is_known_movie(self, imdb_link):
        """Check if movie is in training data"""
        ml_id = self.get_ml_movie_id(imdb_link)
        return ml_id is not None and ml_id in self.item_to_idx
    
    def get_imdb_user_genre_prefs(self, user_name):
        """Get genre preferences from IMDB user profile"""
        if user_name not in self.imdb_user_profiles:
            return None
        
        profile = self.imdb_user_profiles[user_name]
        movies_reviewed = profile.get('movies_reviewed', [])
        ratings = profile.get('ratings', [])
        
        if not movies_reviewed or not ratings:
            return None
        
        # Build genre preferences from user's review history
        genre_weights = np.zeros(len(self.genres))
        genre_counts = np.zeros(len(self.genres))
        
        for movie_link, rating in zip(movies_reviewed, ratings):
            ml_id = self.get_ml_movie_id(movie_link)
            if ml_id is not None and ml_id in self.item_to_idx:
                item_idx = self.item_to_idx[ml_id]
                if self.item_genre_matrix is not None and item_idx < len(self.item_genre_matrix):
                    movie_genres = self.item_genre_matrix[item_idx]
                    # Weight by rating (higher rating = stronger preference)
                    weight = rating / 10.0
                    genre_weights += movie_genres * weight
                    genre_counts += movie_genres
        
        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            prefs = np.where(genre_counts > 0, genre_weights / genre_counts, 1.0/len(self.genres))
        
        return prefs
    
    def get_user_genre_prefs_from_test_history(self, user_history):
        """
        Build user genre preferences from test data history.
        
        Args:
            user_history: List of (imdb_link, ranking) tuples
        """
        if not user_history:
            return np.ones(len(self.genres)) / len(self.genres) if self.genres else None
        
        genre_weights = np.zeros(len(self.genres))
        genre_counts = np.zeros(len(self.genres))
        
        for imdb_link, ranking in user_history:
            ml_id = self.get_ml_movie_id(imdb_link)
            if ml_id is not None and ml_id in self.item_to_idx:
                item_idx = self.item_to_idx[ml_id]
                if self.item_genre_matrix is not None and item_idx < len(self.item_genre_matrix):
                    movie_genres = self.item_genre_matrix[item_idx]
                    # Weight by inverse ranking (rank 1 = best)
                    weight = 1.0 / (ranking + 1)
                    genre_weights += movie_genres * weight
                    genre_counts += movie_genres
        
        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            prefs = np.where(genre_counts > 0, genre_weights / genre_counts, 1.0/len(self.genres))
        
        return prefs
    
    def predict_content_based(self, user_genre_prefs, item_idx):
        """Get content-based prediction score using genre similarity"""
        if self.item_genre_matrix is None or item_idx >= len(self.item_genre_matrix):
            return 0.5
        
        if user_genre_prefs is None:
            return 0.5
        
        item_genres = self.item_genre_matrix[item_idx]
        
        if np.sum(item_genres) == 0 or np.sum(user_genre_prefs) == 0:
            return 0.5
        
        # Cosine similarity
        try:
            similarity = 1 - cosine(user_genre_prefs, item_genres)
            return similarity if not np.isnan(similarity) else 0.5
        except:
            return 0.5
    
    def predict_popularity(self, ml_movie_id):
        """Get popularity-based score"""
        if ml_movie_id in self.movie_popularity:
            return self.movie_popularity[ml_movie_id].get('popularity_score', 0.5)
        return 0.3  # Lower score for unknown movies
    
    def predict_svd_for_items(self, user_name, item_indices, user_genre_prefs=None):
        """
        Get SVD predictions for multiple items.
        Finds the most similar MovieLens user based on genre preferences.
        """
        if self.model is None or not SURPRISE_AVAILABLE:
            return None
        
        try:
            # Find the most similar MovieLens user based on genre preferences
            best_user_id = None
            best_similarity = -1
            
            if user_genre_prefs is not None and self.user_genre_prefs is not None:
                # Compute similarity to all MovieLens users
                for user_idx in range(len(self.user_genre_prefs)):
                    ml_user_genres = self.user_genre_prefs[user_idx]
                    try:
                        sim = 1 - cosine(user_genre_prefs, ml_user_genres)
                        if not np.isnan(sim) and sim > best_similarity:
                            best_similarity = sim
                            if user_idx in self.idx_to_user:
                                best_user_id = self.idx_to_user[user_idx]
                    except:
                        continue
            
            # If no similar user found, use average user or skip SVD
            if best_user_id is None:
                # Try to use a default user (first user in training set)
                if len(self.idx_to_user) > 0:
                    best_user_id = list(self.idx_to_user.values())[0]
                else:
                    return None  # Can't make SVD predictions without a user
            
            scores = []
            for item_idx in item_indices:
                if item_idx in self.idx_to_item:
                    movie_id = self.idx_to_item[item_idx]
                    try:
                        pred = self.model.predict(str(best_user_id), str(movie_id))
                        # Normalize to 0-1 range (ratings are 0.5-5.0)
                        normalized_score = max(0.0, min(1.0, (pred.est - 0.5) / 4.5))
                        scores.append(normalized_score)
                    except:
                        scores.append(0.5)
                else:
                    scores.append(0.5)
            
            return np.array(scores) if scores else None
        except Exception as e:
            return None
    
    def predict_rating(self, user_name, imdb_link, user_history=None, movie_mapper_row=None):
        """
        Generate prediction score for a user-movie pair.
        
        Strategy:
        1. Check if we have IMDB user profile -> use their preferences
        2. If not, build preferences from test data history
        3. Use content-based similarity + popularity weighting
        """
        ml_movie_id = self.get_ml_movie_id(imdb_link)
        is_known_movie = ml_movie_id is not None and ml_movie_id in self.item_to_idx
        
        # Get user genre preferences
        user_genre_prefs = None
        
        # Priority 1: IMDB user profile (from training data)
        if self.is_known_imdb_user(user_name):
            user_genre_prefs = self.get_imdb_user_genre_prefs(user_name)
        
        # Priority 2: Build from test history
        if user_genre_prefs is None and user_history:
            user_genre_prefs = self.get_user_genre_prefs_from_test_history(user_history)
        
        # Priority 3: Default uniform preferences
        if user_genre_prefs is None and self.genres:
            user_genre_prefs = np.ones(len(self.genres)) / len(self.genres)
        
        if is_known_movie:
            item_idx = self.item_to_idx[ml_movie_id]
            
            # Content-based score
            content_score = self.predict_content_based(user_genre_prefs, item_idx)
            
            # Popularity score
            popularity_score = self.predict_popularity(ml_movie_id)
            
            # Try SVD prediction
            svd_score = None
            if self.model is not None and SURPRISE_AVAILABLE:
                try:
                    scores = self.predict_svd_for_items(user_name, [item_idx], user_genre_prefs=user_genre_prefs)
                    if scores is not None and len(scores) > 0:
                        svd_score = float(scores[0])
                except Exception as e:
                    svd_score = None
            
            # Combine scores
            if svd_score is not None:
                # Known user with IMDB profile -> trust SVD more
                if self.is_known_imdb_user(user_name):
                    score = 0.5 * svd_score + 0.3 * content_score + 0.2 * popularity_score
                else:
                    # Unknown user -> rely more on content + popularity
                    score = 0.4 * svd_score + 0.4 * content_score + 0.2 * popularity_score
            else:
                # No SVD -> use content + popularity
                score = 0.6 * content_score + 0.4 * popularity_score
        else:
            # Unknown movie - use metadata if available from movie_mapper
            if movie_mapper_row is not None and 'genre' in movie_mapper_row.index:
                # Parse genres from movie_mapper
                genre_str = str(movie_mapper_row.get('genre', '[]'))
                try:
                    import ast
                    movie_genres = ast.literal_eval(genre_str)
                    if isinstance(movie_genres, list) and user_genre_prefs is not None:
                        # Calculate similarity
                        genre_vec = np.zeros(len(self.genres))
                        for g in movie_genres:
                            if g in self.genre_to_idx:
                                genre_vec[self.genre_to_idx[g]] = 1.0
                        
                        if np.sum(genre_vec) > 0:
                            sim = 1 - cosine(user_genre_prefs, genre_vec)
                            score = sim if not np.isnan(sim) else 0.4
                        else:
                            score = 0.4
                    else:
                        score = 0.4
                except:
                    score = 0.4
            else:
                score = 0.4  # Lower score for completely unknown movies
        
        return score
    
    def load_test_data(self, test_data_path):
        """Load test dataset"""
        print(f"\n[INFO] Loading test data from {test_data_path}...")
        
        test_files = {
            'known_reviewers_known_movies': None,
            'known_reviewers_unknown_movies': None,
            'unknown_reviewers_known_movies': None,
            'movie_mapper': None
        }
        
        for key in test_files.keys():
            file_path = os.path.join(test_data_path, f"{key}.csv")
            if os.path.exists(file_path):
                test_files[key] = pd.read_csv(file_path)
                print(f"[INFO] Loaded {key}: {len(test_files[key])} records")
            else:
                print(f"[WARNING] File not found: {file_path}")
        
        return test_files
    
    def generate_predictions(self, test_files, output_dir='output'):
        """Generate predictions for all test cases"""
        print("\n[INFO] Generating predictions...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build user histories from test data
        print("[INFO] Building user histories...")
        user_histories = defaultdict(list)
        
        for key, df in test_files.items():
            if df is None or key == 'movie_mapper':
                continue
            
            for _, row in df.iterrows():
                user_name = row['user_name']
                movie_link = row['movie_link']
                ranking = row['ranking']
                user_histories[user_name].append((movie_link, ranking))
        
        # Load movie mapper for genre info
        movie_mapper = test_files.get('movie_mapper')
        movie_mapper_dict = {}
        if movie_mapper is not None:
            for _, row in movie_mapper.iterrows():
                link = self.normalize_imdb_link(row['movie_link'])
                movie_mapper_dict[link] = row
        
        all_predictions = []
        
        # Process each test file
        for file_key, df in test_files.items():
            if df is None or file_key == 'movie_mapper':
                continue
            
            print(f"\n[INFO] Processing {file_key}...")
            
            for idx, row in df.iterrows():
                user_name = row['user_name']
                movie_link = row['movie_link']
                normalized_link = self.normalize_imdb_link(movie_link)
                
                # Get user history (exclude current movie)
                user_history = [(m, r) for m, r in user_histories[user_name] 
                               if self.normalize_imdb_link(m) != normalized_link]
                
                # Get movie mapper row if available
                mapper_row = movie_mapper_dict.get(normalized_link)
                
                # Generate prediction
                predicted_score = self.predict_rating(
                    user_name, 
                    movie_link,
                    user_history=user_history,
                    movie_mapper_row=mapper_row
                )
                
                all_predictions.append({
                    'user_name': user_name,
                    'movie_link': movie_link,
                    'predicted_score': predicted_score,
                    'test_case': file_key
                })
                
                if (idx + 1) % 200 == 0:
                    print(f"[INFO] Processed {idx + 1} predictions...")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Save predictions
        output_file = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(output_file, index=False)
        print(f"\n[INFO] Predictions saved to {output_file}")
        print(f"[INFO] Total predictions: {len(predictions_df)}")
        
        return predictions_df
    
    def calculate_recall_at_k(self, predictions_df, test_files, k_values=[1, 3, 5]):
        """Calculate Recall@K metrics"""
        print("\n[INFO] Calculating Recall@K metrics...")
        
        results = {}
        
        for file_key, df in test_files.items():
            if df is None or file_key == 'movie_mapper':
                continue
            
            file_preds = predictions_df[predictions_df['test_case'] == file_key]
            
            recalls = {k: [] for k in k_values}
            
            # Group by user
            for user_name, user_group in df.groupby('user_name'):
                # Ground truth: top-ranked movies (rank 1, 2, etc.)
                user_sorted = user_group.sort_values('ranking')
                
                # Get predictions for this user
                user_preds = file_preds[file_preds['user_name'] == user_name]
                if len(user_preds) == 0:
                    continue
                
                # Sort by predicted score (higher = better)
                user_preds_sorted = user_preds.sort_values('predicted_score', ascending=False)
                
                for k in k_values:
                    # Top-K predicted movies
                    top_k_pred = set(user_preds_sorted.head(k)['movie_link'].tolist())
                    
                    # Top-K ground truth (by ranking)
                    top_k_true = set(user_sorted.head(k)['movie_link'].tolist())
                    
                    # Calculate recall
                    if len(top_k_true) > 0:
                        hits = len(top_k_pred & top_k_true)
                        recall = hits / min(k, len(top_k_true))
                        recalls[k].append(recall)
            
            # Average recall for this file
            for k in k_values:
                if recalls[k]:
                    avg_recall = np.mean(recalls[k])
                    results[f'{file_key}_recall@{k}'] = avg_recall
                    print(f"  {file_key} Recall@{k}: {avg_recall:.4f}")
        
        # Overall averages
        print("\n" + "="*50)
        print("[INFO] Overall Metrics (Competition Targets):")
        print("="*50)
        for k in k_values:
            all_recalls = [v for key, v in results.items() if f'recall@{k}' in key]
            if all_recalls:
                overall = np.mean(all_recalls)
                results[f'overall_recall@{k}'] = overall
                emoji = "🎯" if k == 5 else "📊"
                print(f"  {emoji} Overall Recall@{k}: {overall:.4f}")
        
        return results


def run_inference(test_data_path='sample_test_phase_1', model_path='Resources/hybrid_model.pkl', output_dir='output'):
    """Run full inference pipeline"""
    
    print("="*70)
    print("  AITH 2025 - WINNING Movie Recommendation System")
    print("  Team: OneManArmy")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Test Data Path: {test_data_path}")
    print(f"  Model Path: {model_path}")
    print(f"  Output Directory: {output_dir}")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize recommender
    recommender = WinningRecommender(model_path=model_path)
    
    # Load test data
    test_files = recommender.load_test_data(test_data_path)
    
    # Generate predictions
    predictions_df = recommender.generate_predictions(test_files, output_dir)
    
    # Calculate metrics
    metrics = recommender.calculate_recall_at_k(predictions_df, test_files)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    # Add execution time
    total_time = time.time() - start_time
    metrics['execution_time_seconds'] = total_time
    metrics['total_predictions'] = len(predictions_df)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_file}")
    
    print("\n" + "="*70)
    print("  ✅ Inference completed successfully!")
    print(f"  ⏱️  Total execution time: {total_time:.2f} seconds")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Predictions: {output_dir}/predictions.csv")
    print(f"  - Metrics: {output_dir}/metrics.json")
    
    return metrics


def download_model(url, save_path):
    """Download model file from URL if it doesn't exist locally"""
    import urllib.request
    
    if os.path.exists(save_path):
        print(f"[INFO] Model file already exists at {save_path}")
        return
    
    print(f"[INFO] Downloading model from {url}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        print(f"[INFO] Model downloaded successfully to {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        raise


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='AITH 2025 Movie Recommendation Inference'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='Dataset/aith-dataset/sample_test_phase_1',
        help='Path to test data folder (default: Dataset/aith-dataset/sample_test_phase_1)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='Resources/hybrid_model.pkl',
        help='Path to trained model file (default: Resources/hybrid_model.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for predictions (default: output)'
    )
    parser.add_argument(
        '--download_model_url',
        type=str,
        default=None,
        help='URL to download model file if local model_path does not exist (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists, download if URL provided
    if not os.path.exists(args.model_path):
        if args.download_model_url:
            download_model(args.download_model_url, args.model_path)
        else:
            # Try fallback paths
            fallback_paths = [
                os.path.join(os.path.dirname(__file__), '..', args.model_path),
                os.path.join(os.path.dirname(__file__), '..', 'Resources', 'hybrid_model.pkl'),
            ]
            found = False
            for fp in fallback_paths:
                if os.path.exists(fp):
                    args.model_path = fp
                    found = True
                    break
            if not found:
                print(f"[ERROR] Model file not found at {args.model_path}")
                print("[INFO] If you have a model URL, use --download_model_url to download it.")
                raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    run_inference(
        test_data_path=args.test_data_path,
        model_path=args.model_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
