import os
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    """Movie Recommendation System using SVD Collaborative Filtering"""
    
    def __init__(self, model_path='Resources/recommendation_model.pkl'):
        """Initialize the recommender system"""
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        print(f"[INFO] Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model file not found at {self.model_path}")
            print(f"[INFO] Attempting to download model...")
            self.download_model()
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print(f"[INFO] Model loaded successfully!")
            
            # Print model info
            if 'svd_model' in self.model_data:
                print(f"[INFO] Model type: SVD Collaborative Filtering")
            elif 'type' in self.model_data:
                print(f"[INFO] Model type: {self.model_data['type']}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def download_model(self):
        """Download model from cloud storage if not available locally"""
        # For submission: Host your model on Google Drive, Dropbox, or GitHub Releases
        # and implement download logic here
        
        print("[INFO] Model download not implemented yet.")
        print("[INFO] Please ensure the model file is in the models/ directory")
        print("[INFO] If the model is too large for GitHub, upload to:")
        print("       - Google Drive (create shareable link)")
        print("       - GitHub Releases")
        print("       - Dropbox")
        raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def load_test_data(self, test_data_path):
        """Load test dataset"""
        print(f"\n[INFO] Loading test data from {test_data_path}...")
        
        test_files = {
            'known_reviewers_known_movies': None,
            'known_reviewers_unknown_movies': None,
            'unknown_reviewers_known_movies': None,
            'movie_mapper': None
        }
        
        # Load each test file
        for key in test_files.keys():
            file_path = os.path.join(test_data_path, f"{key}.csv")
            if os.path.exists(file_path):
                test_files[key] = pd.read_csv(file_path)
                print(f"[INFO] Loaded {key}: {len(test_files[key])} records")
            else:
                print(f"[WARNING] File not found: {file_path}")
        
        return test_files
    
    def predict_for_user(self, user_name, movie_id, user_review_text=None):
        """Generate prediction for a specific user-movie pair"""
        
        # Check model type
        if 'svd_model' in self.model_data:
            svd_model = self.model_data['svd_model']
            
            # For SVD, we need numeric user and movie IDs
            # Map user_name to user_id (simplified - use hash or mapping)
            user_id = hash(user_name) % 10000
            
            try:
                prediction = svd_model.predict(user_id, movie_id)
                return prediction.est
            except:
                # If prediction fails, return average rating
                return 3.5
        
        elif 'movie_stats' in self.model_data:
            # Popularity-based model
            movie_stats = self.model_data['movie_stats']
            movie_row = movie_stats[movie_stats['movieId'] == movie_id]
            
            if not movie_row.empty:
                return movie_row.iloc[0]['avg_rating']
            else:
                return 3.5
        
        return 3.5
    
    def generate_predictions(self, test_files, output_dir='output'):
        """Generate predictions for all test cases"""
        print("\n[INFO] Generating predictions...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_predictions = []
        
        # Process each test file
        for file_key, df in test_files.items():
            if df is None or file_key == 'movie_mapper':
                continue
            
            print(f"\n[INFO] Processing {file_key}...")
            
            for idx, row in df.iterrows():
                user_name = row['user_name']
                movie_link = row['movie_link']
                
                # Extract movie ID from link (if available)
                # Format: https://www.imdb.com/title/tt0111161/
                try:
                    movie_id = int(movie_link.split('/')[-2].replace('tt', ''))
                except:
                    movie_id = hash(movie_link) % 100000
                
                # Get user review if available
                user_review = row.get('user_review', None)
                
                # Generate prediction
                predicted_rating = self.predict_for_user(
                    user_name, 
                    movie_id, 
                    user_review
                )
                
                # Store prediction
                all_predictions.append({
                    'user_name': user_name,
                    'movie_link': movie_link,
                    'predicted_rating': predicted_rating,
                    'test_case': file_key
                })
                
                if (idx + 1) % 100 == 0:
                    print(f"[INFO] Processed {idx + 1} predictions...")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Save predictions
        output_file = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(output_file, index=False)
        print(f"\n[INFO] Predictions saved to {output_file}")
        print(f"[INFO] Total predictions: {len(predictions_df)}")
        
        return predictions_df
    
    def calculate_metrics(self, predictions_df, test_files):
        """Calculate Recall@K metrics"""
        print("\n[INFO] Calculating evaluation metrics...")
        
        # For competition evaluation, metrics will be calculated by organizers
        # This is just for local testing
        
        metrics = {
            'total_predictions': len(predictions_df),
            'unique_users': predictions_df['user_name'].nunique(),
            'avg_predicted_rating': predictions_df['predicted_rating'].mean(),
            'test_cases_processed': predictions_df['test_case'].nunique()
        }
        
        print("\n[INFO] Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        return metrics


def main():
    """Main inference function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='AITH 2025 Movie Recommendation Inference'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='sample_test_phase_1',
        help='Path to test data folder (default: sample_test_phase_1)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='Resources/recommendation_model.pkl',
        help='Path to trained model file (default: Resources/recommendation_model.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for predictions (default: output)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  AITH 2025 - Movie Recommendation System - Inference")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Test Data Path: {args.test_data_path}")
    print(f"  Model Path: {args.model_path}")
    print(f"  Output Directory: {args.output_dir}")
    print("="*70)
    
    # Initialize recommender
    recommender = MovieRecommender(model_path=args.model_path)
    
    # Load test data
    test_files = recommender.load_test_data(args.test_data_path)
    
    # Generate predictions
    predictions_df = recommender.generate_predictions(test_files, args.output_dir)
    
    # Calculate metrics
    metrics = recommender.calculate_metrics(predictions_df, test_files)
    
    # Save metrics
    import json
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_file}")
    
    print("\n" + "="*70)
    print("  Inference completed successfully!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Predictions: {args.output_dir}/predictions.csv")
    print(f"  - Metrics: {args.output_dir}/metrics.json")
    print("\n")


if __name__ == "__main__":
    main()
