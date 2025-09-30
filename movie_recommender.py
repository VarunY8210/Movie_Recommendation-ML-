# Movie Recommendation System
# Requires: ratings.csv and movies.csv inside a 'data/' folder
# Run: python movie_recommender.py

import pandas as pd
import numpy as np
from pathlib import Path
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

# ------------------------------
# File paths
# ------------------------------
DATA_DIR = Path("data")
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"

# ------------------------------
# Load movie and rating data
# ------------------------------
def load_data():
    if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
        raise FileNotFoundError("Place ratings.csv and movies.csv in a 'data/' folder.")
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    return ratings, movies

# ------------------------------
# Train collaborative filtering model
# ------------------------------
def train_svd(ratings):
    reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"SVD RMSE: {rmse:.4f}")
    return algo, predictions

# ------------------------------
# Build TF-IDF matrix for content-based filtering
# ------------------------------
def build_tfidf(movies):
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    movies["content"] = movies["title"].astype(str) + " " + movies["genres"].str.replace("|", " ", regex=False)
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    return tfidf, tfidf_matrix, movies

# ------------------------------
# Precision@K and Recall@K metrics
# ------------------------------
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []

    for user_ratings in user_est_true.values():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)

        if n_rec_k > 0:
            precisions.append(n_rel_and_rec_k / n_rec_k)
        if n_rel > 0:
            recalls.append(n_rel_and_rec_k / n_rel)

    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0

# ------------------------------
# Hybrid recommendation (CF + content)
# ------------------------------
def hybrid_recommend(algo, tfidf_matrix, movies_df, ratings_df, user_id, top_k=10, alpha=0.7):
    all_movie_ids = movies_df["movieId"].values
    collab_scores = np.array([algo.predict(user_id, mid).est for mid in all_movie_ids])

    # Content-based filtering
    user_rated = ratings_df[ratings_df.userId == user_id]
    content_scores = np.zeros(len(all_movie_ids))

    if not user_rated.empty:
        liked_movies = user_rated[user_rated.rating >= 4.0]["movieId"].values
        idx_map = {movie_id: idx for idx, movie_id in enumerate(movies_df["movieId"].values)}
        liked_indices = [idx_map[mid] for mid in liked_movies if mid in idx_map]

        if liked_indices:
            user_profile = tfidf_matrix[liked_indices].mean(axis=0)
            user_profile = np.asarray(user_profile)  # fix: convert to ndarray
            content_scores = linear_kernel(user_profile, tfidf_matrix).flatten()

    # Normalize scores
    collab_norm = collab_scores / (collab_scores.max() or 1)
    content_norm = content_scores / (content_scores.max() or 1)

    # Combine scores
    hybrid_scores = alpha * collab_norm + (1 - alpha) * content_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return movies_df.iloc[top_indices][["movieId", "title"]]

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    ratings, movies = load_data()
    algo, predictions = train_svd(ratings)
    tfidf, tfidf_matrix, processed_movies = build_tfidf(movies)
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=4.0)

    print(f"Precision@10: {precision:.4f}, Recall@10: {recall:.4f}")

    sample_user = int(ratings["userId"].sample(1, random_state=42).iloc[0])
    print(f"Sample hybrid recommendations for user {sample_user}:")
    recommended = hybrid_recommend(algo, tfidf_matrix, processed_movies, ratings, sample_user, top_k=10, alpha=0.7)
    print(recommended.to_string(index=False))
