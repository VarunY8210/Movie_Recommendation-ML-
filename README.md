# 🎬 Movie Recommendation System using Machine Learning

A machine learning project that builds a **Movie Recommendation System** using the [MovieLens Small Dataset](https://www.kaggle.com/grouplens/movielens-small).

The system implements:
- 🎯 **Collaborative Filtering** (SVD matrix factorization using `Surprise`)
- 🧠 **Content-Based Filtering** (TF-IDF on title + genres + cosine similarity)
- 🔀 **Hybrid Recommendations** (Weighted combination of both methods)

---

## 📂 Project Structure

```bash
Movie_Recomendation/
│
├── data/
│   ├── ratings.csv         # User ratings of movies
│   └── movies.csv          # Movie titles and genres
│
├── movie_recommender.py    # Main script for training and recommending
├── requirements.txt        # All required Python libraries
└── README.md               # Project documentation
```
## 📊 Dataset Info

- **Source**: [MovieLens 100k Small Dataset](https://www.kaggle.com/grouplens/movielens-small)
- **Files**:
  - `ratings.csv`: 100,000 ratings from 610 users on over 9,000 movies
  - `movies.csv`: Metadata including movie titles and genres
---
## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Movie_Recomendation.git
cd Movie_Recomendation
```