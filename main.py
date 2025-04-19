import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Assuming recommendation_system.py defined these models and SessionLocal
# If not run from the same directory, adjust paths or imports
try:
    # Try importing from the existing script if they are in the same directory
    from recommendation_system import SessionLocal, Movie, Activity, DB_NAME
except ImportError:
    # Fallback or define necessary parts if import fails
    print("Warning: Could not import from recommendation_system.py. Ensure DB models and session are available.")
    # Add minimal definitions if needed for the script to run, or raise an error
    # For simplicity, we assume the DB exists and SessionLocal can be created.
    if 'DB_NAME' not in locals(): DB_NAME = 'recommendation_system.db'
    if not os.path.exists(DB_NAME):
         raise FileNotFoundError(f"Database file '{DB_NAME}' not found. Please run recommendation_system.py first.")
    engine = create_engine(f'sqlite:///{DB_NAME}')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # Minimal model definitions might be needed if import fails
    from sqlalchemy import Column, Integer, String, Float, ForeignKey
    from sqlalchemy.orm import relationship, declarative_base
    Base = declarative_base()
    class Movie(Base):
        __tablename__ = 'movies'
        movie_id = Column(Integer, primary_key=True)
        title = Column(String)
    class Activity(Base):
         __tablename__ = 'activities'
         activity_id = Column(Integer, primary_key=True)
         user_id = Column(Integer, ForeignKey('users.user_id'))
         movie_id = Column(Integer, ForeignKey('movies.movie_id'))
         rating = Column(Float)


# --- Configuration & Model Loading ---
MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
USER_CLUSTERS_PATH = "user_clusters.pkl"

kmeans_model = None
scaler = None
user_cluster_map = None
activity_df = None
movie_titles = None

def load_resources():
    """Load necessary resources: model, scaler, user clusters, activities, movie titles."""
    global kmeans_model, scaler, user_cluster_map, activity_df, movie_titles

    print("Loading resources...")
    try:
        kmeans_model = joblib.load(MODEL_PATH)
        print(f"- Model loaded from {MODEL_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print(f"- Scaler loaded from {SCALER_PATH}")
        user_cluster_map = joblib.load(USER_CLUSTERS_PATH)
        print(f"- User clusters loaded from {USER_CLUSTERS_PATH}")

        # Load activities and movie titles from DB
        db = SessionLocal()
        try:
            # Load activities
            query_activity = db.query(Activity.user_id, Activity.movie_id, Activity.rating)
            activity_df = pd.read_sql(query_activity.statement, db.bind)
            print(f"- Loaded {len(activity_df)} activities from DB.")

            # Load movie titles into a dictionary {movie_id: title}
            query_movies = db.query(Movie.movie_id, Movie.title)
            movies_data = query_movies.all()
            movie_titles = {m_id: title for m_id, title in movies_data}
            print(f"- Loaded {len(movie_titles)} movie titles from DB.")

        finally:
            db.close()

    except FileNotFoundError as e:
        print(f"Error loading resources: {e}. Make sure recommendation_system.py has been run successfully.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during resource loading: {e}")
        raise
    print("Resources loaded successfully.")

# --- FastAPI App ---
app = FastAPI(
    title="Movie Recommender API",
    description="API to get movie recommendations based on user clustering.",
    version="0.1.0"
)

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[str]
    message: str | None = None

@app.on_event("startup")
def startup_event():
    load_resources()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommender API"}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, n_recommendations: int = 5):
    """Get movie recommendations for a given user_id."""
    if not all([kmeans_model, scaler, user_cluster_map is not None, activity_df is not None, movie_titles]):
        raise HTTPException(status_code=503, detail="Resources not loaded. API is not ready.")

    # Check if user_id exists in our cluster map
    if user_id not in user_cluster_map['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in the dataset used for training.")

    try:
        # Get the cluster for the target user
        user_cluster_info = user_cluster_map[user_cluster_map['user_id'] == user_id]
        if user_cluster_info.empty:
             # This case should theoretically not happen due to the check above, but good practice
             raise HTTPException(status_code=404, detail=f"Could not determine cluster for User ID {user_id}.")
        user_cluster = user_cluster_info['cluster'].iloc[0]

        # Get all user IDs in the same cluster
        users_in_cluster = user_cluster_map[user_cluster_map['cluster'] == user_cluster]['user_id'].tolist()
        # Remove the target user from the list of similar users
        if user_id in users_in_cluster:
            users_in_cluster.remove(user_id)

        if not users_in_cluster:
            return RecommendationResponse(
                user_id=user_id,
                recommendations=[],
                message=f"No other users found in the same cluster as user {user_id} to generate recommendations."
            )

        # Get movies rated by users in the same cluster
        cluster_activity = activity_df[activity_df['user_id'].isin(users_in_cluster)]

        if cluster_activity.empty:
             return RecommendationResponse(
                user_id=user_id,
                recommendations=[],
                message=f"No activity found for other users in cluster {user_cluster}."
            )

        # Calculate average rating for each movie within the cluster
        movie_avg_ratings = cluster_activity.groupby('movie_id')['rating'].mean()

        # Get movies the target user *has* rated
        user_rated_movies = activity_df[activity_df['user_id'] == user_id]['movie_id'].unique()

        # Filter recommendations: high average rating in cluster, not seen by user
        recommendations_filtered = movie_avg_ratings[~movie_avg_ratings.index.isin(user_rated_movies)]

        # Sort by rating and get top N movie IDs
        recommended_movie_ids = recommendations_filtered.sort_values(ascending=False).head(n_recommendations).index.tolist()

        # Get movie titles for the recommended IDs
        recommended_titles = [movie_titles.get(m_id, f"Unknown Movie ID: {m_id}") for m_id in recommended_movie_ids]

        if not recommended_titles:
             return RecommendationResponse(
                user_id=user_id,
                recommendations=[],
                message=f"Could not find new movies to recommend for user {user_id} based on their cluster."
            )

        return RecommendationResponse(user_id=user_id, recommendations=recommended_titles)

    except Exception as e:
        print(f"Error during recommendation generation for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while generating recommendations.")

# --- Run the App ---
if __name__ == "__main__":
    # Ensure resources are loaded before starting the server if not run via `uvicorn main:app`
    # load_resources() # Usually handled by @app.on_event("startup") when run with uvicorn

    print("Starting FastAPI server...")
    print(f"Access the API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 