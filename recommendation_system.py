import pandas as pd
import numpy as np
import random
from faker import Faker
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, MetaData
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# --- Configuration ---
DB_NAME = 'recommendation_system.db'
NUM_USERS = 300
NUM_MOVIES = 100
NUM_ACTIVITIES = 1500
RANDOM_SEED = 42 # for reproducibility

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

# --- SQLAlchemy Setup ---
engine = create_engine(f'sqlite:///{DB_NAME}')
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Models ---
class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    age = Column(Integer)
    # Relationships
    activities = relationship("Activity", back_populates="user")

class Movie(Base):
    __tablename__ = 'movies'
    movie_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    genre = Column(String)
    # Relationships
    activities = relationship("Activity", back_populates="movie")

class Activity(Base):
    __tablename__ = 'activities'
    activity_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    movie_id = Column(Integer, ForeignKey('movies.movie_id'), nullable=False)
    rating = Column(Float, nullable=False) # e.g., 1.0 to 5.0
    # Relationships
    user = relationship("User", back_populates="activities")
    movie = relationship("Movie", back_populates="activities")

# --- Data Generation ---
def generate_data():
    """Generates synthetic data for users, movies, and activities."""
    users_data = []
    usernames = set()
    while len(users_data) < NUM_USERS:
        name = fake.user_name()
        if name not in usernames:
            users_data.append({
                'user_id': len(users_data) + 1,
                'username': name,
                'age': random.randint(15, 70)
            })
            usernames.add(name)

    movies_data = []
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller', 'Animation']
    for i in range(NUM_MOVIES):
        movies_data.append({
            'movie_id': i + 1,
            'title': fake.catch_phrase() + " " + random.choice(['Movie', 'Story', 'Saga', 'Chronicles']),
            'genre': random.choice(genres)
        })

    activities_data = []
    user_movie_pairs = set()
    attempts = 0
    max_attempts = NUM_ACTIVITIES * 5 # Limit attempts to avoid infinite loop

    while len(activities_data) < NUM_ACTIVITIES and attempts < max_attempts:
        user_id = random.randint(1, NUM_USERS)
        movie_id = random.randint(1, NUM_MOVIES)
        pair = (user_id, movie_id)

        if pair not in user_movie_pairs:
            activities_data.append({
                'activity_id': len(activities_data) + 1,
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': round(random.uniform(1.0, 5.0), 1)
            })
            user_movie_pairs.add(pair)
        attempts += 1

    if len(activities_data) < NUM_ACTIVITIES:
        print(f"Warning: Could only generate {len(activities_data)} unique activities out of {NUM_ACTIVITIES} requested.")


    return users_data, movies_data, activities_data

# --- Database Operations ---
def create_database_and_tables():
    """Creates the database and tables."""
    Base.metadata.create_all(bind=engine)
    print(f"Database '{DB_NAME}' and tables created.")

def populate_database(users_data, movies_data, activities_data):
    """Populates the database with generated data."""
    db = SessionLocal()
    try:
        # Bulk insert users
        if users_data:
            db.bulk_insert_mappings(User, users_data)
            print(f"Inserted {len(users_data)} users.")

        # Bulk insert movies
        if movies_data:
            db.bulk_insert_mappings(Movie, movies_data)
            print(f"Inserted {len(movies_data)} movies.")

        # Bulk insert activities
        if activities_data:
             # Insert activities one by one to handle potential IntegrityErrors more easily if needed,
             # though bulk_insert_mappings is generally faster.
             # For this scale, bulk insert is fine.
            db.bulk_insert_mappings(Activity, activities_data)
            print(f"Inserted {len(activities_data)} activities.")

        db.commit()
    except IntegrityError as e:
        db.rollback()
        print(f"Database Error: {e}. Rolling back changes.")
        # Consider more granular error handling if needed
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred: {e}. Rolling back changes.")
    finally:
        db.close()

def load_data_to_dataframe():
    """Loads activity data from the database into a Pandas DataFrame."""
    db = SessionLocal()
    try:
        # Query activities table
        query = db.query(Activity.user_id, Activity.movie_id, Activity.rating)
        df = pd.read_sql(query.statement, db.bind)
        print(f"Loaded {len(df)} activities into DataFrame.")
        return df
    except Exception as e:
        print(f"Error loading data to DataFrame: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    finally:
        db.close()

# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Create DB and Tables
    create_database_and_tables()

    # 2. Generate Data
    users, movies, activities = generate_data()

    # 3. Populate Database
    populate_database(users, movies, activities)

    # 4. Load Data into Pandas DataFrame
    activity_df = load_data_to_dataframe()

    if not activity_df.empty:
        print("Activity DataFrame Head:")
        print(activity_df.head())

        # --- Data Preparation for K-Means ---
        print("Pivoting data to User-Movie matrix...")
        # Create user-movie matrix
        user_movie_matrix = activity_df.pivot_table(index='user_id', columns='movie_id', values='rating')

        # Fill NaN values (users who haven't rated a movie) with 0
        # Alternatively, consider user average or movie average, but 0 is common for sparsity
        user_movie_matrix.fillna(0, inplace=True)

        print("User-Movie Matrix Head:")
        print(user_movie_matrix.head())

        # --- K-Means Clustering ---
        print("Applying K-Means clustering...")
        # Standardize the data (important for distance-based algorithms like K-Means)
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(user_movie_matrix)

        # Determine the optimal number of clusters (Elbow Method - Optional but recommended)
        # For simplicity, we'll choose a fixed number of clusters, e.g., 10
        k = 10
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10) # n_init to suppress warning
        kmeans.fit(scaled_matrix)

        # --- Save the Model and Scaler ---
        print("\nSaving K-Means model and scaler...")
        model_filename = 'kmeans_model.pkl'
        scaler_filename = 'scaler.pkl'
        try:
            joblib.dump(kmeans, model_filename)
            joblib.dump(scaler, scaler_filename)
            print(f"Model saved to {model_filename}")
            print(f"Scaler saved to {scaler_filename}")
        except Exception as e:
            print(f"Error saving model/scaler: {e}")
        # --- End Save ---

        # Add cluster labels to the original matrix (optional, for analysis)
        user_movie_matrix['cluster'] = kmeans.labels_
        print(f"\nAssigned users to {k} clusters.")
        print("User-Movie Matrix with Cluster Labels Head:")
        print(user_movie_matrix.head())

        # --- Save User-Cluster Mapping ---
        print("\nSaving user-cluster mapping...")
        user_cluster_map = user_movie_matrix[['cluster']].reset_index()
        user_cluster_filename = 'user_clusters.pkl'
        try:
            joblib.dump(user_cluster_map, user_cluster_filename)
            print(f"User-cluster map saved to {user_cluster_filename}")
        except Exception as e:
            print(f"Error saving user-cluster map: {e}")
        # --- End Save ---

        # --- Visualization ---
        print("\nVisualizing clusters using PCA...")
        # Reduce dimensions using PCA for visualization
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        reduced_data = pca.fit_transform(scaled_matrix)

        # Create a DataFrame for plotting
        pca_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'], index=user_movie_matrix.index)
        pca_df['cluster'] = kmeans.labels_

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='viridis', s=50, alpha=0.7)
        plt.title(f'User Clusters based on Movie Ratings (K={k}, PCA reduced)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('user_clusters_pca.png') # Save the plot
        print("Cluster visualization saved to 'user_clusters_pca.png'")
        # plt.show() # Uncomment to display the plot directly

        # --- Basic Recommendation Example ---
        def recommend_movies_for_user(user_id, n_recommendations=5):
            if user_id not in user_movie_matrix.index:
                print(f"User ID {user_id} not found.")
                return []

            # Get the cluster for the user
            user_cluster = user_movie_matrix.loc[user_id, 'cluster']

            # Get all users in the same cluster
            users_in_cluster = user_movie_matrix[user_movie_matrix['cluster'] == user_cluster].index.tolist()
            users_in_cluster.remove(user_id) # Remove the target user

            if not users_in_cluster:
                print(f"No other users found in the same cluster as user {user_id}.")
                return []

            # Get movies rated by users in the same cluster
            cluster_activity = activity_df[activity_df['user_id'].isin(users_in_cluster)]

            # Calculate average rating for each movie within the cluster
            movie_avg_ratings = cluster_activity.groupby('movie_id')['rating'].mean()

            # Get movies the target user has *not* rated
            user_rated_movies = activity_df[activity_df['user_id'] == user_id]['movie_id'].unique()
            
            # Filter recommendations: high average rating in cluster, not seen by user
            recommendations = movie_avg_ratings[~movie_avg_ratings.index.isin(user_rated_movies)]
            
            # Sort by rating and get top N
            top_n = recommendations.sort_values(ascending=False).head(n_recommendations).index.tolist()

            return top_n

        # Example: Recommend movies for user 1
        target_user_id = 1
        recommended_movie_ids = recommend_movies_for_user(target_user_id)

        if recommended_movie_ids:
             # Fetch movie titles (requires querying the Movie table again or loading it earlier)
             db = SessionLocal()
             try:
                 recommended_titles = db.query(Movie.title).filter(Movie.movie_id.in_(recommended_movie_ids)).all()
                 recommended_titles = [title[0] for title in recommended_titles] # Extract titles from tuples
                 print(f"Top {len(recommended_movie_ids)} movie recommendations for User {target_user_id}:")
                 for title in recommended_titles:
                     print(f"- {title}")
             finally:
                 db.close()
        else:
             print(f"Could not generate recommendations for User {target_user_id}.")

    else:
        print("Activity DataFrame is empty. Cannot proceed with clustering and recommendations.") 