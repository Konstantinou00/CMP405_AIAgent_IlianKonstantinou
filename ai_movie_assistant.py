
import pandas as pd
import numpy as np
import joblib
import faiss
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Letterbox-Movie-Classification-Dataset.csv")
df.drop(columns=["Unnamed: 0"], errors='ignore', inplace=True)

# Fix stringified lists
for col in ["Genres", "Studios"]:
    df[col] = df[col].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) and x.startswith("[") else x)

# Feature engineering
df["likes_per_rating"] = df["Likes"] / (df["Total_ratings"] + 1)
df["fans_per_watch"] = df["Fans"] / (df["Watches"] + 1)
df["likes_per_fan"] = df["Likes"] / (df["Fans"] + 1)

# Year extraction
if "Release_date" in df.columns:
    df["Year"] = pd.to_datetime(df["Release_date"], errors='coerce').dt.year

# Limit high cardinality for Director
top_directors = df["Director"].value_counts().nlargest(50).index
df["Director"] = df["Director"].apply(lambda x: x if x in top_directors else "Other")

# Load trained model
model = joblib.load("random_forest_movie_rating_model.pkl")

# SentenceTransformer and FAISS
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
desc_embeddings = embedder.encode(df["Description"].fillna(""), show_progress_bar=True)

embedding_dim = desc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(np.array(desc_embeddings))

# Prepare fuzzy sets
all_genres = list({g.strip().lower() for sublist in df["Genres"].str.lower().str.split(",") for g in sublist})
all_languages = df["Original_language"].str.lower().unique()
all_titles = df["Film_title"].fillna("").unique()

def fuzzy_match(input_value, choices, threshold=80):
    if not input_value:
        return ""
    match, score = process.extractOne(input_value, choices)
    return match if score >= threshold else input_value

def collect_user_input():
    print("\n🎥 Let's find the perfect movie for you!")
    title = input("Do you have a movie title in mind? ").strip()
    description = input("Describe a movie or concept you like (or leave blank): ").strip()
    genres = input("Preferred genres (comma separated): ").strip().lower().split(",")
    language = input("Preferred language (optional): ").strip().lower()
    min_runtime = input("Minimum runtime (minutes, optional): ").strip()
    max_runtime = input("Maximum runtime (minutes, optional): ").strip()
    min_year = input("Earliest release year (optional): ").strip()
    max_year = input("Latest release year (optional): ").strip()

    genres = [fuzzy_match(g.strip(), all_genres) for g in genres if g.strip()]
    language = fuzzy_match(language, all_languages) if language else None
    title = fuzzy_match(title, all_titles) if title else ""

    return {
        "Film_title": title,
        "Description": description,
        "Genres": genres,
        "Original_language": language,
        "min_runtime": float(min_runtime) if min_runtime else None,
        "max_runtime": float(max_runtime) if max_runtime else None,
        "min_year": int(min_year) if min_year else None,
        "max_year": int(max_year) if max_year else None,
    }

def embed_user_text(pref):
    text = f"{pref.get('Film_title','')} {pref.get('Description','')}".strip()
    return embedder.encode([text]) if text else None

def filter_movies(prefs):
    f = df.copy()
    if prefs["Genres"]:
        f = f[f["Genres"].str.lower().apply(lambda x: any(g in x for g in prefs["Genres"]))]
    if prefs["Original_language"]:
        f = f[f["Original_language"].str.lower() == prefs["Original_language"]]
    if prefs["min_runtime"]:
        f = f[f["Runtime"] >= prefs["min_runtime"]]
    if prefs["max_runtime"]:
        f = f[f["Runtime"] <= prefs["max_runtime"]]
    if "Year" in f.columns:
        if prefs["min_year"]:
            f = f[f["Year"] >= prefs["min_year"]]
        if prefs["max_year"]:
            f = f[f["Year"] <= prefs["max_year"]]
    return f

def recommend_movies(prefs, top_n=5):
    filtered = filter_movies(prefs)

    if filtered.empty:
        print("😢 Sorry, no matching movies found.")
        return

    desc_embed = embed_user_text(prefs)
    if desc_embed is not None:
        D, I = faiss_index.search(np.array(desc_embed), 100)
        similar_idxs = I[0]
        similarity_scores = D[0]
        filtered["similarity"] = filtered.index.map(lambda idx: 1 - similarity_scores[similar_idxs.tolist().index(idx)] if idx in similar_idxs else 0)
        filtered = filtered.sort_values("similarity", ascending=False)

    filtered["Director"] = filtered["Director"].apply(lambda x: x if x in top_directors else "Other")

    for col in ["likes_per_rating", "fans_per_watch", "likes_per_fan"]:
        if col not in filtered:
            filtered[col] = df[col].mean()

    try:
        filtered_for_pred = filtered[model.feature_names_in_]
    except:
        filtered_for_pred = filtered

    filtered["pred_rating"] = model.predict(filtered_for_pred)
    final = filtered.sort_values("pred_rating", ascending=False).head(top_n)

    print(f"\n🎯 Top {top_n} recommended movies for you:\n")
    for _, row in final.iterrows():
        print(f"🎬 {row['Film_title']} ({row['Runtime']} min) ⭐ {row['pred_rating']:.2f}")
        print(f"    Genres: {row['Genres']}")
        print(f"    Language: {row['Original_language']}, Likes: {row['Likes']}, Fans: {row['Fans']}")
        print(f"    Description: {row['Description'][:200]}...\n")

def main():
    print("🎬 Welcome to the Movie AI Recommender!")
    while True:
        prefs = collect_user_input()
        recommend_movies(prefs)
        again = input("\n🔁 Try again? (y/n): ").lower()
        if again != 'y':
            print("🎉 Enjoy your movies! Goodbye!")
            break

if __name__ == "__main__":
    main()
