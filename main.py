from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import SongRequest, SongResponse
from services import predict_song_from_lyrics

app = FastAPI()

# Enable CORS (for frontend later)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Song Guesser AI Backend Running!"}

import os 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_FOLDER = "datasets"

all_dfs = []

for file in os.listdir(DATASET_FOLDER):
    if file.endswith(".csv"):
        file_path = os.path.join(DATASET_FOLDER, file)
        print(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        all_dfs.append(df)
combined_df = pd.concat(all_dfs, ignore_indez=True)

# Clean column names (important)
combined_df.columns = combined_df.columns.str.strip()

print("Columns found:", combined_df.columns)
print(f"Total songs loaded: {len(combined_df)}")

song_titles = combined_df["title"].tolist()
lyrics_list = combined_df["lyric"].fillna("").tolist()

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    max_df=0.8,
    min_df=5
)

tfidf_matrix = vectorizer.fit_transform(lyrics_list)

@app.post("/predict", response_model=SongResponse)
def predict_song(request: SongRequest):
    predicted_song, confidence = predict_song_from_lyrics(request.lyrics)

    if not predicted_song:
        raise HTTPException(status_code=404, detail="Song not found")
    
    return {
        "success": True,
        "data": {
            "predicted_song": predicted_song,
            "confidence": confidence
        },
        "message": "Prediction successful"
    }