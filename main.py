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