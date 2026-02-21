from database import songs_db
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Prepare data once(not every request)

song_titles = list(songs_db.keys())
song_lyrics = list(songs_db.values())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(song_lyrics)

def predict_song_from_lyrics(user_lyrics: str):
    user_vector = vectorizer.transform([user_lyrics])
    similarities = cosine_similarity(user_vector, tfidf_matrix)

    best_match_index = similarities.argmax()
    confidence = float(similarities[0][best_match_index])

    # TEMPORARY: Remove threshold
    return song_titles[best_match_index], confidence