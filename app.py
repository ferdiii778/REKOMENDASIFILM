import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load movie data
file_path = 'indonesian_movies.csv'
data = pd.read_csv(file_path)
data['description'] = data['description'].fillna('')
data['genre'] = data['genre'].str.lower()

# Define genre synonyms dictionary
genre_synonyms = {
    'seram': 'horror',
    'menakutkan': 'horror',
    'teror': 'horror',
    'petualangan': 'adventure',
    'adrenaline': 'adventure',
    'perjalanan': 'adventure',
    'eksplore': 'adventure',
    'eksplorasi': 'adventure',
    'penjelajah': 'adventure',
    'pengembara': 'adventure',
    'lucu': 'comedy',
    'lawakan': 'comedy',
    'konyol': 'comedy',
    'percintaan': 'romance',
    'romantis': 'romance',
    'cinta': 'romance',
    'gore': 'thriller',
    'pembunuhan': 'thriller',
    'menegangkan': 'thriller',
    'psikopat': 'thriller',
    # Tambahkan persamaan kata lain sesuai dengan kebutuhan anda
}

# Using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['description'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(genre_filter=None, age_rating=None, min_rating=None, year=None, cosine_sim=cosine_sim):
    recommendations = data.copy()

    # Mengisi nilai NaN terlebih dahulu
    recommendations['genre'] = recommendations['genre'].fillna('')
    recommendations['age_rating'] = recommendations['age_rating'].fillna('')
    recommendations['users_rating'] = recommendations['users_rating'].fillna(0)
    recommendations['year'] = recommendations['year'].fillna(0)

    if genre_filter:
        # Translate genre_filter using genre_synonyms
        genre_filter = genre_synonyms.get(genre_filter.lower(), genre_filter.lower())
        recommendations = recommendations[recommendations['genre'].str.contains(genre_filter, na=False)]

    if age_rating is not None:
        if age_rating == 'remaja':
            recommendations = recommendations[recommendations['age_rating'] == '13+']
        elif age_rating == 'dewasa':
            recommendations = recommendations[recommendations['age_rating'] == '17+']

    if min_rating:
        recommendations = recommendations[recommendations['users_rating'] >= min_rating]

    if year:
        recommendations = recommendations[recommendations['year'] == year]

    recommendations = recommendations.fillna('-')
    recommendations = recommendations.sort_values(by='users_rating', ascending=False)

    top_recommendation = recommendations.iloc[0].to_dict() if not recommendations.empty else None
    other_recommendations = recommendations.iloc[1:].to_dict(orient='records') if len(recommendations) > 1 else []

    return top_recommendation, other_recommendations

@app.route('/')
def index():
    random_films = data.sample(n=5).to_dict(orient='records')
    return render_template('index.html', films=random_films)

@app.route('/movie/<title>')
def movie_detail(title):
    movie_data = data[data['title'].str.lower() == title.lower()].iloc[0].to_dict()
    return render_template('movie_detail.html', movie=movie_data)

@app.route('/recommend', methods=['POST'])
def recommend():
    genre = request.form.get('genre')
    age_rating = request.form.get('age_rating')
    min_rating = request.form.get('min_rating', type=float)
    year = request.form.get('year', type=int)

    recommendations = get_recommendations(genre_filter=genre, age_rating=age_rating, min_rating=min_rating, year=year)
    
    if not recommendations[0]:  # Jika tidak ada rekomendasi yang ditemukan
        return render_template('error.html', message="Tidak ada rekomendasi yang ditemukan.")
    
    top_recommendation, other_recommendations = recommendations
    
    return render_template('result.html', top_recommendation=top_recommendation, other_recommendations=other_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
