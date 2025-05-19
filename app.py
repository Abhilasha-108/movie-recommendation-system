from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

df = pd.read_csv('tmdb_5000_movies.csv')
df['overview'] = df['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_by_keyword(keyword, num=5):
    keyword = keyword.lower()
    filtered = df[df['overview'].str.contains(keyword) | df['title'].str.contains(keyword, case=False)]
    if filtered.empty:
        return ["No movies found."]
    tfidf_filtered = tfidf.transform(filtered['overview'])
    cos_sim_filtered = cosine_similarity(tfidf_filtered, tfidf_filtered)
    sim_scores = list(enumerate(cos_sim_filtered[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num+1]]
    return filtered['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        keyword = request.form.get("keyword")
        recommendations = recommend_by_keyword(keyword)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

