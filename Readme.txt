Features:

- Reads raw data from PostgreSQL (`car_news`, `car_review`)
- Cleans and preprocesses text (stopwords, lemmatization)
- Sentiment Analysis with VADER
- Topic Modeling with LDA
- Writes back to SQL: temp_sentiments, news_articles_topics, ...


    Interactive Dashboard (`app.py`)
    Sentiment trends over time
    Source analysis (by news outlet)
    Top keywords (uni/bi/tri-grams)
    Word cloud
    Topic modelling visualisation
    Named Entity Recognition (brands, organisations, products)
    Market trends (price, rating, article count)

Tech Stack:

    Languages: Python 3.10+
    Database: PostgreSQL 14+
    Libraries: NLTK, gensim, spaCy, vaderSentiment, scikit-learn, matplotlib, seaborn, Altair, WordCloud, Streamlit, SQLAlchemy

Set up virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Connect to Database:

    DB_URL = "postgresql://auto_intel_user:G2ifL76ULDi9YOFJ4tHZ1ikUEORMx7Oe@dpg-d2i7ptvdiees73d1b4lg-a.oregon-postgres.render.com/auto_intel"
    engine = create_engine(DB_URL)

Run the dashboard:
Terminal> cd [folder location path]
    streamlit run app.py