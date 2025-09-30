
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from collections import Counter
import altair as alt
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from dotenv import load_dotenv
import os

#Point to db.env 
env_path = Path.cwd() / "db.env"    

# 2) Load to override ensures fresh values)
load_dotenv(dotenv_path=env_path, override=True)

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)
# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# LOADERS
@st.cache_data
def load_data():
    """Car News"""
    df = pd.read_sql("SELECT * FROM car_news;", engine)
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'], utc=True)
    return df

@st.cache_data
def load_car_data():
    """Car Reviews"""
    df1 = pd.read_sql("SELECT * FROM car_review;", engine)
    if 'publication_date' in df1.columns:
        df1['publication_date'] = pd.to_datetime(df1['publication_date'], utc=True)
    return df1

@st.cache_data
def load_keyword_data():
    return pd.read_sql("SELECT * FROM keyword_pairs;", engine)

@st.cache_data
def load_topic_data():
    df = pd.read_sql("SELECT * FROM news_articles_topics;", engine)

    # sanitize topic_keywords (string of comma-separated tokens expected)
    def sanitize_keywords(val):
        if not isinstance(val, str):
            return ""
        keywords = [kw.strip().lower() for kw in val.split(",") if kw.strip()]
        return ", ".join(keywords)

    if 'topic_keywords' in df.columns:
        df['topic_keywords'] = df['topic_keywords'].apply(sanitize_keywords)
    return df

@st.cache_data
def load_ner_news():
    # adjust this table name to your actual news NER table if different
    return pd.read_sql("SELECT * FROM car_news_named_entities;", engine)

@st.cache_data
def load_ner_reviews():
    # your snippet had two review NER tables; use the one that exists
    try:
        return pd.read_sql("SELECT * FROM car_reviews_named_entities;", engine)
    except Exception:
        return pd.read_sql("SELECT * FROM car_review_named_entities;", engine)

@st.cache_data
def load_market_trend():
    return pd.read_sql("SELECT * FROM market_trend_monthly;", engine)

@st.cache_data
def load_sentiment_trend():
    return pd.read_sql("SELECT * FROM sentiment_trend_monthly;", engine)


# DATA
news_df   = load_data()
reviews_df = load_car_data()
try:
    topic_df  = load_topic_data()
except Exception as _e:
    st.error("Failed to load from the database. Use **Run DB diagnostics** in the sidebar to see why.")
    st.stop()
keywords  = load_keyword_data()
ner_news = load_ner_news()
ner_reviews = load_ner_reviews()


# pick available text columns for searching
def pick_text_cols(df: pd.DataFrame):
    candidates = ["content", "verdict", "cleaned_content", "article_text", "body", "description"]
    return [c for c in candidates if c in df.columns]


# UI
st.sidebar.title("Auto-Intel Dashboard")

section = st.sidebar.selectbox("Choose Dataset:", ["Car News", "Car Reviews"], index=0)

if section == "Car News":
    news_option = st.sidebar.radio(
        "Car News • Choose Analysis:",
        ("Sentiment Trends", "Source Analysis", "Top Keywords", "Word Cloud", "Topic Modeling", "Named Entities"),
        index=0
    )
else:
    reviews_option = st.sidebar.radio(
        "Car Reviews • Choose Analysis:",
        ("Market Trend", "Named Entities", "Source Analysis"),
        index=0
    )

# CAR NEWS VIEWS

if section == "Car News" and news_option == "Sentiment Trends":
    st.subheader("Average Sentiment Score Over Time by Sentiment Type")
    needed = {'publication_date','sentiment_score','sentiment_label'}
    if needed.issubset(news_df.columns):
        tmp = news_df.copy()
        tmp['month'] = pd.to_datetime(tmp['publication_date']).dt.to_period('M').dt.to_timestamp()
        trend_df = tmp.groupby(['month','sentiment_label'])['sentiment_score'].mean().reset_index()

        chart = alt.Chart(trend_df).mark_line(point=True).encode(
            x=alt.X('month:T', title='Overtime'),
            y=alt.Y('sentiment_score:Q', title='Average Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color('sentiment_label:N', title='Sentiment Type',
                            scale=alt.Scale(domain=["positive","negative","neutral"],
                                            range=["green","red","gray"])),
            tooltip=['month:T','sentiment_label:N','sentiment_score:Q']
        ).properties(title="Average Sentiment Score by Type")
        st.altair_chart(chart, use_container_width=True)

        # Overall average + pie chart
        c1, c2 = st.columns([1,2])
        with c1:
            overall_avg = float(news_df['sentiment_score'].mean())
            st.metric("Overall Avg Sentiment", f"{overall_avg:.3f}")

        with c2:
            # Distribution pie (counts per label)
            dist = (news_df['sentiment_label']
                    .value_counts()
                    .rename_axis('sentiment_label')
                    .reset_index(name='count'))

            pie = alt.Chart(dist).mark_arc().encode(
                theta=alt.Theta('count:Q'),
                color=alt.Color('sentiment_label:N', title='Sentiment',
                                scale=alt.Scale(domain=["positive","negative","neutral"],
                                                range=["green","red","gray"])),
                tooltip=['sentiment_label:N','count:Q']
            ).properties(title="News Sentiment Distribution")
            st.altair_chart(pie, use_container_width=True)

    else:
        st.warning("Required columns not found in car_news (need publication_date, sentiment_score, sentiment_label).")


# Source Analysis
elif section == "Car News" and news_option == "Source Analysis":
    st.subheader("Sentiment by News Source")
    tmp = news_df.copy()
    # use 'link' if exists; else 'url'
    link_col = 'link' if 'link' in tmp.columns else ('url' if 'url' in tmp.columns else None)
    if link_col is None or 'sentiment_label' not in tmp.columns:
        st.warning("Missing columns for source analysis (need link/url and sentiment_label).")
    else:
        tmp['source_domain'] = tmp[link_col].apply(lambda x: urlparse(str(x)).netloc if pd.notnull(x) else "")
        sentiment_by_source = tmp.groupby(['source_domain','sentiment_label']).size().reset_index(name='count')
        pivot_df = sentiment_by_source.pivot(index='source_domain', columns='sentiment_label', values='count').fillna(0)
        # reorder if all three present
        for col in ["positive","negative","neutral"]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
        pivot_df = pivot_df[["positive","negative","neutral"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_df.plot(kind='bar', stacked=False, ax=ax, color=["green","red","gray"])
        ax.set_xlabel("News Source"); ax.set_ylabel("Article Count"); ax.set_title("Sentiment Distribution by News Source")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig)

# Top KeyWords

elif section == "Car News" and news_option == "Top Keywords":
    st.subheader("Top Keywords Frequency")
    if 'phrase' in keywords.columns and 'count' in keywords.columns:
        # Sort by count descending
        top_keywords = keywords.sort_values(by='count', ascending=False)
        # Display table
        st.dataframe(top_keywords)
        # Plot top 30
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_keywords.head(30), x='count', y='phrase', ax=ax)
        ax.set_title("Top 20 Keyword Phrases")
        st.pyplot(fig)
    else:
        st.warning("Required columns 'phrase' or 'count' not found in keyword data.")

#Word Cloud
elif section == "Car News" and news_option == "Word Cloud":
    st.subheader("Word Cloud of Keyword Phrases")
    if {'phrase','count'}.issubset(keywords.columns):
        expanded = []
        for _, row in keywords.iterrows():
            phrase = str(row['phrase']); count = int(row['count'])
            expanded.extend([phrase]*count)
        text = " ".join(expanded)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("The 'keyword_pairs' table must contain 'phrase' and 'count'.")

# Topic Modeling
elif section == "Car News" and news_option == "Topic Modeling":
    st.subheader("Dominant Topics in News Articles")
    if 'dominant_topic' in topic_df.columns:
        topic_counts = topic_df['dominant_topic'].value_counts().sort_index()
        topic_labels = {
            0: "Electric Vehicles",
            1: "Driving Experience",
            2: "Car Design & Features",
            3: "Technology & Innovation",
            4: "Market Trends"
        }
        topic_counts.index = topic_counts.index.map(lambda i: topic_labels.get(i, f"Topic {i}"))
        st.bar_chart(topic_counts)

        st.subheader("Topic Keywords")
        if 'topic_keywords' in topic_df.columns:
            for i in sorted(topic_df['dominant_topic'].dropna().unique()):
                kws = topic_df[topic_df['dominant_topic']==i]['topic_keywords'].iloc[0]
                label = topic_labels.get(i, f"Topic {i}")
                st.markdown(f"**{label}**: {kws}")
    else:
        st.warning("Missing 'dominant_topic' in topics table.")

# Named Entity
elif section == "Car News" and news_option == "Named Entities":
    st.subheader("Top Named Entities in News Articles")
    if ner_news.empty:
        st.info("No NER data for news.")
    else:
        label_descriptions = {
            "ORG": "Organizations (e.g., Tesla, Ford, UN)",
            "GPE": "Geo-Political Entities (countries, cities)",
            "PRODUCT": "Commercial Products (e.g., Model 3)"
        }
        available_labels = ner_news['label'].dropna().unique().tolist()
        selected_label = st.selectbox("Filter by Entity Type", sorted(available_labels))
        st.caption(label_descriptions.get(selected_label, ""))

        # label-specification
        exclude_by_label = {
            "GPE": {"dc", "n't", "n’t", "f1", "n t", "ai", "gt", "skoda", "v6", "ferrari"},
            "ORG": {"dc", "n't", "f1", "n t", "n’t", "ai", "gt", "car", "ev", "apple", "digital"},
            "PRODUCT": set()
        }
        ex = exclude_by_label.get(selected_label, set())
        
        sub = ner_news[ner_news['label'] == selected_label].copy()
        sub = sub[~sub['entity'].str.lower().isin(ex)]   
        sub = sub.sort_values('count', ascending=False)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=sub.head(30), x='count', y='entity', ax=ax)
        ax.set_title(f"Top Entities of Type: {selected_label}")
        st.pyplot(fig)

        # Drill-down
        selected_entity = st.selectbox("Select an Entity to Explore Mentions", sub['entity'].head(30).tolist())
        st.markdown(f"### Articles mentioning **{selected_entity}**")
        text_cols = pick_text_cols(news_df)
        if not text_cols and 'title' not in news_df.columns:
            st.warning("No searchable text columns in car_news.")
        else:
            if text_cols:
                search_series = news_df[text_cols].astype(str).agg(" ".join, axis=1)
            else:
                search_series = news_df['title'].astype(str)
            title_series = news_df['title'].astype(str) if 'title' in news_df.columns else pd.Series("", index=news_df.index)

            mask = search_series.str.contains(selected_entity, case=False, na=False) | \
                   title_series.str.contains(selected_entity, case=False, na=False)

            cols = [c for c in ['title','publication_date','sentiment_label','sentiment_score'] if c in news_df.columns]
            if text_cols: cols.append(text_cols[0])
            mentions = news_df.loc[mask, cols].copy()

            if mentions.empty:
                st.info("No articles matched this entity.")
            else:
                if 'sentiment_label' in mentions.columns:
                    st.write("Sentiment distribution:")
                    st.bar_chart(mentions['sentiment_label'].value_counts())
                st.write("Sample mentions:")
                for _, row in mentions.head(5).iterrows():
                    ttl = row.get('title', '(no title)')
                    dt  = row.get('publication_date', '')
                    lab = row.get('sentiment_label', '')
                    sc  = row.get('sentiment_score', '')
                    st.markdown(f"**{ttl}** — _{dt}_ • **{lab}** {f'(score: {sc:.2f})' if isinstance(sc, float) else ''}")
                    if text_cols:
                        snip = str(row.get(text_cols[0], ""))[:300]
                        if snip: st.caption(snip + "…")
                    st.markdown("---")


# CAR REVIEWS VIEWS
if section == "Car Reviews" and reviews_option == "Market Trend":
    st.subheader("Car Reviews • Market Trend")

    # Market metrics
    market_df = load_market_trend()
    if market_df.empty:
        st.warning("`market_trend_monthly` is empty. Run the market trend pipeline.")
    else:
        market_df['publication_date'] = pd.to_datetime(market_df['publication_date'])
        market_df = market_df.set_index('publication_date').sort_index()
        st.line_chart(market_df[['avg_price','avg_rating']])

        st.subheader("Monthly Article Review Count")
        st.bar_chart(market_df['article_count'])

    # Review sentiment timeline (if available)
    if {'publication_date','sentiment_score','sentiment_label'}.issubset(reviews_df.columns):
        tmp = reviews_df.copy()
        tmp['month'] = tmp['publication_date'].dt.to_period('M').dt.to_timestamp()
        r_trend = tmp.groupby(['month','sentiment_label'])['sentiment_score'].mean().reset_index()
    
        chart = alt.Chart(r_trend).mark_line(point=True).encode(
            x=alt.X('month:T', title='Overtime'),
            y=alt.Y('sentiment_score:Q', title='Average Sentiment Score', scale=alt.Scale(domain=[-1,1])),
            color=alt.Color('sentiment_label:N', title='Sentiment Type',
                            scale=alt.Scale(domain=["positive","negative","neutral"],
                                            range=["green","red","gray"])),
            tooltip=['month:T','sentiment_label:N','sentiment_score:Q']
        ).properties(title="Average Review Sentiment by Type")
        st.altair_chart(chart, use_container_width=True)
    
        # Pie chart + overall average metric
        c1, c2 = st.columns([1,2])
        with c1:
            overall_avg = float(reviews_df['sentiment_score'].mean())
            st.metric("Overall Avg Sentiment", f"{overall_avg:.3f}")
    
        with c2:
            dist = (reviews_df['sentiment_label']
                    .value_counts()
                    .rename_axis('sentiment_label')
                    .reset_index(name='count'))
    
            pie = alt.Chart(dist).mark_arc().encode(
                theta=alt.Theta(field='count', type='quantitative'),
                color=alt.Color('sentiment_label:N', title='Sentiment',
                                scale=alt.Scale(domain=["positive","negative","neutral"],
                                                range=["green","red","gray"])),
                tooltip=['sentiment_label:N','count:Q']
            ).properties(title="Review Sentiment Distribution (Pie)")
            st.altair_chart(pie, use_container_width=True)

# Named Entities

elif section == "Car Reviews" and reviews_option == "Named Entities":
    st.subheader("Top Named Entities in Car Reviews")
    if ner_reviews.empty:
        st.info("No NER data for reviews.")
    else:
        label_descriptions = {
            "ORG": "Organizations (e.g., Tesla, Ford, UN)",
            "GPE": "Geo-Political Entities (countries, cities)",
            "PRODUCT": "Commercial Products (e.g., Model 3)"
        }
        available_labels = ner_reviews['label'].dropna().unique().tolist()
        selected_label = st.selectbox("Filter by Entity Type", sorted(available_labels))
        st.caption(label_descriptions.get(selected_label, ""))

        exclude_by_label = {
            "GPE": {"dc", "n't", "f1", "n t", "ai", "gt", "skoda", "v6", "ferrari"},
            "ORG": {"dc", "n't", "f1", "n t", "ai", "gt", "car", "ev", "apple", "digital"},
            "PRODUCT": set()
        }
        ex = exclude_by_label.get(selected_label, set())
        
        sub = ner_news[ner_news['label'] == selected_label].copy()
        sub = sub[~sub['entity'].str.lower().isin(ex)]   # exclude unwanted
        sub = sub.sort_values('count', ascending=False)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=sub.head(30), x='count', y='entity', ax=ax)
        ax.set_title(f"Top Entities of Type: {selected_label}")
        st.pyplot(fig)

        selected_entity = st.selectbox("Select an Entity to Explore Mentions", sub['entity'].head(30).tolist())
        st.markdown(f"### Review Mentions of **{selected_entity}**")

        text_cols = pick_text_cols(reviews_df)
        if not text_cols and 'title' not in reviews_df.columns:
            st.warning("No searchable text columns in car_reviews.")
        else:
            if text_cols:
                search_series = reviews_df[text_cols].astype(str).agg(" ".join, axis=1)
            else:
                search_series = reviews_df['title'].astype(str)
            title_series = reviews_df['title'].astype(str) if 'title' in reviews_df.columns else pd.Series("", index=reviews_df.index)

            mask = search_series.str.contains(selected_entity, case=False, na=False) | \
                   title_series.str.contains(selected_entity, case=False, na=False)

            cols = [c for c in ['title','publication_date','sentiment_label','sentiment_score','rate','price'] if c in reviews_df.columns]
            if text_cols: cols.append(text_cols[0])
            mentions = reviews_df.loc[mask, cols].copy()

            if mentions.empty:
                st.info("No reviews matched this entity.")
            else:
                if 'sentiment_label' in mentions.columns:
                    st.write("Sentiment distribution:")
                    st.bar_chart(mentions['sentiment_label'].value_counts())
                st.write("Sample mentions:")
                for _, row in mentions.head(5).iterrows():
                    ttl = row.get('title', '(no title)')
                    dt  = row.get('publication_date', '')
                    lab = row.get('sentiment_label', '')
                    sc  = row.get('sentiment_score', '')
                    st.markdown(f"**{ttl}** — _{dt}_ • **{lab}** {f'(score: {sc:.2f})' if isinstance(sc, float) else ''}")
                    if text_cols:
                        snip = str(row.get(text_cols[0], ""))[:300]
                        if snip: st.caption(snip + "…")
                    st.markdown("---")
# Source Analysis

elif section == "Car Reviews" and reviews_option == "Source Analysis":
    st.subheader("Sentiment by review Source")

    # Extract domain names from full links
    reviews_df['source_domain'] = reviews_df['link'].apply(lambda x: urlparse(x).netloc)

    # Group by domain and sentiment
    sentiment_by_source = reviews_df.groupby(['source_domain', 'sentiment_label']).size().reset_index(name='count')

    # Optional: Keep only top 10 sources by total article count
    top_sources = sentiment_by_source.groupby('source_domain')['count'].sum().nlargest(10).index
    sentiment_by_source = sentiment_by_source[sentiment_by_source['source_domain'].isin(top_sources)]

    # Pivot for plotting
    pivot_df = sentiment_by_source.pivot(index='source_domain', columns='sentiment_label', values='count').fillna(0)

    #re-order 
    pivot_df = pivot_df[[ "positive", "negative", "neutral"]]

    # Plot as stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df.plot(kind='bar', stacked=False, ax=ax, color=[ "green", "red", "gray"])

    plt.xlabel("News Source")
    plt.ylabel("Article Count")
    plt.title("Sentiment Distribution by News Source")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

