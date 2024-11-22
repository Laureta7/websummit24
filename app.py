import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
file_path = 'websummit_startups_2024.csv'
startups_data = pd.read_csv(file_path)

# Streamlit app
st.title("Web Summit Startups Analysis Dashboard")

# Overview Section
st.header("Overview")
total_startups = len(startups_data)
total_countries = startups_data['Country'].nunique()
total_categories = startups_data['Category'].nunique()
st.metric("Total Startups", total_startups)
st.metric("Total Countries", total_countries)
st.metric("Total Categories", total_categories)

# Analysis Selection
analysis_type = st.selectbox("Select Analysis Type", ["Overview", "Category Analysis", "Country Analysis", "Pitch Analysis", "Potential Collaborations", "Emerging Categories", "Focus on Web3 Startups", "Comparison"])

# Category Analysis
if analysis_type == "Category Analysis":
    st.header("Category Analysis")
    category_counts = startups_data['Category'].value_counts()
    st.bar_chart(category_counts)

    # Word Cloud for Top Categories
    st.subheader("Word Cloud for Top Categories")
    top_categories = startups_data['Category'].value_counts().head(10).index
    selected_category = st.selectbox("Select a Category", top_categories)
    category_startups = startups_data[startups_data['Category'] == selected_category]
    category_pitches = category_startups['Pitch'].dropna()
    category_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(category_pitches))
    plt.figure(figsize=(10, 6))
    plt.imshow(category_wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.close()

# Country Analysis
elif analysis_type == "Country Analysis":
    st.header("Country Analysis")
    num_countries = st.slider("Select Number of Countries", 1, 20, 10)
    country_counts = startups_data['Country'].value_counts().head(num_countries)
    st.bar_chart(country_counts)

    # Word Cloud for Top Countries
    st.subheader("Word Cloud for Top Countries")
    top_countries = startups_data['Country'].value_counts().head(num_countries).index
    selected_country = st.selectbox("Select a Country", top_countries)
    country_startups = startups_data[startups_data['Country'] == selected_country]
    country_pitches = country_startups['Pitch'].dropna()
    country_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(country_pitches))
    plt.figure(figsize=(10, 6))
    plt.imshow(country_wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.close()

# Pitch Analysis
elif analysis_type == "Pitch Analysis":
    st.header("Pitch Analysis")
    clean_pitches = startups_data['Pitch'].dropna()
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    word_counts = vectorizer.fit_transform(clean_pitches)
    common_words = pd.DataFrame({
        'Word': vectorizer.get_feature_names_out(),
        'Frequency': word_counts.toarray().sum(axis=0)
    }).sort_values(by='Frequency', ascending=False)
    st.bar_chart(common_words.set_index('Word'))

# Potential Collaborations
elif analysis_type == "Potential Collaborations":
    st.header("Potential Collaborations")
    category_country_pivot = startups_data.pivot_table(index='Category', columns='Country', aggfunc='size', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(category_country_pivot, cmap='viridis', annot=True, fmt='d', cbar_kws={'label': 'Number of Startups'})
    plt.title('Heatmap of Startups by Category and Country')
    plt.xlabel('Country')
    plt.ylabel('Category')
    st.pyplot(plt)
    plt.close()

# Emerging Categories
elif analysis_type == "Emerging Categories":
    st.header("Emerging Categories")
    emerging_categories = startups_data['Category'].value_counts().tail(10)
    st.bar_chart(emerging_categories)

# Focus on Web3 Startups
elif analysis_type == "Focus on Web3 Startups":
    st.header("Focus on Web3 Startups")
    web3_startups = startups_data[startups_data['Category'].str.contains('Web3', case=False, na=False)]
    web3_pitches = web3_startups['Pitch'].dropna()
    web3_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(web3_pitches))
    plt.figure(figsize=(10, 6))
    plt.imshow(web3_wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.close()

# Comparison
elif analysis_type == "Comparison":
    st.header("Comparison")
    comparison_type = st.selectbox("Select Comparison Type", ["Country Comparison", "Category Comparison"])

    if comparison_type == "Country Comparison":
        countries_to_compare = st.multiselect("Select Countries to Compare", startups_data['Country'].unique(), default=["Switzerland", "Portugal"])
        for country in countries_to_compare:
            st.subheader(f"Word Cloud for {country} Startups")
            country_startups = startups_data[startups_data['Country'] == country]
            country_pitches = country_startups['Pitch'].dropna()
            country_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(country_pitches))
            plt.figure(figsize=(10, 6))
            plt.imshow(country_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            plt.close()

    elif comparison_type == "Category Comparison":
        categories_to_compare = st.multiselect("Select Categories to Compare", startups_data['Category'].unique(), default=[])
        for category in categories_to_compare:
            st.subheader(f"Word Cloud for {category} Startups")
            category_startups = startups_data[startups_data['Category'] == category]
            category_pitches = category_startups['Pitch'].dropna()
            category_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(category_pitches))
            plt.figure(figsize=(10, 6))
            plt.imshow(category_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            plt.close()