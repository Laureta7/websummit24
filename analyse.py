import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
file_path = 'websummit_startups_2024.csv'
startups_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(startups_data.head())

# Validate the data
print(startups_data.describe(include='all'))

# 1. Count the number of startups per category
category_counts = startups_data['Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Number of Startups per Category')
plt.xlabel('Number of Startups')
plt.ylabel('Category')

# 2. Count the number of startups per country
country_counts = startups_data['Country'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
plt.title('Number of Startups per Country')
plt.xlabel('Number of Startups')
plt.ylabel('Country')

# 3. Analyze the pitches to find recurring themes or keywords

# Remove NaN values in 'Pitch' column for text analysis
clean_pitches = startups_data['Pitch'].dropna()

# Use CountVectorizer to extract common words from pitches
vectorizer = CountVectorizer(stop_words='english', max_features=20)
word_counts = vectorizer.fit_transform(clean_pitches)
common_words = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Frequency': word_counts.toarray().sum(axis=0)
}).sort_values(by='Frequency', ascending=False)

# Display the results
print(common_words)

# Plot the most common words in pitches
plt.figure(figsize=(10, 6))
sns.barplot(x=common_words['Frequency'], y=common_words['Word'], palette='viridis')
plt.title('Most Common Words in Pitches')
plt.xlabel('Frequency')
plt.ylabel('Word')

# 4. Focused analysis on Swiss startups

# Filter the dataset for startups based in Switzerland
swiss_startups = startups_data[startups_data['Country'] == 'Switzerland']

# Analyze the categories for Swiss startups
swiss_categories = swiss_startups['Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=swiss_categories.values, y=swiss_categories.index, palette='viridis')
plt.title('Number of Swiss Startups per Category')
plt.xlabel('Number of Startups')
plt.ylabel('Category')

# Extract and analyze pitches for common themes or keywords
swiss_pitches = swiss_startups['Pitch'].dropna()
swiss_word_counts = vectorizer.fit_transform(swiss_pitches)
swiss_common_words = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Frequency': swiss_word_counts.toarray().sum(axis=0)
}).sort_values(by='Frequency', ascending=False)

# Plot the most common words in Swiss pitches
plt.figure(figsize=(10, 6))
sns.barplot(x=swiss_common_words['Frequency'], y=swiss_common_words['Word'], palette='viridis')
plt.title('Most Common Words in Swiss Pitches')
plt.xlabel('Frequency')
plt.ylabel('Word')

# 5. Focused analysis on Portuguese startups

# Filter the dataset for startups based in Portugal
portuguese_startups = startups_data[startups_data['Country'] == 'Portugal']

# Extract and analyze pitches for common themes or keywords
portuguese_pitches = portuguese_startups['Pitch'].dropna()
portuguese_word_counts = vectorizer.fit_transform(portuguese_pitches)
portuguese_common_words = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Frequency': portuguese_word_counts.toarray().sum(axis=0)
}).sort_values(by='Frequency', ascending=False)

# Plot the most common words in Portuguese pitches
plt.figure(figsize=(10, 6))
sns.barplot(x=portuguese_common_words['Frequency'], y=portuguese_common_words['Word'], palette='viridis')
plt.title('Most Common Words in Portuguese Pitches')
plt.xlabel('Frequency')
plt.ylabel('Word')

# 6. Word cloud comparison between Swiss and Portuguese startups

# Generate word clouds
swiss_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(swiss_pitches))
portuguese_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(portuguese_pitches))

# Plot word clouds
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(swiss_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Swiss Startups')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(portuguese_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Portuguese Startups')
plt.axis('off')

# 7. Heatmap of startups by category and country

# Create a pivot table
category_country_pivot = startups_data.pivot_table(index='Category', columns='Country', aggfunc='size', fill_value=0)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(category_country_pivot, cmap='viridis', annot=True, fmt='d')
plt.title('Heatmap of Startups by Category and Country')
plt.xlabel('Country')
plt.ylabel('Category')

# 8. Top categories in each country

# Get top categories for each country
top_categories_per_country = startups_data.groupby('Country')['Category'].apply(lambda x: x.value_counts().head(3))

# Plot top categories for each country
plt.figure(figsize=(12, 8))
top_categories_per_country.unstack().plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Top Categories in Each Country')
plt.xlabel('Country')
plt.ylabel('Number of Startups')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# 9. Word cloud for all startups

# Generate word cloud for all pitches
all_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(clean_pitches))

# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(all_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for All Startups')
plt.axis('off')

# Show all plots
plt.show()