import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx

# Load the dataset
file_path = 'websummit_startups_2024.csv'
startups_data = pd.read_csv(file_path)

# 1. Analyze collaborations: Country vs Category
category_country_pivot = startups_data.pivot_table(index='Category', columns='Country', aggfunc='size', fill_value=0)

# Heatmap for collaborations
plt.figure(figsize=(12, 8))
sns.heatmap(category_country_pivot, cmap='viridis', annot=True, fmt='d', cbar_kws={'label': 'Number of Startups'})
plt.title('Heatmap of Startups by Category and Country')
plt.xlabel('Country')
plt.ylabel('Category')
plt.show()

# 2. Correlation between category and keywords in pitches
vectorizer = CountVectorizer(stop_words='english', max_features=20)
clean_pitches = startups_data['Pitch'].dropna()

word_counts = vectorizer.fit_transform(clean_pitches)
words_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
words_df['Category'] = startups_data.loc[clean_pitches.index, 'Category']

# Average frequency of each word per category
words_by_category = words_df.groupby('Category').mean().transpose()
plt.figure(figsize=(12, 6))
sns.heatmap(words_by_category, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Average Frequency'})
plt.title('Keyword Frequency by Category')
plt.xlabel('Category')
plt.ylabel('Word')
plt.show()

# 3. Visualize relationships between countries and categories using a network graph
category_country_counts = startups_data.groupby(['Category', 'Country']).size().reset_index(name='Count')
graph = nx.Graph()

for _, row in category_country_counts.iterrows():
    graph.add_edge(row['Category'], row['Country'], weight=row['Count'])

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph, k=0.5)
edges = graph.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw_networkx(graph, pos, with_labels=True, edge_color=weights, edge_cmap=plt.cm.viridis, node_size=700, font_size=10)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
sm.set_array([])
plt.colorbar(sm, label='Number of Startups')
plt.title('Network of Countries and Categories')
plt.show()

# 4. Trends by region: Compare top keywords in pitches for each region
def analyze_keywords_by_country(country_name):
    country_pitches = startups_data[startups_data['Country'] == country_name]['Pitch'].dropna()
    word_counts = vectorizer.fit_transform(country_pitches)
    common_words = pd.DataFrame({
        'Word': vectorizer.get_feature_names_out(),
        'Frequency': word_counts.toarray().sum(axis=0)
    }).sort_values(by='Frequency', ascending=False)
    return common_words

# Example: Analyze for Switzerland
swiss_words = analyze_keywords_by_country('Switzerland')
print(swiss_words.head(10))

# Plot keyword frequencies for Switzerland
plt.figure(figsize=(10, 6))
sns.barplot(x=swiss_words['Frequency'], y=swiss_words['Word'], palette='viridis')
plt.title('Top Keywords in Swiss Startups Pitches')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

# Example: Analyze for Portugal
portuguese_words = analyze_keywords_by_country('Portugal')
print(portuguese_words.head(10))

# Plot keyword frequencies for Portugal
plt.figure(figsize=(10, 6))
sns.barplot(x=portuguese_words['Frequency'], y=portuguese_words['Word'], palette='viridis')
plt.title('Top Keywords in Portuguese Startups Pitches')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

# 5. Word cloud comparison for all startups
all_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(clean_pitches))
plt.figure(figsize=(10, 6))
plt.imshow(all_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for All Startups')
plt.axis('off')
plt.show()



# 6. Analyze potential collaborations
# Identify categories with high potential for collaboration
collaboration_potential = category_country_pivot.sum(axis=1).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=collaboration_potential.values, y=collaboration_potential.index, palette='viridis')
plt.title('Potential for Collaboration by Category')
plt.xlabel('Number of Startups')
plt.ylabel('Category')
plt.show()

# 7. Emerging categories
# Identify categories with fewer startups but high growth potential
emerging_categories = category_country_pivot.sum(axis=1).sort_values().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=emerging_categories.values, y=emerging_categories.index, palette='viridis')
plt.title('Emerging Categories with High Growth Potential')
plt.xlabel('Number of Startups')
plt.ylabel('Category')
plt.show()

# 8. Focus on Web3 startups
web3_startups = startups_data[startups_data['Category'].str.contains('Web3', case=False, na=False)]
web3_pitches = web3_startups['Pitch'].dropna()
web3_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(web3_pitches))
plt.figure(figsize=(10, 6))
plt.imshow(web3_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Web3 Startups')
plt.axis('off')
plt.show()