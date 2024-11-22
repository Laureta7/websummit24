import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_webpage(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    startups = []
    
    # Find all startup names, links, and categories
    for startup in soup.select('figure.ListItemStyles__StyledListItemWrapper-sc-94ce60d2-2'):
        name = startup.select_one('a').get_text(strip=True)
        link = startup.select_one('a')['href']
        category = startup.select_one('span.ListItemStyles__ItemDescription-sc-94ce60d2-5').get_text(strip=True)
        startups.append((name, link, category))
    
    return startups

def fetch_company_details(url):
    html = fetch_webpage(url)
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract country
    country = soup.select_one('a.ContentTagList__ContentTagListItem-sc-6e6a07b7-1 .bodyCopy__P-sc-986c63f9-1')
    country_text = country.get_text(strip=True) if country else 'N/A'
    
    # Extract social media links
    social_links = {}
    for link in soup.select('div.SocialButton__SocialButtonWrapper-sc-29e85cc1-0 a'):
        platform = link['href'].split('.')[1]  # Extract platform name from URL
        social_links[platform] = link['href']
    
    # Extract pitch
    pitch = soup.select_one('div.ProfileDetails__ProfileDetailsContent-sc-8beaea78-1')
    pitch_text = pitch.get_text(strip=True) if pitch else 'N/A'
    
    return country_text, social_links, pitch_text

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['Startup Name', 'Link', 'Country', 'Category', 'Social Links', 'Pitch'])
    df.to_csv(filename, index=False)

def main():
    base_url = 'https://websummit.com/startups/featured-startups/page/'
    page = 1
    detailed_startups = []
    
    while page <= 20:
        url = f'{base_url}{page}/'
        try:
            html = fetch_webpage(url)
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching page {page}: {e}")
            break
        
        startups = parse_html(html)
        
        if not startups:
            break
        
        for name, link, category in startups:
            full_url = 'https://websummit.com' + link
            try:
                country, social_links, pitch = fetch_company_details(full_url)
            except Exception as e:
                print(f"Error fetching details for {name} at {full_url}: {e}")
                country, social_links, pitch = 'N/A', {}, 'N/A'
            detailed_startups.append((name, full_url, country, category, social_links, pitch))
        
        page += 1
    
    save_to_csv(detailed_startups, 'websummit_startups_2024.csv')
    print(f'Successfully saved {len(detailed_startups)} startups to websummit_startups_2024.csv')

if __name__ == '__main__':
    main()