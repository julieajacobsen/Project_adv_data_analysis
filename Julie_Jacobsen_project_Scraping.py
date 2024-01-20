import requests
from bs4 import BeautifulSoup
import pandas as pd, numpy as np
from IPython.display import display, Image
import re


headers = {'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

url = "https://www.transfermarkt.com/statistik/saisontransfers"
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")


#Saving the link to the pages
num_pages = 80
links = []
for i in range(1, num_pages + 1):
    links.append("https://www.transfermarkt.com/statistik/saisontransfers?page=" + str(i))


#List of names and ages
name = []
age = []
club = []
position = []
market_value = []
Liga = []

#Going through each page
for link in links:
    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    #finding the player names and club names and saving them in the list
    Players = soup.find_all("td", {"rowspan": "2"})
    i = 0
    for player in Players:
        if i%2==0:
            player_name = player.find("img")["alt"]
            name.append(player_name)
        else:
            club_name = player.find("img")["alt"]
            club.append(club_name)
        i+=1

    #Finding the player ages and saving them in the list
    Ages = [td.text.strip() for td in soup.find_all('td', class_='zentriert')]
    i = 0
    for value in Ages:
        if i%4==1:
            if not value == "":
                age.append(value)
        i+=1

    #Finding the market value and saving them in the list
    columns = soup.select('.rechts:not(.rechts.hauptlink), .rechts.bg_gruen_20:not(.rechts.hauptlink)')

    for column in columns[2:]:
        value_text = column.text.strip().replace('€', '').replace('-', '')
    
    # Check if 'k' or 'm' is present and convert to the common scale
        if 'k' in value_text:
            try:
                value = float(value_text.replace('k', '')) / 1000.0  # Convert to million
            except ValueError:
                value = None  # Handle invalid values
        elif 'm' in value_text:
            try:
                value = float(value_text.replace('m', ''))  # Already in million
            except ValueError:
                value = None  # Handle invalid values
        else:
            try:
                value = float(value_text)  # No 'k' or 'm', keep it as is
            except ValueError:
                value = None
            
        market_value.append(value)

    #Finding the position and saving them in the list
    player_entries = soup.find_all('tr', class_=['even', 'odd'])
    pattern = r'(Goalkeeper|Sweeper|Centre-Back|Left-Back|Right-Back|Defensive Midfield|Central Midfield|Right Midfield|Left Midfield|Attacking Midfield|Left Winger|Right Winger|Second Striker|Centre-Forward)'
    for entry in player_entries:
        position_elements = entry.find_all('td', recursive=False)
        position_value = position_elements[1].get_text(strip=True)
        match = re.search(pattern, position_value)
        player_position = match.group(0)
        position.append(player_position)

    
        league_links = entry.find_all('a', href=True)
        found = False
        for link in league_links:
            if '/transfers/wettbewerb/' in link['href']:
                league_name = link.get_text(strip=True)
                found = True
                break  # Exit the loop if found
        if not found:
            league_name = 'N/A'
        Liga.append(league_name)  
        


#Make into Pandas DataFrame
transfermarkt = pd.DataFrame(
    {
        "Player name": name,
        "Age": age,
        "club": club,
        "market value": market_value,
        "League": Liga,
        "position": position
    }
)

#Making dataFrame into csv
transfermarkt.head()
transfermarkt.to_csv("TransferMarkt.csv", index=False)



    

