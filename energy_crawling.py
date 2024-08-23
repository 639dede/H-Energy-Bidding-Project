import requests
from bs4 import BeautifulSoup

# Define the URL
url = "https://bd.kma.go.kr/kma2020/fs/energySelect1.do?menuCd=F050701000"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the data using BeautifulSoup and CSS selectors
    # The XPath /html/body/div[3]/div[2]/div/div/div/div[2]/div[3]/table/tbody/tr[3]/td[2]
    # translates to the following CSS selector:
    selector = '#wrap > div.sub_main > div.sub_main__con > div > h2'
    
    # Extract the data
    data = soup.select_one(selector)
    
    # Check if the data was found
    if data:
        print(data.text.strip())
    else:
        print("Data not found")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
    

							