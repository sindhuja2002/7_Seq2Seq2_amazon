import requests
import csv
from bs4 import BeautifulSoup

# Specify the number of pages to scrape
num_pages = 5

# Create a CSV file to store the scraped data
csv_file = open('amazon_data.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Title', 'Price', 'Rating'])

# Scrape data from each page
for page in range(1, num_pages + 1):
    # Specify the URL of the Amazon page
    url = f"https://www.amazon.com/s?k=laptop&page={page}"

    # Send a GET request to the URL and retrieve the page content
    response = requests.get(url)
    content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")

    # Find all product listings on the page
    product_listings = soup.find_all("div", {"data-component-type": "s-search-result"})

    # Extract data from each product listing
    for product in product_listings:
        product_title = product.find("span", class_="a-size-medium").text.strip()
        product_price = product.find("span", class_="a-offscreen").text.strip()
        product_rating = product.find("span", class_="a-icon-alt").text.strip()

        # Write the extracted data to the CSV file
        csv_writer.writerow([product_title, product_price, product_rating])

# Close the CSV file
csv_file.close()
