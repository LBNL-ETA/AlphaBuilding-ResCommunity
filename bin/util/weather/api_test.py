import json
import requests

# # response = requests.get("https://api.weather.gov/stations")
# response = requests.get("https://api.weather.gov/gridpoints/MTR/95,89/forecast/hourly")

# # print(response.__attrs__)
# # print(json.loads(response._content))
# # print(response.text)

# json_data = json.loads(response.text)

# print(json_data)

url = requests.get("https://graphical.weather.gov/xml/sample_products/browser_interface/ndfdXMLclient.php?lat=38.99&lon=-77.01&product=time-series&begin=2020-06-09T00:00:00&end=2020-06-10T00:00:00&sky=sky&mint=mint")
htmltext = url.text

print(htmltext)