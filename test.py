
from google import genai

client = genai.Client(api_key="AIzaSyAV3t3l3DC8DLP9WJY14hTpKDdYVyrG4EA")

for m in client.models.list():
    print(m.name)