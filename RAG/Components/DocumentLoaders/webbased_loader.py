from langchain_community.document_loaders import WebBaseLoader




url = 'https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR6HXPB9/ref=sr_1_1_sspa?crid=14XZ653DXG5J&dib=eyJ2IjoiMSJ9.VwAQgXeiXX8YDAtxqvsM2JMqcy9NJSfIn6ty4GizGt5D5CYvFx7c2lF2MG7Kq73S2hONFP96QurLIDJ59JBHJtgrDmFx3F6__vNaO5YqRYRHv_mZSqlTnDNsxyKo8DDaAUJPt8Diz2S8g0f5XK4OjQWDYaC3U8DH3LoKNWPR1MbwjbOIMlhCrK3ApVWTeywUBY8qv7pwASFXFbqYt8xlPDE8SnvopM4gdwGVt2829zg.UJZmfWT-Iyro13VtjCyRK-wpEF3qRkEbv8p5Zgow9T4&dib_tag=se&keywords=macbook%2Bneo&qid=1781020843&sprefix=macbook%2Caps%2C1750&sr=8-1-spons&aref=bFzgkl4ViZ&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'


loader = WebBaseLoader(url)

web_data = loader.load()
print(web_data[0].page_content)