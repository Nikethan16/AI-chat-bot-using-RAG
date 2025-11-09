
from utils.pdf_parser import extract_text_from_pdf
text = extract_text_from_pdf("C:\\Users\\niket\\Downloads\\NeoStats AI Engineer Use Case\\NeoStats AI Engineer Use Case\\healthcare_rag_chatbot\\data\\raw_pdfs\\01 - Healthy Diet and Nutrition.pdf")
print(text[:500])
