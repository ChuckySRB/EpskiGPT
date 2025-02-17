import pdfplumber

def extract_page_from_pdf(pdf_file, current_page):
    # Отвори PDF фајл
    with pdfplumber.open(pdf_file) as pdf:
        # Изврши извлачење текста од странице start_page до end_page
        text = ""
        for page_num in range(current_page-1, current_page):
            page = pdf.pages[page_num]
            text += page.extract_text()

    return text

def clean_page(text: str) -> str:
    # Уклоните нежељене делове текста као што су наслови, футери и заглавља
    cleaned_text = []
    naslov = False
    prvo_slovo = ""
    for line in text.splitlines():
        if line[0].isdigit() and line.isupper():
            naslov = True
            cleaned_text.append("\n\n")
            continue
        if naslov:
            naslov = False
            prvo_slovo = line[0]
            continue
        if prvo_slovo != "":
            line = prvo_slovo + line
            prvo_slovo = ""
        if line.isupper():
            continue
        
        cleaned_text.append(line)

    return '\n'.join(cleaned_text)


if __name__ == "__main__":

    # Промените овде са својим PDF путем
    pdf_file = "data/antologija_narodnih_pesama.pdf"
    output_file = "data/antologija_narodnih_pesama.txt"

    # Дефинишите странице које желите да обрадите (од странице а до странице б)
    start_page = 111  # Почетна страница
    end_page = 889  # Крајња страница (погледајте колико вам је потребно)

    # Извлачење текста из PDF-а
    with open(output_file, "w", encoding="utf-8") as output_file:
        for current_page in range(start_page, end_page+1):
            text_page = extract_page_from_pdf(pdf_file, current_page)
            cleaned_text_page = clean_page(text_page)
            output_file.write(cleaned_text_page)
            if current_page%50 == 0:
                print(f"Обрађена страница {current_page}")
        


    print(f"Текст је успешно извучен и сачуван у {output_file}")
