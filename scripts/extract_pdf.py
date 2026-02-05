# -*- coding: utf-8 -*-
import fitz
import sys

pdf_path = r'C:\Users\citiz\OneDrive\Documents\Papers\Heins et al. 2025 AXIOM.pdf'
output_path = r'C:\Users\citiz\Documents\research\axiom_text.txt'

doc = fitz.open(pdf_path)
with open(output_path, 'w', encoding='utf-8') as f:
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        f.write(f'=== PAGE {page_num + 1} ===\n')
        f.write(text)
        f.write('\n\n')

print(f"Extracted {min(10, len(doc))} pages to {output_path}")
