import pdfplumber
import csv
import re
pdf_file = r"C:\Users\prithvika\OneDrive\Desktop\springboard\Merchant-Category-Codes.pdf"
output_csv = r"C:\Users\prithvika\OneDrive\Desktop\springboard\extracted_mcc_codes.csv"
mcc_data = []

# Regular expression for validating 4-digit MCC codes
mcc_pattern = re.compile(r"^\d{4}$")

# Extract MCC codes and categories from tables with headers "MCC" and "Description"
with pdfplumber.open(pdf_file) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            headers = table[0]
            if "MCC" in headers and "Description" in headers:
                mcc_index = headers.index("MCC")
                description_index = headers.index("Description")
                current_description = ""
                for row in table[1:]:
                    mcc = row[mcc_index].strip() if row[mcc_index] else ""
                    description = row[description_index].strip() if row[description_index] else ""
                    # Handle multi-line descriptions by checking if we have a continuation
                    if description:
                        if current_description:
                            current_description += " " + description
                        else:
                            current_description = description
                    
                    # If we encounter a new MCC code and it's a valid 4-digit code, store the entry
                    if mcc and mcc_pattern.match(mcc) and current_description:
                        mcc_data.append([mcc, current_description])
                        current_description = "" 

# Save extracted data to a CSV file
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)  
    writer.writerow(["MCC Code", "Description"])
    for row in mcc_data:
        # Ensure that descriptions with commas or newlines are quoted automatically by CSV writer
        writer.writerow([row[0], row[1].replace("\n", " ")])  # Replace newlines within description

print(f"MCC codes and descriptions have been extracted and saved to {output_csv}")
