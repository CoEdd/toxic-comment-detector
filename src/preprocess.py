def clean_text(text):
    import re
    import ftfy

    # Replace newlines, tabs, carriage returns with space
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix encoding artifacts
    text = ftfy.fix_text(text)

    return text

def preprocess_data(df):
    # Apply cleaning to the 'comment_text' column
    df['comment_text'] = df['comment_text'].apply(clean_text)
    return df