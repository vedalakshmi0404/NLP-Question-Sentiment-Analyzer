import tkinter as tk
from tkinter import scrolledtext
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")

sia = SentimentIntensityAnalyzer()

def generate_questions(paragraph):
    paragraph = paragraph.strip()
    output_text.delete("1.0", tk.END)

    if not paragraph:
        output_text.insert(tk.END, "Please enter a paragraph.")
        return []

    doc = nlp(paragraph)
    sentences = list(doc.sents)
    questions = []

    for sent in sentences:
        entities = [ent.text for ent in sent.ents]
        if entities:
            highlight = sent.text.replace(entities[0], f"<hl> {entities[0]} <hl>")
            input_text_q = f"generate question: {highlight}"
        else:
            input_text_q = f"generate question: {sent.text.strip()}"

        input_ids = tokenizer.encode(input_text_q, return_tensors="pt", max_length=512, truncation=True)
        output = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        questions.append(question)

    return questions

def analyze_sentiment(paragraph):
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]
    
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentiment_sentences = {"Positive": [], "Negative": [], "Neutral": []}
    
    for sentence in sentences:
        if sentence.strip():
            sentiment_score = sia.polarity_scores(sentence)
            compound_score = sentiment_score['compound']
            
            if compound_score > 0.1:
                sentiment_counts["Positive"] += 1
                sentiment_sentences["Positive"].append(sentence.strip())
            elif compound_score < -0.1:
                sentiment_counts["Negative"] += 1
                sentiment_sentences["Negative"].append(sentence.strip())
            else:
                sentiment_counts["Neutral"] += 1
                sentiment_sentences["Neutral"].append(sentence.strip())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['green', 'red', 'gray'])
    ax.set_title("Sentiment Analysis of Paragraph")
    plt.show()

    return sentiment_counts, sentiment_sentences

def on_generate():
    paragraph = text_input.get("1.0", tk.END)
    output_text.delete("1.0", tk.END)

    questions = generate_questions(paragraph)
    
    output_text.insert(tk.END, "Generated Questions:\n\n")
    for idx, q in enumerate(questions, 1):
        output_text.insert(tk.END, f"Q{idx}: {q}\n")
    
    sentiment_result, sentiment_sentences = analyze_sentiment(paragraph)
    
    output_text.insert(tk.END, "\n\nSentiment Analysis by Sentence:\n")
    for sentiment_type in ["Positive", "Negative", "Neutral"]:
        output_text.insert(tk.END, f"\n{sentiment_type} Sentences ({sentiment_result[sentiment_type]}):\n")
        for sentence in sentiment_sentences[sentiment_type]:
            output_text.insert(tk.END, f"- {sentence}\n")

root = tk.Tk()
root.title("NLP Question Generator and Sentiment Analyzer")
root.geometry("800x700")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Enter Paragraph:", bg="#f0f0f0", font=("Helvetica", 12, "bold")).pack()
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("Helvetica", 10))
text_input.pack(padx=10, pady=5)

generate_button = tk.Button(root, text="Generate Questions & Analyze Sentiment", command=on_generate, bg="blue", fg="white", font=("Helvetica", 12))
generate_button.pack(pady=10)

tk.Label(root, text="Output:", bg="#f0f0f0", font=("Helvetica", 12, "bold")).pack()
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Helvetica", 10))
output_text.pack(padx=10, pady=5)

root.mainloop()
