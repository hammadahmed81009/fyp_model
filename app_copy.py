from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import re
import string
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.special import expit as sigmoid

app = Flask(__name__)

MODEL_NAME = "bert-base-uncased"
model_path = "./model_path/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=8)


# Initialize MultiLabelBinarizers
mlb_aspects = MultiLabelBinarizer()
mlb_response = MultiLabelBinarizer()

aspect_labels = ["quality", "price", "delivery", "service", "satisfaction"]
sentiment_labels = [
    "positive",
    "negative",
    "neutral",
]  # Modify as per your model's training


mlb_aspects.fit(aspect_labels)
mlb_response.fit(sentiment_labels)


# Define preprocessing functions
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub("[0-9]+", "", text)
    return text


def convert_to_lower_case(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    return text


# List of stopwords omitted for brevity, please include your list here
stopwords = [
    "a",
    "aaaa",
    "aaj",
    "ab",
    "abi",
    "above",
    "acha",
    "aese",
    "ai",
    "aik",
    "all",
    "am",
    "an",
    "and",
    "any",
    "ap",
    "are",
    "ati",
    "aur",
    "aye",
    "ayi",
    "aysa",
    "b",
    "baad",
    "barh",
    "bata",
    "bhi",
    "bilkl",
    "bol",
    "both",
    "bta",
    "but",
    "by",
    "chal",
    "chle",
    "could",
    "couldn't",
    "da",
    "dain",
    "dana",
    "de",
    "dekh",
    "dena",
    "dey",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "dono",
    "down",
    "during",
    "e",
    "each",
    "few",
    "for",
    "from",
    "further",
    "ga",
    "gai",
    "gaya",
    "gi",
    "gy",
    "gya",
    "gye",
    "gyi",
    "h",
    "ha",
    "haan",
    "hai",
    "hain",
    "han",
    "hay",
    "he",
    "hein",
    "her",
    "hers",
    "herself",
    "hi",
    "him",
    "himself",
    "his",
    "ho",
    "hoa",
    "hone",
    "hota",
    "hotay",
    "hotey",
    "hoty",
    "houn",
    "hova",
    "how",
    "how's",
    "hu",
    "hui",
    "hum",
    "hun",
    "hwa",
    "hy",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "inhi",
    "inko",
    "insey",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "itna",
    "its",
    "itself",
    "ja",
    "jaa",
    "jab",
    "jana",
    "jata",
    "jati",
    "jaye",
    "jb",
    "jbk",
    "je",
    "jee",
    "jo",
    "jye",
    "k",
    "ka",
    "kafi",
    "kar",
    "karao",
    "karna",
    "karwao",
    "kary",
    "kase",
    "kay",
    "kb",
    "kch",
    "khud",
    "ki",
    "kis",
    "kiu",
    "kiun",
    "kiya",
    "kl",
    "kn",
    "ko",
    "koi",
    "kon",
    "kr",
    "kri",
    "krna",
    "krne",
    "krny",
    "krta",
    "ku",
    "kuch",
    "kya",
    "kyun",
    "kyunkey",
    "le",
    "let's",
    "lg",
    "li",
    "liye",
    "lolxxxx",
    "lya",
    "lye",
    "ma",
    "mai",
    "main",
    "me",
    "mera",
    "meri",
    "mil",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "na",
    "nahin",
    "nai",
    "nay",
    "ne",
    "no",
    "nope",
    "nor",
    "not",
    "ny",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "ovi",
    "own",
    "oye",
    "paa",
    "par",
    "para",
    "pata",
    "pe",
    "per",
    "phir",
    "phr",
    "pr",
    "q",
    "r",
    "raha",
    "rahay",
    "rahi",
    "rai",
    "re",
    "reh",
    "rha",
    "rhy",
    "roko",
    "sa",
    "sahi",
    "sai",
    "saktah",
    "same",
    "san",
    "saray",
    "say",
    "sb",
    "se",
    "see",
    "sent",
    "sey",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "sy",
    "tak",
    "teen",
    "teeno",
    "tha",
    "than",
    "that",
    "that's",
    "thay",
    "the",
    "their",
    "theirs",
    "them",
    "thi",
    "thy",
    "to",
    "toh",
    "tou",
    "tu",
    "tumhara",
    "tumko",
    "un",
    "us",
    "uski",
    "waisay",
    "wala",
    "wali",
    "wli",
    "wo",
    "woh",
    "wohi",
    "ya",
    "yeh",
    "yehi",
    "yep",
    "yha",
    "you",
    "zyada",
]

dictStopWords = {}  # global variable
forFastTextData = []


def removeStopWordss(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    text = text.lower()
    wordList = str(text).split()
    filtered_words = [word for word in wordList if word not in stopwords]
    newSentence = " ".join(filtered_words)
    forFastTextData.append(newSentence.split())
    return newSentence


def replacing_characters(word):
    """Input Parameter:
    word: word from the sentences"""

    word = re.sub(r"ain$", r"ein", word)
    word = re.sub(r"ai", r"ae", word)
    word = re.sub(r"ay$", r"e", word)
    word = re.sub(r"ey$", r"e", word)
    word = re.sub(r"aa+", r"aa", word)
    word = re.sub(r"e+", r"ee", word)
    word = re.sub(r"ai", r"ahi", word)  # e.g "sahi and sai nahi"
    word = re.sub(r"ai", r"ahi", word)
    word = re.sub(r"ie$", r"y", word)
    word = re.sub(r"^es", r"is", word)
    word = re.sub(r"a+", r"a", word)
    word = re.sub(r"j+", r"j", word)
    word = re.sub(r"d+", r"d", word)
    word = re.sub(r"u", r"o", word)
    word = re.sub(r"o+", r"o", word)
    if not re.match(r"ar", word):
        word = re.sub(r"ar", r"r", word)
        word = re.sub(r"iy+", r"i", word)
        word = re.sub(r"ih+", r"eh", word)
        word = re.sub(r"s+", r"s", word)
    if re.search(r"[rst]y", "word") and word[-1] != "y":
        word = re.sub(r"y", r"i", word)
    if re.search(r"[^a]i", word):
        word = re.sub(r"i$", r"y", word)
    if re.search(r"[a-z]h", word):
        word = re.sub(r"h", "", word)
    return word


def preprocess_text(text):
    text = remove_punct(text)
    text = convert_to_lower_case(text)
    text = removeStopWordss(text)
    text = replacing_characters(text)
    return text


def predict_sentiment(review_text):
    preprocessed_text = preprocess_text(review_text)
    inputs = tokenizer.encode_plus(
        preprocessed_text,
        add_special_tokens=True,
        return_tensors="tf",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    predictions = model.predict(
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )[0]
    probabilities = sigmoid(predictions)
    return probabilities


def decode_probabilities(probabilities):
    aspect_probs = probabilities[:5]
    sentiment_probs = probabilities[5:]
    predicted_sentiment = sentiment_labels[np.argmax(sentiment_probs)]

    # Assuming threshold
    threshold = 0.5
    predicted_aspects = [
        aspect_labels[i] for i, prob in enumerate(aspect_probs) if prob > threshold
    ]

    response = {aspect: "neutral" for aspect in aspect_labels}
    for aspect in predicted_aspects:
        response[aspect] = predicted_sentiment

    return response


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    review_text = data["review"]
    probabilities = predict_sentiment(review_text)
    response_dict = decode_probabilities(probabilities)
    return jsonify(response_dict)


@app.route("/predict", methods=["POST"])
def predict():
    print("PRINTING FOR DEBUGGIN")
    data = request.get_json(force=True)
    review_text = data["review"]
    probabilities = predict_sentiment(
        review_text
    )  # Ensure predict_sentiment returns the probabilities as shown earlier
    print("haha")
    print(probabilities)
    # response = decode_probabilities(probabilities)
    return jsonify(probabilities)


if __name__ == "__main__":
    app.run(debug=True)
