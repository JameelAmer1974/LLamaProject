from nltk.corpus import stopwords
import re
arabic_stopwords = set(stopwords.words('arabic'))
# Initialize Farasa
#segmenter = FarasaSegmenter(interactive=True)

def normalize_arabic_text(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", '', text)
    return text


if __name__ == '__main__':
    text = "السَّلامُ عَلَيكُم ورَحمَةُ اللَّهِ"
    clean_text = normalize_arabic_text(text)
    print(clean_text)
