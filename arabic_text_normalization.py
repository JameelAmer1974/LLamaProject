from camel_tools.utils.normalize import normalize_unicode,  normalize_alef_ar, normalize_teh_marbuta_ar
#from farasa.segmenter import FarasaSegmenter
from pyarabic.araby import normalize_ligature
from camel_tools.tokenizers.word import simple_word_tokenize

# Initialize Farasa
#segmenter = FarasaSegmenter(interactive=True)

def normalize_arabic_text(text):
    """ Normalize and tokenize Arabic text """
    text = normalize_unicode(text)  # Normalize Unicode characters
    text = normalize_alef_ar(text)  # Normalize different Alef forms (إ, أ → ا)
    text = normalize_teh_marbuta_ar(text)  # Normalize Teh Marbuta (ة → ه)
    text = normalize_ligature(text)  # Normalize Arabic ligatures (e.g., "ﻻ" → "لا")
    text = simple_word_tokenize(text)  # Tokenize Arabic words
    return text

if __name__ == '__main__':
    text = "السَّلامُ عَلَيكُم ورَحمَةُ اللَّهِ"
    clean_text = normalize_arabic_text(text)
    print(clean_text)
