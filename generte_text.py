from dotenv import load_dotenv
from pprint import pprint
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from arabert.preprocess import ArabertPreprocessor
import requests
import os
load_dotenv()
# Load GPT-2 model and tokenizer for Arabic
model_name = "./sss-arabic_gpt2_finetuned"  # Pre-trained Arabic GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(query="SSS Document"):
    # Create parser and summarizer
    input_ids = tokenizer.encode(query, return_tensors="pt", padding=True).to("cpu")
    # Generate text
    output = model.generate(
        input_ids,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def generate_text(text,query="SSS Document"):
    # Create parser and summarizer
    input_ids = tokenizer.encode(query, return_tensors="pt", padding=True).to("cpu")
    # Generate text
    output = model.generate(
        input_ids,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    query ="""
الانتخابات السورية
    """
    generated_text = generate_text(query)
    print(generated_text)

    """
    الحرب السورية مستمرة منذ أكثر من ثلاث سنوات وقال المرصد السوري لحقوق الإنسان المعارض ، ومقره بريطانيا ، إن الغارات الجوية التي يشنها التحالف الدولي بقيادة الولايات المتحدة على مواقع تنظيم " الدولة الإسلامية " في محافظة دير الزور أسفرت عن مقتل العشرات من مسلحي التنظيم . وأضاف المرصد أن الغارات أسفرت أيضا عن تدمير عدد من المنازل في المنطقة . وقال رامي عبد الرحمن مدير المرصد ، الذي يتخذ من بريطانيا مقرا له ، لبي بي سي إن الضربات الجوية استهدفت مواقع للتنظيم في مدينة البوكمال ، شرقي سوريا . مواضيع قد تهمك نهاية وأضاف أن طائرات التحالف شنت أيضا غارات على مناطق أخرى في سوريا ، بما في ذلك مدينة الرقة ، التي يسيطر عليها مسلحو التنظيم ، بحسب المرصد . وأشار المرصد إلى أن غارات التحالف استهدفت أيضا مواقع لمسلحي التنظيم في الرقة ودير الزور . وكانت القوات الحكومية السورية ، مدعومة بغطاء جوي روسي ، قد شنت في وقت سابق من الشهر الجاري ، عملية عسكرية لاستعادة السيطرة على مدينة تدمر الأثرية ، الواقعة على بعد 50 كيلومترا غرب العاصمة السورية دمشق ، والتي تعد من أهم المواقع الأثرية في العالم . وكان التنظيم قد سيطر على تدمر في يونيو - حزيران
    """

