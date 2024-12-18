from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import re

def load_voice2text_file(file_path):
    """
    Load the transcribed voice2text file and extract timeline and sentences.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    entries = []
    pattern = re.compile(r"\[(\d+\.\d+s) -> (\d+\.\d+s)\]\s+(.*)")
    for line in data:
        match = pattern.match(line.strip())
        if match:
            start_time, end_time, sentence = match.groups()
            entries.append({
                "start_time": start_time,
                "end_time": end_time,
                "sentence": sentence
            })
    return entries

def translate_sentences_to_english(entries, llm):
    """
    Translate each Tamil sentence to concise and polished English using the LLM.
    """
    translated_entries = []
    
    for entry in entries:
        prompt_template = """
        You are an expert translator. Translate the following Tamil or mixed-language sentence into clear, meaningful, and fluent English.

        Do not include any explanations or comments. Provide only the translated sentence.

        Input:
        "{sentence}"

        Output:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        final_prompt = prompt.format(sentence=entry["sentence"])
        
        print(f"Translating: {entry['sentence']}")
        translation = llm(final_prompt)
        
        translated_entries.append({
            "start_time": entry["start_time"],
            "end_time": entry["end_time"],
            "translation": translation.strip()
        })
    return translated_entries

def save_translated_entries(translated_entries, output_file):
    """
    Save the translated entries to a file with the correct timeline.
    """
    with open(output_file, 'w') as file:
        for entry in translated_entries:
            file.write(f"[{entry['start_time']} -> {entry['end_time']}] {entry['translation']}\n")

def main():
    # Path to the input and output files
    input_file = "output/voice2text.txt"
    output_file = "output/translated_voice2text.txt"
    
    # Load the Ollama LLM model using LangChain and Mistral 7B
    llm = Ollama(model="mistral:7b")  # Load Mistral 7B model from Ollama
    
    # Load the transcribed data
    entries = load_voice2text_file(input_file)
    
    # Translate sentences
    translated_entries = translate_sentences_to_english(entries, llm)
    
    # Save the translations
    save_translated_entries(translated_entries, output_file)
    print(f"Translation complete. Saved to {output_file}.")

if __name__ == "__main__":
    main()
