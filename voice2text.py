def transcribe_audio(file_path):
    from faster_whisper import WhisperModel

    model_size = "large"
    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(file_path, beam_size=5)

    output = f"Detected language '{info.language}' with probability {info.language_probability}\n"
    for segment in segments:
        output += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)

    return output

def write_output_to_file(output, output_file):
    with open(output_file, 'w') as f:
        f.write(output)

def main():
    file_path = "assets/sample3.wav"
    output = transcribe_audio(file_path)
    write_output_to_file(output, "voice2text.txt")

if __name__ == "__main__":
    main()