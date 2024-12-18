from pydub import AudioSegment
from pydub.silence import detect_silence
from faster_whisper import WhisperModel
import os

def find_silence_near_mark(audio, mark_ms, search_window=2000, silence_thresh=-50, min_silence_len=500):
    """
    Attempt to find a silence interval near a given mark (in ms).
    We look for silence within ±search_window ms of the mark.
    """
    start_search = max(0, mark_ms - search_window)
    end_search = min(len(audio), mark_ms + search_window)

    silence_regions = detect_silence(
        audio[start_search:end_search],
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if silence_regions:
        return choose_best_silence_region(silence_regions, start_search, mark_ms)

    return None

def choose_best_silence_region(silence_regions, offset, mark_ms):
    """
    Among the detected silence regions, pick the one closest to mark_ms.
    """
    best_region = None
    best_distance = float('inf')
    for (s_start, s_end) in silence_regions:
        global_s_start = s_start + offset
        global_s_end = s_end + offset

        candidate_points = [global_s_start, global_s_end, (global_s_start + global_s_end)/2]
        for point in candidate_points:
            dist = abs(point - mark_ms)
            if dist < best_distance:
                best_distance = dist
                best_region = (global_s_start, global_s_end)

    if best_region:
        return (best_region[0] + best_region[1]) // 2
    return None

def find_best_silence_point(audio, initial_mark, silence_thresh, min_silence_len, search_window=2000, max_shift_attempts=3, shift_step_ms=500):
    """
    Try to find the best silence point starting from initial_mark.
    """
    silence_point = find_silence_near_mark(audio, initial_mark, search_window, silence_thresh, min_silence_len)
    if silence_point is not None:
        return silence_point

    for attempt in range(1, max_shift_attempts + 1):
        forward_mark = initial_mark + attempt * shift_step_ms
        if forward_mark < len(audio):
            silence_point = find_silence_near_mark(audio, forward_mark, search_window, silence_thresh, min_silence_len)
            if silence_point is not None:
                return silence_point

        backward_mark = initial_mark - attempt * shift_step_ms
        if backward_mark > 0:
            silence_point = find_silence_near_mark(audio, backward_mark, search_window, silence_thresh, min_silence_len)
            if silence_point is not None:
                return silence_point

    return initial_mark

def split_audio_on_silence_intervals(file_path, chunk_length=10, silence_thresh=-50, min_silence_len=500, search_window=2000):
    """
    Split the audio file into chunks of approximately 'chunk_length' seconds,
    splitting at silence intervals.
    """
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    chunk_length_ms = chunk_length * 1000

    boundaries = [0]
    current_mark = chunk_length_ms

    while current_mark < total_length_ms:
        silence_point = find_best_silence_point(
            audio, current_mark,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            search_window=search_window,
            max_shift_attempts=3,
            shift_step_ms=500
        )

        boundaries.append(silence_point)
        current_mark = silence_point + chunk_length_ms

    boundaries.append(total_length_ms)

    chunks = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        chunk = audio[start:end]
        chunks.append(chunk)

    return chunks

def transcribe_audio_chunk(audio_segment, model, global_start):
    """
    Transcribe a single audio chunk using Faster Whisper and adjust timestamps globally.
    """
    temp_filename = "temp_chunk.wav"
    audio_segment.export(temp_filename, format="wav")
    segments, info = model.transcribe(temp_filename, beam_size=5)
    os.remove(temp_filename)

    output = ""
    for segment in segments:
        global_segment_start = global_start + segment.start
        global_segment_end = global_start + segment.end
        output += "[%.2fs -> %.2fs] %s\n" % (global_segment_start, global_segment_end, segment.text)
    return output, info

def transcribe_audio(file_path):
    model_size = "large"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    chunks = split_audio_on_silence_intervals(
        file_path,
        chunk_length=10,
        silence_thresh=-50,
        min_silence_len=500,
        search_window=2000
    )

    combined_output = ""  # Combined output with global timestamps
    global_start = 0  # Track the global start time of each chunk
    language_detected = None
    language_probability = None

    for idx, chunk in enumerate(chunks):
        print(f"\nProcessing Chunk {idx+1}...")
        
        # Transcribe the chunk with the current global start time
        chunk_output, info = transcribe_audio_chunk(chunk, model, global_start)
        
        # For the first chunk, capture language info
        if idx == 0:
            language_detected = info.language
            language_probability = info.language_probability
        
        # Append transcriptions directly (no chunk headers)
        combined_output += chunk_output

        # Update global start time for the next chunk
        global_start += len(chunk) / 1000  # Convert chunk length to seconds

    # Add language detection to the top if available
    if language_detected and language_probability:
        header = f"Detected language '{language_detected}' with probability {language_probability}\n\n"
    else:
        header = "Language detection not available.\n\n"

    return header + combined_output


def write_output_to_file(output, output_file):
    with open(output_file, 'w') as f:
        f.write(output)

def main():
    file_path = "assets/sample.wav"  # Path to your input audio file
    output = transcribe_audio(file_path)
    write_output_to_file(output, "output/voice2text.txt")

if __name__ == "__main__":
    main()
