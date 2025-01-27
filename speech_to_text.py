import os
import subprocess
from google.cloud import speech
from google.oauth2 import service_account

# Load Google Cloud credentials
client_file = "resp.json"
creds = service_account.Credentials.from_service_account_file(client_file)
speech_client = speech.SpeechClient(credentials = creds)

def merge_chunks(chunks, output_path):
    """
    Merges multiple audio chunks into a single file.
    """
    with open(output_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

def convert_to_wav(input_path, output_path):
    """
    Converts a given audio file to WAV format with 16kHz sample rate and mono channel.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert audio to WAV: {e.stderr.decode()}")

def merge_with_overlap(previous, current):
    """
    Merges two overlapping transcriptions by checking letter-by-letter overlaps.
    Handles exact matches and partial word overlaps.
    """
    previous_words = previous.split()
    current_words = current.split()

    if not previous_words or not current_words:
        return previous + " " + current

    # Last word of previous chunk and first word of current chunk
    last_word_prev = previous_words[-1]
    first_word_curr = current_words[0]

    # Check for exact match
    if last_word_prev == first_word_curr:
        return previous + " " + " ".join(current_words[1:])

    # Check for partial match
    overlap_index = -1
    for i in range(len(last_word_prev)):
        if first_word_curr.startswith(last_word_prev[i:]):
            overlap_index = i
            break

    if overlap_index != -1:
        # Merge partial match
        merged_word = last_word_prev[:overlap_index] + first_word_curr
        return " ".join(previous_words[:-1] + [merged_word] + current_words[1:])

    # No match, just concatenate
    return previous + " " + current

def transcribe_audio_with_overlap(wav_path):
    """
    Transcribes a WAV file in overlapping chunks using Google Speech-to-Text API.
    Handles incomplete words across chunk boundaries.
    """
    transcription = ""
    overlap_duration = int(0.5 * 16000 * 2)  # 0.5 second overlap
    chunk_duration = int(5 * 16000 * 2)  # 5 seconds

    try:
        with open(wav_path, "rb") as audio_file:
            audio_content = audio_file.read()

        previous_transcription = ""
        start = 0

        while start < len(audio_content):
            # Define the end of the chunk with overlap
            end = min(start + chunk_duration + overlap_duration, len(audio_content))
            chunk = audio_content[start:end]

            # Configure Google Speech-to-Text API
            audio = speech.RecognitionAudio(content=chunk)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )

            # Perform transcription
            response = speech_client.recognize(config=config, audio=audio)

            chunk_transcription = " ".join(
                [result.alternatives[0].transcript for result in response.results]
            )

            # Merge transcriptions across overlapping chunks
            if previous_transcription:
                transcription += merge_with_overlap(previous_transcription, chunk_transcription)
            else:
                transcription += chunk_transcription

            start += chunk_duration
            previous_transcription = chunk_transcription

        return transcription.strip()
    except Exception as e:
        raise RuntimeError(f"Error during audio transcription: {e}")
