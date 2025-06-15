import wave
import struct
import math

def create_alert_sound(filename="alert.wav", duration=1.0, frequency=1000.0):
    """Create a simple alert sound file"""
    # Audio parameters
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # Create the audio data
    audio_data = []
    for i in range(num_samples):
        # Generate a sine wave
        value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        audio_data.append(value)
    
    # Convert to bytes
    audio_bytes = struct.pack('<%dh' % len(audio_data), *audio_data)
    
    # Write to WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

if __name__ == "__main__":
    create_alert_sound()
    print("Alert sound file created: alert.wav") 