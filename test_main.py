import subprocess
import os

# Test pro generování náhodného BMP souboru
def test_generate_random_bmp():
    result = subprocess.run(
        ["python", "main.py", "--generate-bmp", "100", "100", "./content/media/out/test/random_bmp.bmp"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to generate BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test/random_bmp.bmp"), "Random BMP file was not created"

# Test pro generování náhodného WAV souboru
def test_generate_random_wav():
    result = subprocess.run(
        ["python", "main.py", "--generate-wav", "1000", "./content/media/out/test/random_wav.wav"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to generate WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/random_wav.wav"), "Random WAV file was not created"

# Test pro generování sinus WAV souboru
def test_generate_sine_wav():
    result = subprocess.run(
        ["python", "main.py", "--generate-wav", "1000", "./content/media/out/test/sine_wav.wav", "--freq", "440"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to generate sine WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/sine_wav.wav"), "Sine WAV file was not created"

# Test pro konverzi BMP na WAV (1-bit BMP)
def test_bmp_to_wav_1bit():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/1bit.bmp", "./content/media/out/test/bmp_to_wav_1bit.wav"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP 1-bit to WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/bmp_to_wav_1bit.wav"), "WAV file was not created"

# Test pro konverzi BMP na WAV (4-bit BMP)
def test_bmp_to_wav_4bit():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/4bit.bmp", "./content/media/out/test/bmp_to_wav_4bit.wav"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP 4-bit to WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/bmp_to_wav_4bit.wav"), "WAV file was not created"

# Test pro konverzi BMP na WAV (8-bit BMP)
def test_bmp_to_wav_8bit():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/8bit.bmp", "./content/media/out/test/bmp_to_wav_8bit.wav"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP 8-bit to WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/bmp_to_wav_8bit.wav"), "WAV file was not created"

# Test pro konverzi BMP na WAV (24-bit BMP)
def test_bmp_to_wav_24bit():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/24bit.bmp", "./content/media/out/test/bmp_to_wav_24bit.wav"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP 24-bit to WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/bmp_to_wav_24bit.wav"), "WAV file was not created"

# Test pro konverzi WAV na BMP (WAV do BMP)
def test_wav_to_bmp():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/wav/32bit.wav", "./content/media/out/test/wav_to_bmp.bmp", "--dimensions", "100", "100"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert WAV to BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test/wav_to_bmp.bmp"), "BMP file was not created"




def test_generate_random_bmp_show():
    result = subprocess.run(
        ["python", "main.py", "--generate-bmp", "100", "100", "./content/media/out/test/random_bmp.bmp", "-s", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to generate BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test/random_bmp.bmp") and os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "Random BMP file was not created"


def test_generate_random_bmp_show_axes():
    result = subprocess.run(
        ["python", "main.py", "--generate-bmp", "100", "100", "./content/media/out/test/random_bmp.bmp", "-sx", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to generate BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test/random_bmp.bmp") and os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "Random BMP file was not created"

def test_convert_bmp_to_wav_and_display():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/24bit.bmp", "./content/media/out/test/bmp2wav.wav", "-s", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP to WAV: {result.stderr}"
    assert os.path.exists("./content/media/out/test/bmp2wav.wav") and os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "WAV file was not created"

def test_convert_wav_to_bmp_and_display():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/wav/32bit.wav", "./content/media/out/test/wav2bmp.bmp", "-s", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert WAV to BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test/wav2bmp.bmp") and os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "BMP file was not created"


def test_convert_bmp_to_wav_show_axes():
    result = subprocess.run(
        ["python", "main.py", "--convert", "./content/media/bmp/24bit.bmp", "./content/media/out/test/bmp2wav.wav", "-s", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to convert BMP to WAV with axes: {result.stderr}"
    assert os.path.exists("./content/media/out/test_mode/display_wav_output.png"), "WAV display with axes was not generated"

def test_show_existing_bmp():
    result = subprocess.run(
        ["python", "main.py", "-i", "./content/media/bmp/24bit.bmp", "-s", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to show existing BMP: {result.stderr}"
    assert os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "BMP display was not generated"

def test_show_existing_bmp_with_axes():
    result = subprocess.run(
        ["python", "main.py", "-i", "./content/media/bmp/24bit.bmp", "-s", "-sx", "--test"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Failed to show BMP with axes: {result.stderr}"
    assert os.path.exists("./content/media/out/test_mode/display_bmp_output.png"), "BMP display with axes was not generated"


if __name__ == "__main__":
    # Spusť všechny testy
    # test_generate_random_bmp()
    # test_generate_random_wav()
    # test_generate_sine_wav()
    # test_bmp_to_wav_1bit()
    # test_bmp_to_wav_4bit()
    # test_bmp_to_wav_8bit()
    # test_bmp_to_wav_24bit()
    test_wav_to_bmp()

