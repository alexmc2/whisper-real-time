# Real Time Whisper Transcription

![Demo gif](demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply r

```bash
pip install -r requirements.txt
```

in an environment of your choosing.

Whisper also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

For more information on Whisper please see [https://github.com/openai/whisper](https://github.com/openai/whisper)

The code in this repository is public domain

## Running the Application

There are two ways to run this application:

### Web UI Mode

To run the application with a web-based user interface:

```bash
python web_output.py
```

This will start a web server at <http://localhost:5000> where you can view and interact with the transcriptions in your browser.

### Terminal Mode

To run the application with output directly in your terminal:

```bash
python transcibe_demo.py
```

This will display the transcriptions directly in your terminal window.
