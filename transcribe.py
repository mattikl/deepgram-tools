import click
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
import logging
from datetime import datetime
import os

config = DeepgramClientOptions(
    verbose=logging.SPAM,
)
deepgram = DeepgramClient("", config)

@click.command()
@click.option('--model', help='Model to use', default='nova-2')
@click.argument('filename', type=click.Path(exists=True))
def transcribe(model, filename):
    print(f'Transcribing with model {model}')
    name, extension = os.path.splitext(filename)
    outfile = f"{name}_{model}.json"
    try:
        with open(filename, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model=model,
            smart_format=True,
            utterances=True,
            punctuate=True,
            language="en-US",
        )

        before = datetime.now()
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        after = datetime.now()

        with open(outfile, "w") as file:
            file.write(response.to_json(indent=4) + "\n")

        result = response['results'].channels[0].alternatives[0]

        difference = after - before
        print(f"Transcription: '{result.transcript}' in {difference.seconds}s with confidence {result.confidence}")
        print(f"Written to {outfile}")
        return result.transcript

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    transcribe()