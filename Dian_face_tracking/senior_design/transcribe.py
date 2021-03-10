#uses AWS to transcribe audio stored in AWS s3 storage to text
#to do: add voice input and analyze output
#change job name for each run
from __future__ import print_function
import time
import boto3
import urllib
import json

#set up AWS transcribe
transcribe = boto3.client('transcribe',
	region_name='us-east-2',
	aws_secret_access_key ='Ix+1mniYH/8N6krJuR4j5Cp574ltULGAgF1KhTks',
    aws_access_key_id = 'AKIAIJ54TA7WTPFPFFHA')
job_name = "Transcribe30"
job_uri = "https://s3.us-east-2.amazonaws.com/sound-joelmussell/Merry+Christmas-SoundBible.com-1120316507.mp3"

#start transcription
transcribe.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': job_uri},
    MediaFormat='mp3',
    LanguageCode='en-US'
)

#wait for AWS to respond
while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    print("Not ready yet...")
    time.sleep(5)

#record transcribe output
text = transcribe.get_transcription_job(TranscriptionJobName=job_name)

#open json reults file  and extract information
jsonText = urllib.urlopen(text['TranscriptionJob']['Transcript']['TranscriptFileUri']).read()
index = jsonText.find('\"transcript\"') + 14
transcript = jsonText[index:]
index = transcript.find('\"')
transcript = transcript[:index]
print(transcript)