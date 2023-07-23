import gradio as gr
import os
from datetime import datetime
import logging
import sys
from llama_index import SimpleDirectoryReader
import llama_index.readers.file.base
import glob
import numpy as np
import soundfile as sf
import shutil
import openai
import json
import cv2

from llama_index import download_loader

ImageCaptionReader = download_loader('ImageCaptionReader')

openai.api_key = os.environ['OPENAI_API_KEY']

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# print('done processing import')
with open('config.json', encoding='utf8') as f:
    config = json.load(f)

def process_inputs(text: str, image: np.ndarray, video: str, audio: tuple, ):
    output = ""
    # # print('video', type(video), video)
    # # print('text', type(text), text)
    # # print('audio', type(audio), audio)
    # # print('image', type(image), image)
    if not text and image is not None and not video and audio is not None:
        return "Please upload at least one of the following: text, image, video, audio."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a folder named 'media_files' if it doesn't exist
    os.makedirs(f"media_files/{timestamp}", exist_ok=True)


    if video:
        video_path = os.path.join("media_files", f"{timestamp}/video.mp4")
        # copy from "video" to "video_path"

        shutil.copyfile(video, video_path)
        # os.rename(video_path, video_path)
        
        ffmpeg_cmd = f'ffmpeg -i {video_path} -vf "select=not(mod(n\,100))" -vsync vfr media_files/{timestamp}/frame_%03d.jpg'
        os.system(ffmpeg_cmd)

        output += "Video processed and saved.\n"
        print("Video processed and saved.")
        # gr.Interface.update("Video saved.")

    if text:
        text_path = os.path.join("media_files", f"{timestamp}/text.txt")
        with open(text_path, "w", encoding='utf8') as f:
            f.write(text)
        output += "Text processed and saved: " + text + "\n"
        # print("Text processed and saved: " + text + "")
        # gr.Interface.update("Text processed and saved: " + "")

    if audio is not None:
        sr, audio = audio
        audio_path = os.path.join("media_files", f"{timestamp}/audio.mp3")
        sf.write(audio_path, audio, sr)
        output += "Audio processed and saved.\n"
        print("Audio processed and saved.")
        # gr.Interface.update("Audio saved.")

    if image is not None:
        image_path = os.path.join("media_files", f"{timestamp}/image.png")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)
        output += "Image processed and saved.\n"
        print("Image processed and saved.")
        # gr.Interface.update("Image saved.")


    root = f"media_files/{timestamp}"
    
    image_caption_prompt = "Question: Describe what you see in this image and if there are any dangers or emergencies there any dangers and how sever they are. Answer:"

    text_files = glob.glob(f'{root}/*.txt')
    text_content = ''
    if text_files:
        # print('processing text_files ...')
        text_content = SimpleDirectoryReader(
            input_files=text_files,
            file_extractor={
                ".jpg": ImageCaptionReader(),
                ".png": ImageCaptionReader(),
                ".jpeg": ImageCaptionReader(),
                ".wav": llama_index.readers.file.video_audio_reader,
                ".mp4": llama_index.readers.file.video_audio_reader,
            }
        ).load_data()
        texts = [x.text for x in text_content]
        text = '\n\n'.join(texts)
        text_content = text.replace('"', "'").replace('\n', '.  ')
        # print('done processing text_files')

    image_files = glob.glob(f'{root}/*.png') + glob.glob(f'{root}/*.jpg')
    image_content = ''
    if image_files:
        # print('processing image_files ...')
        image_content = SimpleDirectoryReader(
            input_files=image_files,
            file_extractor={
                ".jpg": ImageCaptionReader(),
                ".png": ImageCaptionReader(),
                ".jpeg": ImageCaptionReader(),
                ".wav": llama_index.readers.file.video_audio_reader,
                ".mp4": llama_index.readers.file.video_audio_reader,
            }
        ).load_data()
        texts = [x.text for x in image_content]
        text = '\n\n'.join(texts)
        image_content = text.replace('"', "'").replace('\n', '.  ')
        # print('done processing image_files')

    audio_files = glob.glob(f'{root}/*.mp3')
    audio_content = ''
    if audio_files:
        # print('processing audio_files ...')
        audio_content = SimpleDirectoryReader(
            input_files=audio_files,
            file_extractor={
                ".jpg": ImageCaptionReader(),
                ".png": ImageCaptionReader(),
                ".jpeg": ImageCaptionReader(),
                ".mp3": llama_index.readers.file.video_audio_reader,
                ".mp4": llama_index.readers.file.video_audio_reader,
            }
        ).load_data()
        texts = [x.text for x in audio_content]
        text = '\n\n'.join(texts)
        audio_content = text.replace('"', "'").replace('\n', '.  ')
        # print('done processing audio_files')

    video_files = glob.glob(f'{root}/*.mp4')
    video_content = ''
    if video_files:
        # print('processing video_files ...')
        video_content = SimpleDirectoryReader(
            input_files=video_files,
            file_extractor={
                ".jpg": ImageCaptionReader(),
                ".png": ImageCaptionReader(),
                ".jpeg": ImageCaptionReader(),
                ".mp3": llama_index.readers.file.video_audio_reader,
                ".mp4": llama_index.readers.file.video_audio_reader,
            }
        ).load_data()
        texts = [x.text for x in video_content]
        text = '\n\n'.join(texts)
        video_content = text.replace('"', "'").replace('\n', '.  ')
        # print('done processing video_files')


    ar2en = {v:k for (k,v) in config["en2ar"].items()}
    emergencies_en = [ar2en[k] for k in config['redirects']]

    system_prompt = f"""I want you to act as a 911 operator that understands Arabic.
I will give you text and audio transcripts that the users upload in an emergency, and I need you to classify the different types of emergencies.
The incoming information could be Arabic or English, and you must output only in English.

The different types of emergencies are only one of {len(emergencies_en)}: {json.dumps(emergencies_en)}

I will give you the information provided by the user bellow, and you should classify from the {len(emergencies_en)} types of emergencies.
"""


    prompt = """

=== User information for emergency

"""
    if text_content:
        prompt += f'User text: "{text_content}"\n'
    if image_content:
        prompt += f'User uploaded an image of: "{image_content}"\n'
    if audio_content:
        prompt += f'User uploaded an audio, the text in that audio sounds like: "{audio_content} {video_content}" \n'

    prompt += """
=== End of user information for emergency

Now you must output only in JSON in the following format: {"emergency_class": string, "explaination_arabic": string}
Note that "explaination_arabic" must be in Arabic.

For the emergency_class, you must choose one of the following: """ + json.dumps(emergencies_en)

    # print('prompt', prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    # parse model JSON output
    content = completion.choices[0].message.content
    content = content.replace(",}", "}") # just in case
    # start from first "{" until the first "}"
    content = content[content.find("{") : content.find("}")+1]
    # print('ChatGPT response:', content)
    try:
        result = json.loads(content)
    except:
        result = {
            "emergency_class": "unknown",
            "explaination_arabic": "Could not parse output.: " + content
        }

    emergency_class_ar = config['en2ar'].get(result['emergency_class'], "غير معروف")
    redirects = config['redirects'].get(emergency_class_ar, ["<غير معروف>"])
    
    output = f"""نوع الحالة: {emergency_class_ar}


الجهات المسؤولة:
- """ + ('\n - '.join(redirects)) + f"\n\nالشرح: {result['explaination_arabic']}"

    return output if output else "No input provided."

video_input = gr.inputs.Video(optional=True, label="Input Video")
text_input = gr.inputs.Textbox(lines=3, optional=True, label="Input Text")
audio_input = gr.inputs.Audio(optional=True, label="Input Audio")
image_input = gr.inputs.Image(optional=True, label="Input Image")

output_text = gr.outputs.Textbox(label="Output Text")
examples = [
    # text_input, image_input, video_input, audio_input
    ["", None,"data/fire_at_gas_station.mp4", None,],
    ["", "data/small-car-accident.jpg", None, None],
    ["", "data/electrical-fire.jpg", None, None],
    ["", "data/major-car-accident.jpg", None, None],
    ["", "data/gettyimages-50908538-612x612.jpg", None, None],
    ["", None, None, "data/fire_at_gas_station.mp3",],
    ["السلام عليكم، أنا أتصل لأبلغ عن حريق كبير في مبنى سكني بشارع المنصور. يبدو أن النيران اندلعت في الطابق الثالث وتنتشر بسرورة. يرجى إرسال رجال الإطفاء فوراً", None, None, None],
    ["السلام عليكم، أنا أتصل لأبلغ عن حادثة تحرش حدثت لي في مترو الأنفاق بمحطة المرج. كان هناك رجل يلمسني بشكل غير لائق ويحاول مضايقتي. يرجى إرسال دورية أمنية للموقع فوراً", None, None, None],
    ["السلام عليكم، أنا أتصل لأبلغ عن سرقة تعرضت لها قبل قليل. شخصان قاما بسلب هاتفي الجوال ومحفظتي تحت تهديد السلاح. حدث ذلك في حي النزهة بالقرب من متجر السوبر ماركت. أرجو إرسال دورية أمنية وفتح تحقيق في الواقعة", None, None, None],
]

iface = gr.Interface(
    fn=process_inputs,
    inputs=[text_input, image_input, video_input, audio_input],
    outputs=output_text,
    title="<img src='https://i.imgur.com/Qakrqvn.png' width='100' height='100'> منصة استجابة",
    description="تحديد نوع المخاطر والحالات الطارئة تلقائيا باستخدام الذكاء الاصطناعي,\nشبيها بتطبيق 'كلنا امن' بامكانك رفع نص او صور اومقطع او صوت وسيتم تحديد نوع الحالة والجهات المسؤولة عنها",
    examples=examples,
    cache_examples=True,
)
# image = gr.Image("logo.png", style=(100, 100))
# iface.add(image)

# "text-align: right;"

# print('http://127.0.0.1:7860/?__theme=light')
iface.launch(
    share=True,
    favicon_path='logo.png'
)
