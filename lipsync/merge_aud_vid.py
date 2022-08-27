import cv2
from moviepy.editor import *

def merge_audio_video():
    audioclip1 = AudioFileClip("/home/chathushkavi/Downloads/superimpose/output_scene1_audio.mp3")

    clip_1 = VideoFileClip("/home/chathushkavi/Downloads/superimpose/output_video.mp4")

    final_clip = concatenate_videoclips([clip_1])
    final_clip_audio = concatenate_audioclips([audioclip1])
    # final_clip_audio.write_audiofile("/home/chathushkavi/Downloads/merge/final_bahu_anushka.mp3")

    new_audioclip = CompositeAudioClip([final_clip_audio])
    final_clip.audio = new_audioclip
    final_clip.write_videofile("/home/chathushkavi/Downloads/superimpose/mix.mp4")


if __name__ == '__main__':
    merge_audio_video()