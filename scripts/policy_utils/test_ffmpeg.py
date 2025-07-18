import av

codec_name = 'hevc_nvenc' # or 'hevc_nvenc'

try:
    codec = av.Codec(codec_name, 'w') # 'w' for encoder
    print(f"Codec: {codec.name}")
    print(f"Long name: {codec.long_name}")
    print(f"Type: {codec.type}")

    # Supported pixel formats for this encoder
    print(f"Supported pixel formats: {codec.video_formats}")

    # Some codecs might expose more properties,
    # but the above are the most commonly relevant for this issue.

except av.AVError as e:
    print(f"Could not find or open codec {codec_name}: {e}")