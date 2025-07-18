import numpy as np
import av
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from policy_utils.shared_memory_queue import SharedMemoryQueue, Empty
from policy_utils.timestamp_accumulator import get_accumulate_timestamp_idxs
import os

import av.logging
av.logging.set_level(av.logging.VERBOSE)

class VideoRecorder(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    class Command(enum.Enum):
        START_RECORDING = 0
        STOP_RECORDING = 1
    
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        buffer_size=128,
        no_repeat=False,
        # options for codec
        **kwargs     
    ):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.buffer_size = buffer_size
        self.no_repeat = no_repeat
        self.kwargs = kwargs
        
        self.img_queue = None
        self.cmd_queue = None
        self.stop_event = None
        self.ready_event = None
        self.is_started = False
        self.shape = None
        
        self._reset_state()
        
    # ======== custom constructors =======
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj
    
    @classmethod
    def create_hevc_nvenc(cls,
            fps,
            codec='hevc_nvenc',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            bit_rate=6000*1000,
            options={
                'tune': 'll', 
                'preset': 'll'
            },
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            bit_rate=bit_rate,
            options=options,
            **kwargs
        )
        return obj

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, shm_manager:SharedMemoryManager, data_example: np.ndarray):
        super().__init__()
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.img_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                'img': data_example,    
                'repeat': 1
            },
            buffer_size=self.buffer_size
        )
        self.cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                'cmd': self.Command.START_RECORDING.value,
                'video_path': np.array('a'*self.MAX_PATH_LENGTH)
                # 'video_path': np.array(b'\x00' * self.MAX_PATH_LENGTH, dtype='S{}'.format(self.MAX_PATH_LENGTH))
            },
            buffer_size=self.buffer_size
        )
        self.shape = data_example.shape
        self.is_started = True
        super().start()
    
    def stop(self):
        self.stop_event.set()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()
        
    def is_ready(self):
        return (self.start_time is not None) \
            and (self.ready_event.is_set()) \
            and (not self.stop_event.is_set())
        
    def start_recording(self, video_path: str, start_time: float=-1):
        # path_len = len(video_path.encode('utf-8'))
        # if path_len > self.MAX_PATH_LENGTH:
        #     raise RuntimeError('video_path too long.')
        # self.start_time = start_time
        # self.cmd_queue.put({
        #     'cmd': self.Command.START_RECORDING.value,
        #     'video_path': video_path
        # })
        pass
    
    def stop_recording(self):
        self.cmd_queue.put({
            'cmd': self.Command.STOP_RECORDING.value
        })
        self._reset_state()
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
            
        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)
        
        self.img_queue.put({
            'img': img,
            'repeat': n_repeats
        })
    
    def get_img_buffer(self):
        """
        Get view to the next img queue memory
        for zero-copy writing
        """
        data = self.img_queue.get_next_view()
        img = data['img']
        return img
    
    def write_img_buffer(self, img: np.ndarray, frame_time=None):
        """
        Must be used with the buffer returned by get_img_buffer
        for zero-copy writing
        """
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
            
        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)
            # n_repeats = max(1, len(local_idxs)) if (local_idxs is not None) else 1

        
        self.img_queue.put_next_view({
            'img': img,
            'repeat': n_repeats
        })

    # ========= interval API ===========
    def _reset_state(self):
        self.start_time = None
        self.next_global_idx = 0
    
    def run(self):
        # I'm sorry it has to be this complicated...
        self.ready_event.set()
        while not self.stop_event.is_set():
            video_path = None
            # ========= stopped state ============
            while (video_path is None) and (not self.stop_event.is_set()):
                try:
                    commands = self.cmd_queue.get_all()
                    for i in range(len(commands['cmd'])):
                        cmd = commands['cmd'][i]
                        print("Commands", commands) 
                        if cmd == self.Command.START_RECORDING.value:
                                # raw_path = str(commands['video_path'][i])
                                # if ".mp4" in raw_path:
                                #     video_path = raw_path[:raw_path.index(".mp4")+4]
                                # else:
                                #     video_path = raw_path
                                # print("Cleaned video path:", video_path)
                            video_path = str(commands['video_path'][i])
                            print("Video path", video_path)
                            # video_path = commands['video_path'][i].decode('utf-8').rstrip('\x00')
                        elif cmd == self.Command.STOP_RECORDING.value:
                            video_path = None
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                except Empty:
                    time.sleep(0.1/self.fps)
            if self.stop_event.is_set():
                break
            assert video_path is not None
            # if video_path.count('.mp4') > 1:
            #     # Keep only up to the first ".mp4"
            #     video_path = video_path[:video_path.find('.mp4') + 4]


            # ========= recording state ==========
            with av.open(video_path, mode='w') as container:
                print("What about now?", os.path.exists(video_path))
                # stream = container.add_stream(self.codec, rate=self.fps, options=self.kwargs.get('options', {}))
                stream = container.add_stream(self.codec, rate=self.fps)

                # print("Stream", stream)

                h,w,c = self.shape

                # print("Video shape", h, w, c)

                # # Ensure even dimensions (required for NVENC)
                # if w % 2 != 0 or h % 2 != 0:
                #     raise ValueError(f"NVENC requires even dimensions, got {w}x{h}")
                
                stream.width = w
                stream.height = h
                codec_context = stream.codec_context
                # # valid_codec_attrs = dir(codec_context)
                # # for k, v in self.kwargs.items():
                # #     if k != 'options' and k in valid_codec_attrs:
                # #         setattr(codec_context, k, v)
                valid_codec_attrs = dir(codec_context)
                for k, v in self.kwargs.items():
                    if k in valid_codec_attrs:
                        setattr(codec_context, k, v)

                # stream = container.add_stream(self.codec, rate=self.fps, options=self.kwargs.get('options', {}))
                # h,w,c = self.shape

                # stream.width = w
                # stream.height = h

                # # Set the intended output pixel format for the stream's codec context
                # # This is crucial for many codecs, especially h264/hevc which often prefer yuv420p
                # stream.codec_context.pix_fmt = self.kwargs.get('pix_fmt', 'yuv420p')

                # print(f"Stream's codec context pixel format (expected for encoding): {stream.codec_context.pix_fmt}")
                # print(f"Stream's codec context width: {stream.codec_context.width}")
                # print(f"Stream's codec context height: {stream.codec_context.height}")

                # stream.pix_fmt = self.kwargs.get('pix_fmt', 'yuv420p')

                # # Only set known safe attributes
                # if 'bit_rate' in self.kwargs:
                #     stream.bit_rate = self.kwargs['bit_rate']

                
                # loop
                while not self.stop_event.is_set():
                    try:
                        command = self.cmd_queue.get()
                        cmd = int(command['cmd'])
                        if cmd == self.Command.STOP_RECORDING.value:
                            break
                        elif cmd == self.Command.START_RECORDING.value:
                            continue
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                    except Empty:
                        pass
                    
                    try:
                        with self.img_queue.get_view() as data:
                            img = data['img']
                            repeat = data['repeat']
                            # print("Image", img.shape)
                            # print("repeat", repeat)
                            # frame = av.VideoFrame.from_ndarray(
                            #     img, format=self.input_pix_fmt)
                            # print("Frame", frame)
                            # frame = frame.reformat(format='yuv420p')
                            # print("Input format", self.input_pix_fmt)
                            frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
                            # print("Stream codec context pix_fmt", stream.codec_context.pix_fmt)
                            # if stream.codec_context.pix_fmt != self.input_pix_fmt:
                            #     frame = frame.reformat(format=stream.codec_context.pix_fmt)
                        # if not hasattr(self, "next_pts"):
                        #     self.next_pts = 0
                        for _ in range(repeat):
                            # frame.pts = self.next_pts
                            # self.next_pts += 1
                            for packet in stream.encode(frame):
                                # print(f"Packet info: {packet}")
                                container.mux(packet)

                            # print("Encoded")
                            # try:
                            #     container.mux(packet)
                            # except Exception as e:
                            #     print("=== ERROR DURING MUXING ===")
                            #     print(f"Packet info: {packet}")
                            #     print(f"Frame info: {frame.width}x{frame.height}, format: {frame.format.name}")
                            #     traceback.print_exc()  # Full traceback
                            #     raise
                    except Empty:
                        time.sleep(0.1/self.fps)
                        
                # Flush queue
                try:
                    while not self.img_queue.empty():
                        with self.img_queue.get_view() as data:
                            img = data['img']
                            repeat = data['repeat']
                            frame = av.VideoFrame.from_ndarray(
                                img, format=self.input_pix_fmt)
                            # if stream.codec_context.pix_fmt != self.input_pix_fmt:
                            #     frame = frame.reformat(format=stream.codec_context.pix_fmt)
                            # frame = frame.reformat(format='yuv420p')
                        # if not hasattr(self, "next_pts"):
                        #         self.next_pts = 0
                        for _ in range(repeat):
                            # frame.pts = self.next_pts
                            # self.next_pts += 1
                            for packet in stream.encode(frame):
                                container.mux(packet)
                except Empty:
                    pass

                 # Flush stream
                for packet in stream.encode():
                    container.mux(packet)