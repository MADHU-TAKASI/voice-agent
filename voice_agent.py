import os
import asyncio
import time
from openpyxl import Workbook
from dotenv import load_dotenv
from livekit import rtc, agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

class MetricsLogger:
    def __init__(self):
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.append(["Timestamp", "EOU-delay (ms)", "TTT (ms)", "TTRB (ms)", "Total Latency (ms)"])
        self._call_start_time = None

    def log_metrics(self, eou_delay, ttt, ttrb, latency):
        self.ws.append([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(eou_delay * 1000, 2),
            round(ttt * 1000, 2),
            round(ttrb * 1000, 2),
            round(latency * 1000, 2)
        ])

    def save(self, filename="call_metrics.xlsx"):
        self.wb.save(filename)

class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful, friendly AI assistant. Respond conversationally and keep answers concise."
        )
        self._active_session = None
        self._metrics = MetricsLogger()
        self._last_user_speech_time = 0
        self._response_start_time = 0

    async def on_track_published(self, track, publication):
        if self._active_session and track.kind == agents.TrackKind.KIND_AUDIO:
            async for event in self._active_session.stt.stream_transcribe(track):
                if event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                    user_speech_time = time.time()
                    eou_delay = user_speech_time - self._last_user_speech_time if self._last_user_speech_time else 0
                    self._last_user_speech_time = user_speech_time

                    user_text = event.alternatives[0].text
                    print(f"User said: {user_text}")

                    if self._active_session.tts.is_speaking():
                        await self._active_session.tts.interrupt()

                    ttrb_start = time.time()
                    response = await self._active_session.llm.chat(
                        messages=[
                            {"role": "system", "content": self.instructions},
                            {"role": "user", "content": user_text}
                        ],
                        max_tokens=200
                    )
                    ttrb = time.time() - ttrb_start

                    self._response_start_time = time.time()
                    await self._active_session.tts.synthesize(
                        text=response.choices[0].message.content
                    )
                    ttt = time.time() - self._response_start_time

                    total_latency = time.time() - user_speech_time
                    self._metrics.log_metrics(
                        eou_delay=eou_delay,
                        ttt=ttt,
                        ttrb=ttrb,
                        latency=total_latency
                    )
                    print(f"Metrics - Latency: {total_latency:.2f}s")

async def entrypoint(ctx):
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            api_key=os.getenv("DEEPGRAM_API_KEY")
        ),
        llm=openai.LLM(
            model="gpt-4-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
            api_key=os.getenv("CARTESIA_API_KEY")
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    assistant = VoiceAssistant()
    assistant._active_session = session

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await session.tts.synthesize(
        text="Hello! I'm your AI assistant. How can I help you today?"
    )

    await ctx.connect()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        assistant._metrics.save()

if __name__ == "__main__":
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))