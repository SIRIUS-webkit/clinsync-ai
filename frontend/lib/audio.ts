function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWav(audioBuffer: AudioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const samples = audioBuffer.length;
  const buffer = new ArrayBuffer(44 + samples * numChannels * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples * numChannels * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples * numChannels * 2, true);

  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const sample = audioBuffer.getChannelData(channel)[i];
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
      offset += 2;
    }
  }

  return buffer;
}

export async function convertToWavFile(blob: Blob, filename: string) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  void audioContext.close();
  const wavBuffer = encodeWav(audioBuffer);
  return new File([wavBuffer], filename, { type: "audio/wav" });
}
